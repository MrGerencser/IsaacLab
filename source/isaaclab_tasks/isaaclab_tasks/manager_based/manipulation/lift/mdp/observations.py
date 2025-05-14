# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as tv_models


from typing import TYPE_CHECKING, TypedDict, Optional, List, Dict

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors.camera import Camera as CameraSensor
from isaaclab.sim.spawners.sensors.sensors_cfg import PinholeCameraCfg
from isaaclab.utils.math import subtract_frame_transforms, quat_apply, quat_conjugate

from isaaclab.sensors.camera.utils import create_pointcloud_from_depth


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors.camera import CameraData
    from isaaclab.sensors.frame_transformer import FrameTransformer
    

def multi_camera_pointclouds_in_robot_frame(
    env: ManagerBasedRLEnv,
    camera_names: List[str],
    robot_entity_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    target_num_points: int = 2048,
) -> torch.Tensor:
    """
    Generates a fused point cloud from multiple cameras, transformed into the robot's root frame,
    and downsamples it to a fixed number of points.
    
    Args:
        env: The environment containing the scene.
        camera_names: List of camera entity names to use for point cloud generation.
        robot_entity_cfg: Configuration for the robot entity to use as reference frame.
        target_num_points: Number of points in the final downsampled point cloud.
    
    Returns:
        Tensor of shape (num_envs, target_num_points, 3) containing the fused point cloud.
    """
    batch_size = env.num_envs
    device = env.device
    
    # Get robot pose for transformations
    robot: RigidObject = env.scene[robot_entity_cfg.name]
    robot_pos_w = robot.data.root_pos_w[:, :3]  # Shape: (B, 3)
    robot_quat_w = robot.data.root_quat_w      # Shape: (B, 4)
    robot_quat_w_inv = quat_conjugate(robot_quat_w) # Shape: (B, 4)

    # Lists to store point clouds and validity masks from each camera
    all_pcs_robot_frame_list = []
    all_valid_masks_list = []

    # Process each camera
    for cam_name in camera_names:
        # Validate camera exists in the scene
        if cam_name not in env.scene.keys():
            print(f"Warning: Camera '{cam_name}' not found in scene. Skipping.")
            continue
        
        camera: CameraSensor = env.scene[cam_name]
        
        # Check camera provides depth data
        if "distance_to_camera" not in camera.data.output:
            print(f"Warning: Camera '{cam_name}' does not provide 'distance_to_camera' data. Skipping.")
            continue
        
        # Process depth data
        depth_data = camera.data.output["distance_to_camera"]
        if depth_data.ndim == 4 and depth_data.shape[-1] == 1:
            depth_data = depth_data.squeeze(-1)  # Ensure (B, H, W)

        # Get camera intrinsics
        if not hasattr(camera.data, 'intrinsic_matrices'):
            print(f"Warning: Camera '{cam_name}' does not have 'intrinsic_matrices'. Skipping.")
            continue
        
        intrinsic_matrix = camera.data.intrinsic_matrices  # Shape: (B, 3, 3)
        
        # Get clipping planes
        if not hasattr(camera.cfg, 'spawn') or not hasattr(camera.cfg.spawn, 'clipping_range'):
            print(f"Warning: Camera '{cam_name}' missing clipping_range. Skipping.")
            continue
        
        near_clip, far_clip = camera.cfg.spawn.clipping_range

        # 1. Convert depth to point cloud in camera frame
        # Filter depth values based on clipping planes first
        valid_depth_mask = torch.isfinite(depth_data) & (depth_data >= near_clip) & (depth_data <= far_clip)
        filtered_depth = depth_data.clone()
        filtered_depth[~valid_depth_mask] = float('nan')  # Mark invalid depths as NaN
        
        points_cam_frame = create_pointcloud_from_depth(
            intrinsic_matrix=intrinsic_matrix,
            depth=filtered_depth,
            keep_invalid=False,  # Discard invalid points
            device=env.device
        )
        
        # Create validity mask from non-zero points
        valid_mask_cam = torch.any(points_cam_frame != 0, dim=-1)
        
        # 2. Transform points from camera to world frame
        cam_pos_w = camera.data.pos_w
        cam_quat_w = camera.data.quat_w_world
        
        # Explicitly expand quaternions to match point count for correct broadcasting in quat_apply
        cam_quat_expanded = cam_quat_w.unsqueeze(1).expand(-1, points_cam_frame.shape[1], -1)
        points_world_frame = quat_apply(cam_quat_expanded, points_cam_frame) + cam_pos_w.unsqueeze(1)
        
        # 3. Transform points from world to robot frame
        robot_quat_expanded = robot_quat_w_inv.unsqueeze(1).expand(-1, points_world_frame.shape[1], -1)
        points_robot_frame = quat_apply(
            robot_quat_expanded, 
            points_world_frame - robot_pos_w.unsqueeze(1)
        )
        
        # Zero out invalid points
        points_robot_frame = points_robot_frame * valid_mask_cam.unsqueeze(-1)
        
        # Store results
        all_pcs_robot_frame_list.append(points_robot_frame)
        all_valid_masks_list.append(valid_mask_cam)

    # Handle case where no cameras provided valid data
    if not all_pcs_robot_frame_list:
        return torch.zeros(batch_size, target_num_points, 3, device=device)

    # 4. Fuse point clouds from all cameras
    fused_pc = torch.cat(all_pcs_robot_frame_list, dim=1)  # Shape: (B, N_total, 3)
    fused_mask = torch.cat(all_valid_masks_list, dim=1)    # Shape: (B, N_total)

    # 5. Downsample to target_num_points
    batch_downsampled_pcs = []
    for i in range(batch_size):
        # Extract valid points for this batch item
        valid_points = fused_pc[i, fused_mask[i]]  # Shape: (Num_valid, 3)
        num_valid = valid_points.shape[0]

        if num_valid == 0:
            # No valid points, fill with zeros
            downsampled = torch.zeros(target_num_points, 3, device=device)
        elif num_valid >= target_num_points:
            # Enough points, randomly sample without replacement
            indices = torch.randperm(num_valid, device=device)[:target_num_points]
            downsampled = valid_points[indices]
        else:
            # Not enough points, pad with zeros
            padding = torch.zeros(target_num_points - num_valid, 3, device=device)
            downsampled = torch.cat([valid_points, padding], dim=0)
        
        batch_downsampled_pcs.append(downsampled)
    
    # Stack all batch items into final tensor
    return torch.stack(batch_downsampled_pcs, dim=0)  # Shape: (B, target_num_points, 3)


def pointcloud_to_voxel_grid(
    point_cloud: torch.Tensor,  # Shape: (B, N, 3)
    voxel_size: tuple[float, float, float] | float,
    grid_range_min: tuple[float, float, float],
    grid_range_max: tuple[float, float, float],
    mode: str = "binary",  # "binary" or "density"
) -> torch.Tensor:
    """
    Converts a batch of point clouds to voxel grids.

    Args:
        point_cloud: Batch of point clouds (B, N, 3).
        voxel_size: Size of each voxel (vx, vy, vz) or a single float for isotropic voxels.
        grid_range_min: Minimum coordinates (x_min, y_min, z_min) of the grid.
        grid_range_max: Maximum coordinates (x_max, y_max, z_max) of the grid.
        mode: "binary" for occupancy (0 or 1), "density" for point count per voxel.

    Returns:
        Voxel grid. Shape: (B, 1, Dx, Dy, Dz) for use with Conv3D.
    """
    B, N, _ = point_cloud.shape
    device = point_cloud.device

    if isinstance(voxel_size, float):
        vx, vy, vz = voxel_size, voxel_size, voxel_size
    else:
        vx, vy, vz = voxel_size
    
    if not (vx > 0 and vy > 0 and vz > 0):
        raise ValueError(f"Voxel sizes must be positive. Got: {(vx,vy,vz)}")

    min_x, min_y, min_z = grid_range_min
    max_x, max_y, max_z = grid_range_max

    # Calculate grid dimensions
    Dx = int(torch.ceil(torch.tensor((max_x - min_x) / vx, device=device)).item())
    Dy = int(torch.ceil(torch.tensor((max_y - min_y) / vy, device=device)).item())
    Dz = int(torch.ceil(torch.tensor((max_z - min_z) / vz, device=device)).item())

    if not (Dx > 0 and Dy > 0 and Dz > 0):
        # This can happen if grid_range_min is very close to or greater than grid_range_max
        print(f"Warning: Voxel grid dimensions are not all positive: Dx={Dx}, Dy={Dy}, Dz={Dz}."
              f" Min range: {grid_range_min}, Max range: {grid_range_max}, Voxel size: {(vx,vy,vz)}."
              " Returning empty voxel grid.")
        return torch.zeros((B, 1, 1, 1, 1), dtype=torch.float32, device=device)


    # Normalize points to voxel indices
    normalized_points_x = (point_cloud[..., 0] - min_x) / vx
    normalized_points_y = (point_cloud[..., 1] - min_y) / vy
    normalized_points_z = (point_cloud[..., 2] - min_z) / vz

    indices_x = torch.floor(normalized_points_x).long()
    indices_y = torch.floor(normalized_points_y).long()
    indices_z = torch.floor(normalized_points_z).long()

    batch_indices = torch.arange(B, device=device).view(B, 1).expand(B, N)

    # Filter out-of-bounds points
    valid_mask = (
        (indices_x >= 0) & (indices_x < Dx) &
        (indices_y >= 0) & (indices_y < Dy) &
        (indices_z >= 0) & (indices_z < Dz)
    )

    batch_indices_valid = batch_indices[valid_mask]
    indices_x_valid = indices_x[valid_mask]
    indices_y_valid = indices_y[valid_mask]
    indices_z_valid = indices_z[valid_mask]
    
    # Initialize voxel grid based on mode
    if mode == "binary":
        voxel_grid_data = torch.zeros((B, Dx, Dy, Dz), dtype=torch.bool, device=device)
        voxel_grid_data[batch_indices_valid, indices_x_valid, indices_y_valid, indices_z_valid] = True
        return voxel_grid_data.float().unsqueeze(1)  # (B, 1, Dx, Dy, Dz)
    elif mode == "density":
        voxel_grid_data = torch.zeros((B, Dx, Dy, Dz), dtype=torch.float32, device=device)
        # Combine indices for unique operation: (Num_valid_points, 4)
        combined_indices = torch.stack([
            batch_indices_valid,
            indices_x_valid,
            indices_y_valid,
            indices_z_valid
        ], dim=1)

        if combined_indices.numel() > 0:
            unique_voxels, counts = torch.unique(combined_indices, dim=0, return_counts=True)
            voxel_grid_data[unique_voxels[:, 0], unique_voxels[:, 1], unique_voxels[:, 2], unique_voxels[:, 3]] = counts.float()
        return voxel_grid_data.unsqueeze(1)  # (B, 1, Dx, Dy, Dz)
    else:
        raise ValueError(f"Unsupported voxelization mode: {mode}")


class Static3DCNNEncoder(nn.Module):
    def __init__(self, input_channels=1, cnn_output_features=256):
        super().__init__()
        self.conv1 = nn.Conv3d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.adaptive_pool = nn.AdaptiveAvgPool3d((1, 1, 1)) # Output (B, 128, 1, 1, 1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(128, cnn_output_features) # 128 * 1 * 1 * 1 = 128

    def forward(self, x): # x is the voxel_grid (B, C, Dx, Dy, Dz)
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        x = self.adaptive_pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x
    

class ResNet2DEncoder(nn.Module):
    def __init__(self, model_name: str = "resnet18", pretrained: bool = True, num_input_voxel_channels: int = 1, cnn_output_features: int = 256):
        super().__init__()
        self.model_name = model_name
        self.pretrained = pretrained
        # num_input_voxel_channels is the channel count of the voxel grid (typically 1)
        self.num_input_voxel_channels = num_input_voxel_channels

        if not hasattr(tv_models, model_name):
            raise ValueError(f"ResNet model name '{model_name}' not found in torchvision.models.")

        # Load the specified ResNet model
        # For simplicity, assuming ResNet family. Add more if needed.
        if model_name.startswith("resnet"):
            resnet_model_constructor = getattr(tv_models, model_name)
            # Use new 'weights' parameter for torchvision >= 0.13, otherwise 'pretrained'
            try: # Try new API first
                if pretrained:
                    weights = getattr(tv_models, f"{model_name.upper()}_Weights").DEFAULT
                    resnet_model = resnet_model_constructor(weights=weights)
                else:
                    resnet_model = resnet_model_constructor(weights=None)
            except AttributeError: # Fallback to old API
                resnet_model = resnet_model_constructor(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported model family for '{model_name}'. Example supports ResNet.")

        # Remove the original fully connected layer (and avgpool for some models like ViT, but ResNet usually has adaptive avg pool before fc)
        self.features = nn.Sequential(*list(resnet_model.children())[:-1])

        # Determine the number of output features from the ResNet backbone
        with torch.no_grad():
            # Create a dummy input: (B, C_voxel, Dx, Dy, Dz) -> mean over Dz -> (B, C_voxel, Dx, Dy) -> repeat to 3 channels -> (B, 3, Dx, Dy)
            # Assuming Dx, Dy are somewhat like image dimensions (e.g., 32x32 or more)
            # ResNet typically handles various input sizes due to global average pooling at the end.
            dummy_2d_input_shape = (1, 3, 64, 64) # Example H, W for ResNet after preprocessing
            dummy_output = self.features(torch.zeros(dummy_2d_input_shape))
        
        num_resnet_output_features = dummy_output.view(dummy_output.size(0), -1).shape[1]

        self.fc = nn.Linear(num_resnet_output_features, cnn_output_features)

    def forward(self, x_voxel: torch.Tensor): # Expected x_voxel: (B, C_voxel, Dx, Dy, Dz)
        if not (x_voxel.dim() == 5 and x_voxel.shape[1] == self.num_input_voxel_channels):
            raise ValueError(
                f"Input voxel grid has unexpected shape: {x_voxel.shape}. "
                f"Expected (B, {self.num_input_voxel_channels}, Dx, Dy, Dz)."
            )

        # 1. Reduce Dz dimension (e.g., by averaging) to get (B, C_voxel, Dx, Dy)
        x_2d_intermediate = torch.mean(x_voxel, dim=4)

        # 2. Replicate the single channel to 3 channels if C_voxel is 1
        if self.num_input_voxel_channels == 1:
            x_3channel = x_2d_intermediate.repeat(1, 3, 1, 1) # Now (B, 3, Dx, Dy)
        elif self.num_input_voxel_channels == 3:
            x_3channel = x_2d_intermediate # Already 3 channels
        else:
            raise ValueError(f"Expected 1 or 3 input voxel channels, got {self.num_input_voxel_channels}")

        # 3. Pass through ResNet features
        features = self.features(x_3channel)
        features_flat = features.view(features.size(0), -1) # Flatten

        # 4. Pass through the new FC layer
        output = self.fc(features_flat)
        return output

class PointNetPlusPlusEncoder(nn.Module):
    def __init__(self, output_features=256, input_channels=0, use_xyz=True):
        """
        PointNet++ encoder that processes point clouds directly (no voxelization)
        
        Args:
            output_features: Size of the output feature vector
            input_channels: Number of input features per point (0 means XYZ only)
            use_xyz: Whether to use XYZ coordinates as additional features
        """
        super().__init__()
        
        # Set sampling and grouping parameters
        self.SA_modules = nn.ModuleList()
        
        # # First set abstraction layer (sampling and grouping)
        # self.SA_modules.append(
        #     PointnetSAModule(
        #         npoint=512,
        #         radius=0.2,
        #         nsample=32,
        #         mlp=[input_channels, 64, 64, 128],
        #         use_xyz=use_xyz,
        #     )
        # )
        
        # # Second set abstraction layer
        # self.SA_modules.append(
        #     PointnetSAModule(
        #         npoint=128,
        #         radius=0.4,
        #         nsample=64,
        #         mlp=[128, 128, 128, 256],
        #         use_xyz=use_xyz,
        #     )
        # )
        
        # # Third set abstraction layer (global pooling)
        # self.SA_modules.append(
        #     PointnetSAModule(
        #         npoint=None,  # Global pooling
        #         radius=None,  # Global pooling
        #         nsample=None,  # Global pooling
        #         mlp=[256, 256, 512, 1024],
        #         use_xyz=use_xyz,
        #     )
        # )
        
        # Feature projection to desired output size
        self.fc = nn.Linear(1024, output_features)
        
    def forward(self, pointcloud):
        """
        Forward pass through the network
        
        Args:
            pointcloud: (B, N, 3) tensor containing point coordinates
                       or (B, N, 3+C) if points have additional features
        
        Returns:
            (B, output_features) tensor of point cloud features
        """
        # Prepare points for PointNet++ processing
        # PointNet++ expects (B, C, N) where C is num_features
        if pointcloud.shape[-1] > 3:
            # If there are additional features beyond XYZ
            xyz = pointcloud[:, :, :3].contiguous()
            features = pointcloud[:, :, 3:].transpose(1, 2).contiguous()
        else:
            # XYZ only
            xyz = pointcloud.contiguous()
            features = None
            
        # Process through set abstraction layers
        for module in self.SA_modules:
            xyz, features = module(xyz, features)
            
        # After the last SA module, features will be (B, C, 1)
        # Reshape to (B, C)
        features = features.squeeze(-1)
        
        # Final feature projection
        output = self.fc(features)
        
        return output


# Define a TypedDict for better type checking of the cache
class EncoderCache(TypedDict):
    model_type: Optional[str]
    model: Optional[nn.Module] # More generic type
    config_params: Optional[Dict[str, any]] # Using Dict for params
    device: Optional[torch.device | str]

_global_encoder_cache: EncoderCache = {
    "model_type": None,
    "model": None,
    "config_params": None,
    "device": None
}

def _get_or_initialize_encoder(
    encoder_type: str,
    config_params: Dict[str, any],
    device: torch.device | str
):
    cache = _global_encoder_cache
    
    # More robust change detection based on a copy of config_params
    reinitialize = False
    if (cache["model_type"] != encoder_type or
        str(cache["device"]) != str(device) or
        cache["config_params"] != config_params): # Direct dict comparison
        reinitialize = True
    
    if reinitialize:
        print(f"Initializing '{encoder_type}' encoder (Config: {config_params}, Device: {device})...")
        model_to_cache: nn.Module

        if encoder_type == "static3d":
            model_to_cache = Static3DCNNEncoder(
                input_channels=config_params.get("input_channels", 1),
                cnn_output_features=config_params["cnn_output_features"]
            ).to(device)
            weights_path = config_params.get("cnn_weights_path")
            if weights_path:
                try:
                    model_to_cache.load_state_dict(torch.load(weights_path, map_location=device))
                    print(f"Successfully loaded weights for Static3DCNNEncoder from {weights_path}")
                except Exception as e:
                    print(f"Warning: Error loading Static3DCNNEncoder weights from {weights_path}: {e}.")
        
        elif encoder_type == "resnet2d":
            model_to_cache = ResNet2DEncoder(
                model_name=config_params.get("resnet_model_name", "resnet18"),
                pretrained=config_params.get("resnet_pretrained", True),
                num_input_voxel_channels=config_params.get("input_channels", 1),
                cnn_output_features=config_params["cnn_output_features"]
            ).to(device)
            
            weights_path = config_params.get("cnn_weights_path")
            # If cnn_weights_path is provided for ResNet, it implies custom fine-tuned weights for the whole model,
            # potentially overriding torchvision's pretrained if resnet_pretrained was True (though unusual).
            # Or, if resnet_pretrained is False, this path is essential.
            if weights_path:
                if config_params.get("resnet_pretrained", True):
                    print(f"Warning: 'cnn_weights_path' ({weights_path}) is provided while 'resnet_pretrained' is True. "
                          f"Torchvision's pretrained weights for '{config_params.get('resnet_model_name')}' were loaded first. "
                          f"The weights from 'cnn_weights_path' will now attempt to load, potentially overriding them.")
                try:
                    model_to_cache.load_state_dict(torch.load(weights_path, map_location=device))
                    print(f"Successfully loaded custom weights for ResNet2DEncoder from {weights_path}")
                except Exception as e:
                    print(f"Warning: Error loading ResNet2DEncoder weights from {weights_path}: {e}.")
        elif encoder_type == "pointnet++":
            # New code for PointNet++
            model_to_cache = PointNetPlusPlusEncoder(
                output_features=config_params["cnn_output_features"],
                input_channels=0,  # Only using XYZ coordinates
                use_xyz=True
            ).to(device)
            
            weights_path = config_params.get("cnn_weights_path")
            if weights_path:
                try:
                    model_to_cache.load_state_dict(torch.load(weights_path, map_location=device))
                    print(f"Successfully loaded weights for PointNetPlusPlusEncoder from {weights_path}")
                except Exception as e:
                    print(f"Warning: Error loading PointNetPlusPlus weights from {weights_path}: {e}.")
        else:
            raise ValueError(f"Unsupported encoder_type: {encoder_type}")

        model_to_cache.eval()
        cache["model_type"] = encoder_type
        cache["model"] = model_to_cache
        cache["config_params"] = config_params.copy() # Store a copy for future comparisons
        cache["device"] = device
        
    # Type assertion for clarity, Pylance might still need help if cache["model"] is Optional[nn.Module]
    returned_model = cache["model"]
    if returned_model is None:
        # This case should ideally not be reached if initialization logic is correct
        raise RuntimeError("Encoder model in cache is None after initialization attempt.")
    return returned_model


def pointcloud_cnn_features(
    env: ManagerBasedRLEnv,
    camera_names: List[str],
    robot_entity_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    target_num_points: int = 2048,
    voxel_size: tuple[float, float, float] | float = 0.1,
    grid_range_min: tuple[float, float, float] = (-1.0, -1.0, 0.0),
    grid_range_max: tuple[float, float, float] = (1.0, 1.0, 1.0),
    voxel_mode: str = "binary",
    encoder_type: str = "static3d", 
    cnn_output_features: int = 256,
    cnn_weights_path: str | None = None,
    # PointNet++ specific params
    use_xyz: bool = True,
    # ResNet specific params
    resnet_model_name: str = "resnet18",
    resnet_pretrained: bool = True
) -> torch.Tensor:
    """
    Processes camera data into a point cloud and extracts features using a configurable encoder.
    """
    # 1. Get fused point cloud in robot frame
    fused_pc_robot_frame = multi_camera_pointclouds_in_robot_frame(
        env, camera_names, robot_entity_cfg, target_num_points
    )  # Shape: (B, target_num_points, 3)
    
    # 2. Process according to encoder type
    encoder_config_params = {
        "cnn_output_features": cnn_output_features,
        "cnn_weights_path": cnn_weights_path,
    }
    
    if encoder_type == "pointnet++":
        # No voxelization needed for PointNet++
        encoder_config_params["use_xyz"] = use_xyz
        
        # Get the encoder
        encoder = _get_or_initialize_encoder(
            encoder_type=encoder_type,
            config_params=encoder_config_params,
            device=env.device
        )
        
        # Pass point cloud directly to PointNet++ encoder
        with torch.no_grad():
            features = encoder(fused_pc_robot_frame)
            
    else:
        # Original voxelization approach for other encoder types
        voxel_grid = pointcloud_to_voxel_grid(
            fused_pc_robot_frame, 
            voxel_size, 
            grid_range_min, 
            grid_range_max, 
            mode=voxel_mode
        )
        
        if encoder_type == "resnet2d":
            encoder_config_params["resnet_model_name"] = resnet_model_name
            encoder_config_params["resnet_pretrained"] = resnet_pretrained
            encoder_config_params["input_channels"] = 1
        else:  # static3d
            encoder_config_params["input_channels"] = 1
        
        encoder = _get_or_initialize_encoder(
            encoder_type=encoder_type,
            config_params=encoder_config_params,
            device=env.device
        )
        
        with torch.no_grad():
            features = encoder(voxel_grid)
    
    return features


def object_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """The position of the object in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    object_pos_w = object.data.root_pos_w[:, :3]
    object_pos_b, _ = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], object_pos_w
    )
    return object_pos_b


def object_orientation_in_robot_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """The orientation of the object as a quaternion in the robot's frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    object_quat_w = object.data.root_quat_w
    _, object_quat_b = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], 
        None, object_quat_w
    )
    return object_quat_b



