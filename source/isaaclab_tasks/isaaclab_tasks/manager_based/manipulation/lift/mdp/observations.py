# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as tv_models
import numpy as np
import hashlib
from collections import OrderedDict
import open3d as o3d
import time

from typing import TYPE_CHECKING, TypedDict, Optional, List, Dict

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors.camera import Camera as CameraSensor
from isaaclab.sim.spawners.sensors.sensors_cfg import PinholeCameraCfg
from isaaclab.utils.math import subtract_frame_transforms, quat_apply, quat_conjugate, transform_points, unproject_depth
from isaaclab.sensors.camera.utils import create_pointcloud_from_depth

from .observations_utils import *

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors.camera import CameraData
    from isaaclab.sensors.frame_transformer import FrameTransformer

def test_i2pmae_features(env, camera_names, workspace_min_bounds=None, workspace_max_bounds=None):
    """
    Test if I2P-MAE feature extraction is working correctly
    
    Args:
        env: ManagerBasedRLEnv
        camera_names: List of camera names to use for point cloud creation
        workspace_min_bounds: Optional workspace bounds for cropping
        workspace_max_bounds: Optional workspace bounds for cropping
    
    Returns:
        None - prints diagnostic information
    """
    import torch
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    import numpy as np
    
    print("Starting I2P-MAE feature extraction test...")
    
    # Get processed point cloud from cameras
    pc = multi_camera_pointclouds_in_robot_frame(
        env,
        camera_names,
        target_num_points=2048,
        workspace_min_bounds=workspace_min_bounds,
        workspace_max_bounds=workspace_max_bounds,
        visualize_debug_pointcloud=True
    )
    
    # Print point cloud statistics
    print(f"Point cloud shape: {pc.shape}")
    print(f"Point cloud min: {pc.min(dim=1)[0]}")
    print(f"Point cloud max: {pc.max(dim=1)[0]}")
    print(f"Point cloud mean: {pc.mean(dim=1)}")
    
    # Extract features using the same logic as in pointcloud_cnn_features
    i2pmae_encoder = _get_or_initialize_encoder(
        encoder_type="i2pmae",
        config_params={
            "cnn_output_features": 384,
            "cnn_weights_path": None,
        },
        device=env.device
    )
    
    with torch.no_grad():
        features = i2pmae_encoder(pc, eval=True)
    
    # Print feature statistics
    print(f"Feature shape: {features.shape}")
    print(f"Feature min: {features.min().item():.4f}")
    print(f"Feature max: {features.max().item():.4f}")
    print(f"Feature mean: {features.mean().item():.4f}")
    print(f"Feature std: {features.std().item():.4f}")
    
    # Additional check - move robot and capture a second point cloud
    print("\nMoving robot and capturing second point cloud...")
    
    # Save original features for comparison
    features_1 = features.detach().cpu().numpy()
    
    # Execute a random action to change the scene
    try:
        # Get information about action space for debugging
        if hasattr(env, 'action_space'):
            if hasattr(env.action_space, 'shape'):
                print(f"Environment action_space.shape: {env.action_space.shape}")
            else:
                print("Environment action_space has no shape attribute")
        else:
            print("Environment has no action_space attribute")
            
        # Skip Gym action space detection and ALWAYS use our known correct shape
        print("Using environment-specific action dimension of 8 (7 joints + 1 gripper)")
        random_actions = torch.randn(env.num_envs, 8, device=env.device)
        env.step(random_actions)
    except Exception as e:
        print(f"Warning: Error executing random action: {e}")
        print("Continuing with test, but point cloud might not change significantly")
    
    # Get new point cloud
    pc_2 = multi_camera_pointclouds_in_robot_frame(
        env,
        camera_names,
        target_num_points=2048,
        workspace_min_bounds=workspace_min_bounds,
        workspace_max_bounds=workspace_max_bounds,
        visualize_debug_pointcloud=True
    )
    
    # Extract features from second point cloud
    with torch.no_grad():
        features_2 = i2pmae_encoder(pc_2, eval=True)
    features_2 = features_2.detach().cpu().numpy()
    
    # Compare feature vectors
    feature_diff = np.abs(features_1 - features_2).mean()
    print(f"Average difference between features: {feature_diff:.4f}")
    
    if feature_diff > 0.01:  # Threshold can be adjusted
        print("✅ Features change when the scene changes - encoder appears to be working!")
    else:
        print("❌ Features are too similar - encoder may not be capturing scene changes.")
    
    # Optional: PCA visualization of features
    try:
        # Stack features for PCA
        all_features = np.vstack([features_1, features_2])
        
        # Apply PCA
        pca = PCA(n_components=2)
        reduced_features = pca.fit_transform(all_features)
        
        # Plot
        plt.figure(figsize=(8, 6))
        plt.scatter(reduced_features[0, 0], reduced_features[0, 1], c='blue', s=100, label='Scene 1')
        plt.scatter(reduced_features[1, 0], reduced_features[1, 1], c='red', s=100, label='Scene 2')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.title('I2P-MAE Feature Visualization')
        plt.legend()
        plt.grid(True)
        plt.savefig('i2pmae_feature_test.png')
        print("Feature visualization saved to 'i2pmae_feature_test.png'")
    except Exception as e:
        print(f"Couldn't create PCA visualization: {e}")
    
    print("\nI2P-MAE feature extraction test complete!")


def multi_camera_pointclouds_in_robot_frame(
    env: ManagerBasedRLEnv,
    camera_names: List[str],
    robot_entity_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    target_num_points: int = 2048,
    workspace_min_bounds: Optional[tuple[float, float, float]] = None, # E.g., (-0.5, -0.5, 0.0) in robot frame
    workspace_max_bounds: Optional[tuple[float, float, float]] = None, # E.g., (0.5, 0.5, 0.5) in robot frame
    visualize_debug_pointcloud: bool = False,
) -> torch.Tensor:
    """
    Generates a fused point cloud from multiple cameras, transformed into the robot's root frame,
    optionally crops it to a workspace, and downsamples it to a fixed number of points.
    
    Args:
        // ... existing args ...
        workspace_min_bounds: Optional minimum (x,y,z) bounds for cropping in robot frame.
        workspace_max_bounds: Optional maximum (x,y,z) bounds for cropping in robot frame.
    
    Returns:
        Tensor of shape (num_envs, target_num_points, 3) containing the fused point cloud.
    """
    batch_size = env.num_envs
    device = env.device
    
    robot: RigidObject = env.scene[robot_entity_cfg.name]
    robot_pos_w = robot.data.root_pos_w[:, :3]
    robot_quat_w = robot.data.root_quat_w
    robot_quat_w_inv = quat_conjugate(robot_quat_w)
    
    # Define environment origins - typically the x,y,z offsets of each env in the world
    # This could be derived from environment information or robot positions
    # For now, assuming environments are laid out in a grid and robot position is a good reference
    env_origins = robot_pos_w.clone()
    env_origins[:, 2] = 0.0  # Zero out the z-component

    all_pcs_robot_frame_list = []
    
    ###################
    # print(f"Robot position: {robot_pos_w[0]}, quaternion: {robot_quat_w[0]}")
    ###################
    
    for cam_name in camera_names:
        if cam_name not in env.scene.keys():
            print(f"Warning: Camera '{cam_name}' not found in scene. Skipping.")
            continue
        
        camera: CameraSensor = env.scene[cam_name]
        
        ###################
        # print(f"Camera '{cam_name}' position: {camera.data.pos_w[0]}, quaternion: {camera.data.quat_w_world[0]}")
        ###################
        
        if "distance_to_camera" not in camera.data.output:
            print(f"Warning: Camera '{cam_name}' does not provide 'distance_to_camera' data. Skipping.")
            continue
        
        depth_data = camera.data.output["distance_to_camera"]
        if depth_data.ndim == 4 and depth_data.shape[-1] == 1:
            depth_data = depth_data.squeeze(-1)

        if not hasattr(camera.data, 'intrinsic_matrices'):
            print(f"Warning: Camera '{cam_name}' does not have 'intrinsic_matrices'. Skipping.")
            continue
        intrinsic_matrix = camera.data.intrinsic_matrices
        
        if not hasattr(camera.cfg, 'spawn') or not hasattr(camera.cfg.spawn, 'clipping_range'):
            print(f"Warning: Camera '{cam_name}' missing clipping_range. Skipping.")
            continue

        valid_depth_mask = torch.isfinite(depth_data)
        filtered_depth = depth_data.clone()
        # Using a large number instead of NaN for create_pointcloud_from_depth if it expects valid numbers
        # or ensure create_pointcloud_from_depth handles NaNs by setting points to (0,0,0) or similar
        # For now, let's assume create_pointcloud_from_depth can handle it or zeros out NaN-resulting points.
        filtered_depth[~valid_depth_mask] = camera.cfg.spawn.clipping_range[1] + 1.0 # Mark invalid depths to be filtered by create_pointcloud or range

        # 1. Get points in camera frame first
        points_camera_frame = unproject_depth(filtered_depth, intrinsic_matrix, is_ortho=False)

        # 2. Convert from computer vision convention to Isaac Sim convention
        points_camera_frame_fixed = torch.stack([
            points_camera_frame[..., 2],   # X_isaac = Z_camera (forward)
            -points_camera_frame[..., 0],  # Y_isaac = -X_camera (left)
            -points_camera_frame[..., 1],  # Z_isaac = -Y_camera (up)
        ], dim=-1)
        
        # 3. Transform to world frame
        cam_world = camera.data.pos_w         # shape (N,3)
        cam_rel   = cam_world - env_origins   # now each will be [1.5, ±1.0, 0.5]

        points_world_frame = transform_points(
            points=points_camera_frame_fixed,  # Use the fixed version with proper convention
            pos=cam_rel, # use relative position to the environment origin
            quat=camera.data.quat_w_world,
        )
        
        # 4. Transform to robot frame
        points_robot_frame = points_world_frame  # (B, H*W, 3)
        
        # 5. Flatten H×W into a list per batch
        points_robot_frame_list = points_robot_frame.view(batch_size, -1, 3)  # (B, H*W, 3)
                
        # Append to collection lists
        all_pcs_robot_frame_list.append(points_robot_frame_list)
        

    # Fuse point clouds from all cameras
    if not all_pcs_robot_frame_list:
        print("Warning: No valid point clouds found from any camera.")
        # Return empty point cloud of the expected shape
        return torch.zeros((batch_size, target_num_points, 3), device=device)
    
    all_pcs_robot_frame = torch.cat(all_pcs_robot_frame_list, dim=1)  # (B, N_total, 3)
    all_valid_masks = torch.ones((batch_size, all_pcs_robot_frame.shape[1]), dtype=torch.bool, device=device)

    # Visualize fused point cloud
    # if visualize_debug_pointcloud:
    #     visualize_pointcloud(
    #         all_pcs_robot_frame,
    #         color=[0.0, 0.0, 1.0],  # Blue for fused points
    #         name="fused_pointcloud",
    #         show_plot=True
    #     )
    
    # Apply workspace cropping if provided
    if workspace_min_bounds is not None and workspace_max_bounds is not None:
        workspace_crop_mask = crop_pointcloud_to_workspace(
            all_pcs_robot_frame,
            all_valid_masks,
            workspace_min_bounds,
            workspace_max_bounds,
        )
        
        # Update valid mask with workspace constraints
        all_valid_masks = workspace_crop_mask
        valid_points_count = all_valid_masks.sum(dim=-1, keepdim=True)
        # print(f"Valid points after workspace crop: {valid_points_count.squeeze()}")
    
    # Create cleaned point cloud - replace invalid points with zeros for easier filtering
    cleaned_pc = torch.where(
        all_valid_masks.unsqueeze(-1), 
        all_pcs_robot_frame,
        torch.zeros_like(all_pcs_robot_frame)
    )
    
    # Visualize intermediate result after cropping
    # if visualize_debug_pointcloud:
    #     visualize_pointcloud(
    #         cleaned_pc,
    #         color=[1.0, 0.0, 0.0],  # Red for cropped points
    #         name="cropped_pointcloud",
    #         show_plot=True
    #     )
    
    # return cleaned_pc # Uncomment if you want to return full point cloud
    
    # Initialize tensor to hold downsampled point clouds
    downsampled_pcs = torch.zeros((batch_size, target_num_points, 3), device=device)
    
    # Process each batch item separately
    for b in range(batch_size):
        # Get valid points for this batch
        valid_indices = torch.where(all_valid_masks[b])[0]
        num_valid = valid_indices.shape[0]
        
        if num_valid == 0:
            # No valid points for this batch item
            continue
            
        # Get valid points
        valid_points = all_pcs_robot_frame[b, valid_indices]
        
        if num_valid <= target_num_points:
            # If we have fewer valid points than target, pad with zeros
            downsampled_pcs[b, :num_valid] = valid_points
        else:
            # If we have more points than target, downsample
            # Option 1: Simple random sampling
            random_indices = torch.randperm(num_valid, device=device)[:target_num_points]
            downsampled_pcs[b] = valid_points[random_indices]
            
            # Option 2: Could implement FPS (Furthest Point Sampling) for better coverage
            # This is more complex and would require additional implementation
    
    # Visualize final downsampled result
    # if visualize_debug_pointcloud:  
    #     visualize_pointcloud(
    #         downsampled_pcs,
    #         color=[0.0, 1.0, 0.0],  # Green for downsampled points
    #         name="downsampled_pointcloud",
    #         show_plot=True
    #     )

    # save_pointcloud_to_file(downsampled_pcs, "downsampled_pointcloud.ply", color=[0.0, 1.0, 0.0])

    return downsampled_pcs


# # Define a TypedDict for better type checking of the cache
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
        cache["config_params"] != config_params): 
        reinitialize = True
    
    if reinitialize:
        print(f"Initializing '{encoder_type}' encoder (Config: {config_params}, Device: {device})...")
        model_to_cache: nn.Module

        if encoder_type == "i2pmae":
            model_to_cache = I2PMAEEncoder(
                output_features=config_params.get("cnn_output_features", 384)
            ).to(device)
            weights_path = config_params.get("cnn_weights_path")
            if weights_path:
                try:
                    model_to_cache.load_state_dict(torch.load(weights_path, map_location=device))
                    print(f"Successfully loaded weights for I2PMAEEncoder from {weights_path}")
                except Exception as e:
                    print(f"Warning: Error loading I2PMAEEncoder weights from {weights_path}: {e}.")
        
        elif encoder_type == "static3d":
            model_to_cache = Static3DCNNEncoder(
                input_channels=config_params["input_channels"],
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
                model_name=config_params["resnet_model_name"],
                pretrained=config_params["resnet_pretrained"],
                num_input_voxel_channels=config_params["input_channels"],
                cnn_output_features=config_params["cnn_output_features"]
            ).to(device)
            weights_path = config_params.get("cnn_weights_path")
            if weights_path:
                try:
                    model_to_cache.load_state_dict(torch.load(weights_path, map_location=device))
                    print(f"Successfully loaded weights for ResNet2DEncoder from {weights_path}")
                except Exception as e:
                    print(f"Warning: Error loading ResNet2DEncoder weights from {weights_path}: {e}.")
            
        elif encoder_type == "pvcnn":
            model_to_cache = PVCNNEncoder(
                output_features=config_params["cnn_output_features"],
                input_channels=config_params.get("input_channels", 0),  # Extra features beyond XYZ
                num_points=config_params.get("num_points", 2048)  # Point cloud size
            ).to(device)
            
            weights_path = config_params.get("cnn_weights_path")
            if weights_path:
                try:
                    model_to_cache.load_state_dict(torch.load(weights_path, map_location=device))
                    print(f"Successfully loaded weights for PVCNNEncoder from {weights_path}")
                except Exception as e:
                    print(f"Warning: Error loading PVCNNEncoder weights from {weights_path}: {e}.")
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


# Global cache for Open3D visualizer objects
_global_o3d_visualizer_cache = {
    "voxel_vis": None,          # o3d.visualization.Visualizer object for voxels
    "voxel_geom": None,         # The current o3d.geometry.VoxelGrid object
    "voxel_coord_frame": None,  # Coordinate frame geometry
    "is_voxel_initialized": False
    # Add similar entries if you want to make pointcloud visualization non-blocking too
}



def pointcloud_cnn_features(
    env: ManagerBasedRLEnv,
    camera_names: List[str],
    robot_entity_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    target_num_points: int = 2048,
    workspace_min_bounds: Optional[tuple[float, float, float]] = None,
    workspace_max_bounds: Optional[tuple[float, float, float]] = None,
    voxel_size: tuple[float, float, float] | float = 0.1,
    grid_range_min: tuple[float, float, float] = (-1.0, -1.0, 0.0),
    grid_range_max: tuple[float, float, float] = (1.0, 1.0, 1.0),
    voxel_mode: str = "binary",
    encoder_type: str = "i2pmae",
    cnn_output_features: int = 384,
    cnn_weights_path: str | None = None,
    use_xyz: bool = True, # This parameter seems unused in the current context of pointcloud_cnn_features
    input_channels: int = 0, # For PVCNN: extra features; For Voxel Encoders: channels of voxel grid (usually 1)
    resnet_model_name: str = "resnet18",
    resnet_pretrained: bool = True,
    visualize_debug_pointcloud: bool = False,
    visualize_voxel_grid_debug: bool = False,
    debug_voxel_face_color_rgb: List[float] = [0.0, 0.7, 0.2],
    debug_voxel_edge_color_str: Optional[str] = None, # e.g., 'k' for black edges
    debug_voxel_alpha: float = 0.3,
    debug_voxel_threshold: float = 0.0,
    debug_time: bool = False,
) -> torch.Tensor:
    """
    Processes camera data into a point cloud, crops, downsamples, voxelizes, 
    and extracts features using a configurable encoder.
    """
    # measure time
    start_time = time.time()
    # 1. Get fused, (optionally) cropped, and downsampled point cloud in robot frame
    processed_pc_robot_frame = multi_camera_pointclouds_in_robot_frame(
        env, 
        camera_names, 
        robot_entity_cfg, 
        target_num_points,
        workspace_min_bounds=workspace_min_bounds,
        workspace_max_bounds=workspace_max_bounds,
        visualize_debug_pointcloud=visualize_debug_pointcloud,
    )
    
    stop_time = time.time()
    if debug_time:
        print(f"Time taken to process point cloud: {stop_time - start_time:.4f} seconds")
    
    start_time = time.time()
    # 2. Process according to encoder type
    encoder_config_params = {
        "cnn_output_features": cnn_output_features,
        "cnn_weights_path": cnn_weights_path,
    }

    if encoder_type == "i2pmae":
        # test_i2pmae_features(env, camera_names, workspace_min_bounds, workspace_max_bounds)
        
        encoder = _get_or_initialize_encoder(
            encoder_type=encoder_type,
            config_params=encoder_config_params,
            device=env.device
        )
        
        with torch.no_grad():
            features = encoder(processed_pc_robot_frame, eval=True)
    elif encoder_type in ["pvcnn"]:
        encoder_config_params["input_channels"] = input_channels # Typically 0 if only XYZ, 3 if XYZRGB
        encoder_config_params["num_points"] = target_num_points
        
        encoder = _get_or_initialize_encoder(
            encoder_type=encoder_type,
            config_params=encoder_config_params,
            device=env.device
        )
        
        # PVCNN might expect (B, 3+C, N) or (B, N, 3+C). Your PVCNNEncoder wrapper handles this.
        # Ensure processed_pc_robot_frame is (B, N, 3) if input_channels is 0.
        # If you plan to add color, it should be (B, N, 3+C_color).
        pc_input_for_encoder = processed_pc_robot_frame 
        if input_channels > 0 and processed_pc_robot_frame.shape[-1] == 3:
            # This is a placeholder: if you need color, it must be generated and concatenated
            # For now, if input_channels > 0 but only XYZ is present, this will likely error in PVCNN
            # or you'd need to pad with zeros.
            print(f"Warning: PVCNN input_channels is {input_channels} but point cloud only has 3 features (XYZ). Padding with zeros.")
            padding_features = torch.zeros(pc_input_for_encoder.shape[0], pc_input_for_encoder.shape[1], input_channels, device=env.device)
            pc_input_for_encoder = torch.cat((pc_input_for_encoder, padding_features), dim=-1)


        with torch.no_grad():
            features = encoder(pc_input_for_encoder) # Pass (B, N, 3) or (B, N, 3+C)
            
    else: # Voxel-based encoders like static3d, resnet2d
        # Use the efficient vectorized function
        voxel_grid = pointcloud_to_voxel_grid(
            processed_pc_robot_frame,  # This is already (B, target_num_points, 3)
            voxel_size,
            grid_range_min,  # These define the voxelization volume for the 3D CNN
            grid_range_max,
            mode=voxel_mode
        )
        
        # print(f"Voxel grid shape: {voxel_grid.shape}")  # Debugging output

        if visualize_voxel_grid_debug:
            vis_cache = _global_o3d_visualizer_cache
            
            # Initialize visualizer if not already done
            if not vis_cache["is_voxel_initialized"]:
                try:
                    vis_cache["voxel_vis"] = o3d.visualization.Visualizer()
                    vis_cache["voxel_vis"].create_window(
                        window_name="Live Voxel Grid (Non-Blocking)", 
                        width=1024, height=768, visible=True
                    )
                    # Set render options to match previous style
                    render_options = vis_cache["voxel_vis"].get_render_option()
                    # render_options.background_color = np.asarray([0.0, 0.0, 0.0]) # Removed to use default background
                    render_options.mesh_show_wireframe = True
                    render_options.mesh_show_back_face = True # Enabled to match previous style
                    render_options.point_show_normal = False # Consistent with previous call

                    vis_cache["voxel_coord_frame"] = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
                    vis_cache["voxel_vis"].add_geometry(vis_cache["voxel_coord_frame"])
                    vis_cache["is_voxel_initialized"] = True
                except Exception as e:
                    print(f"Error initializing Open3D visualizer: {e}")
                    vis_cache["is_voxel_initialized"] = False # Prevent further attempts if init fails badly

            voxel_visualizer = vis_cache["voxel_vis"]
            
            if voxel_visualizer and vis_cache["is_voxel_initialized"]:
                # Prepare points for Open3D VoxelGrid creation (for the debug environment)
                debug_env_idx = 0
                points_np = processed_pc_robot_frame[debug_env_idx].detach().cpu().numpy()
                non_zero_mask = np.any(points_np != 0, axis=1)
                points_np_filtered = points_np[non_zero_mask]

                new_o3d_voxel_grid = None
                if len(points_np_filtered) > 0:
                    pcd_for_voxelization = o3d.geometry.PointCloud()
                    pcd_for_voxelization.points = o3d.utility.Vector3dVector(points_np_filtered)
                    
                    # Apply debug color to the points before voxelizing
                    # This color will be used by create_from_point_cloud for the voxel faces
                    if debug_voxel_face_color_rgb:
                        num_points_for_voxel = len(points_np_filtered)
                        colors_np = np.tile(np.array(debug_voxel_face_color_rgb), (num_points_for_voxel, 1))
                        pcd_for_voxelization.colors = o3d.utility.Vector3dVector(colors_np)

                    current_o3d_voxel_size = float(voxel_size) if isinstance(voxel_size, (int, float)) else min(voxel_size)
                    
                    new_o3d_voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(
                        pcd_for_voxelization, 
                        voxel_size=current_o3d_voxel_size
                    )

                # Remove old voxel geometry if it exists
                if vis_cache["voxel_geom"] is not None:
                    voxel_visualizer.remove_geometry(vis_cache["voxel_geom"], reset_bounding_box=False)
                    vis_cache["voxel_geom"] = None # Clear the old reference

                if new_o3d_voxel_grid is not None and new_o3d_voxel_grid.has_voxels():
                    voxel_visualizer.add_geometry(new_o3d_voxel_grid, reset_bounding_box=False)
                    vis_cache["voxel_geom"] = new_o3d_voxel_grid # Cache the new geometry
                    voxel_visualizer.update_geometry(vis_cache["voxel_geom"])
                
                # Poll events and update renderer
                # Important: poll_events() returns False if the window is closed.
                if not voxel_visualizer.poll_events():
                    # Window was closed by user, clean up to allow re-creation or stop visualization
                    if vis_cache["voxel_geom"] is not None: # remove current geom if any
                        voxel_visualizer.remove_geometry(vis_cache["voxel_geom"], reset_bounding_box=False)
                    if vis_cache["voxel_coord_frame"] is not None:
                        voxel_visualizer.remove_geometry(vis_cache["voxel_coord_frame"], reset_bounding_box=False)
                    try:
                        voxel_visualizer.destroy_window()
                    except Exception: # Window might already be gone
                        pass
                    vis_cache["voxel_vis"] = None
                    vis_cache["voxel_geom"] = None
                    vis_cache["voxel_coord_frame"] = None
                    vis_cache["is_voxel_initialized"] = False
                else:
                    voxel_visualizer.update_renderer()
            elif vis_cache["is_voxel_initialized"] and not voxel_visualizer : # Visualizer was initialized but window is now gone
                vis_cache["is_voxel_initialized"] = False # Reset to allow re-initialization




        features = voxel_grid.view(voxel_grid.shape[0], -1)

        
        # if encoder_type == "resnet2d":
        #     encoder_config_params["resnet_model_name"] = resnet_model_name
        #     encoder_config_params["resnet_pretrained"] = resnet_pretrained
        #     encoder_config_params["input_channels"] = voxel_grid.shape[1] # Should be 1 for binary/density
        # elif encoder_type == "static3d":
        #     encoder_config_params["input_channels"] = voxel_grid.shape[1] # Should be 1 for binary/density
        # else:
        #     raise ValueError(f"Unsupported voxel-based encoder type: {encoder_type} in pointcloud_cnn_features")

        # encoder = _get_or_initialize_encoder(
        #     encoder_type=encoder_type,
        #     config_params=encoder_config_params,
        #     device=env.device
        # )
        
        # with torch.no_grad():
        #     features = encoder(voxel_grid)
    
    # Final sanitization of output features
    features = torch.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0) # Or other appropriate values
    stop_time = time.time()
    if debug_time:
        print(f"Time taken to process features: {stop_time - start_time:.4f} seconds")
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



# StandardTwoStreamCNN for direct 2D RGB and Depth image processing
class StandardTwoStreamCNN(nn.Module):
    def __init__(self, rgb_channels=3, depth_channels=1, cnn_stream_output_features=128, final_fused_features=256, resnet_model_name: str = "resnet18"):
        super().__init__()
        
        if not hasattr(tv_models, resnet_model_name):
            raise ValueError(f"ResNet model name '{resnet_model_name}' not found in torchvision.models.")
        
        resnet_constructor = getattr(tv_models, resnet_model_name)
        resnet_weights_enum = getattr(tv_models, f"{resnet_model_name.upper()}_Weights", None)
        default_weights = resnet_weights_enum.DEFAULT if resnet_weights_enum else None

        # RGB Backbone
        rgb_model = resnet_constructor(weights=default_weights)
        if rgb_channels != 3:
            # Modify the first convolutional layer for the specified number of RGB channels
            original_conv1 = rgb_model.conv1
            rgb_model.conv1 = nn.Conv2d(
                rgb_channels, original_conv1.out_channels,
                kernel_size=original_conv1.kernel_size, stride=original_conv1.stride,
                padding=original_conv1.padding, bias=original_conv1.bias
            )
        self.rgb_encoder = nn.Sequential(*list(rgb_model.children())[:-1]) # Remove final FC layer

        # Depth Backbone
        depth_model = resnet_constructor(weights=None) # Initialize without pre-trained weights for depth
        # Modify the first convolutional layer for the specified number of depth channels
        original_conv1_depth = depth_model.conv1
        depth_model.conv1 = nn.Conv2d(
            depth_channels, original_conv1_depth.out_channels,
            kernel_size=original_conv1_depth.kernel_size, stride=original_conv1_depth.stride,
            padding=original_conv1_depth.padding, bias=original_conv1_depth.bias
        )
        self.depth_encoder = nn.Sequential(*list(depth_model.children())[:-1]) # Remove final FC layer

        # Determine the number of output features from the ResNet backbone
        # Create a dummy input to pass through the feature extractor part of ResNet
        with torch.no_grad():
            dummy_input_shape = (1, 3, 224, 224) # Standard ResNet input
            if rgb_channels == 3: # Use RGB model if it has 3 channels, otherwise depth model (assuming it's modified for 3 channels for this test)
                 dummy_output = rgb_model.fc(self.rgb_encoder(torch.zeros(dummy_input_shape)).flatten(1))
            else: # Fallback if rgb_channels is not 3, create a temp 3-channel model for feature size check
                temp_model_for_feat_size = resnet_constructor(weights=None)
                dummy_output = temp_model_for_feat_size.fc(nn.Sequential(*list(temp_model_for_feat_size.children())[:-1])(torch.zeros(dummy_input_shape)).flatten(1))

        num_resnet_output_features = list(rgb_model.children())[-1].in_features


        self.rgb_fc = nn.Linear(num_resnet_output_features, cnn_stream_output_features)
        self.depth_fc = nn.Linear(num_resnet_output_features, cnn_stream_output_features)

        self.fusion_layer = nn.Linear(cnn_stream_output_features * 2, final_fused_features)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, rgb_input: torch.Tensor, depth_input: torch.Tensor):
        # rgb_input: (B, C_rgb, H, W)
        # depth_input: (B, C_depth, H, W)
        
        rgb_features = self.rgb_encoder(rgb_input).flatten(start_dim=1)
        rgb_features = self.relu(self.rgb_fc(rgb_features))

        depth_features = self.depth_encoder(depth_input).flatten(start_dim=1)
        depth_features = self.relu(self.depth_fc(depth_features))

        concatenated_features = torch.cat([rgb_features, depth_features], dim=1)
        fused_output = self.relu(self.fusion_layer(concatenated_features)) # Added ReLU to fusion output
        return fused_output

# Global cache for the StandardTwoStreamCNN model
_global_rgbd_cnn_cache: EncoderCache = {
    "model_type": None,
    "model": None,
    "config_params": None,
    "device": None,
}

def _get_or_initialize_rgbd_cnn(
    config_params: Dict[str, any],
    device: torch.device | str
) -> StandardTwoStreamCNN:
    """Initializes or retrieves a cached StandardTwoStreamCNN model."""
    cache = _global_rgbd_cnn_cache
    model_key = "StandardTwoStreamCNN"

    # Check if reinitialization is needed
    reinitialize = False
    if (cache["model_type"] != model_key or
        str(cache["device"]) != str(device) or
        cache["config_params"] != config_params):
        reinitialize = True

    if reinitialize:
        print(f"Initializing '{model_key}' (Config: {config_params}, Device: {device})...")
        model_to_cache = StandardTwoStreamCNN(
            rgb_channels=config_params.get("rgb_channels", 3),
            depth_channels=config_params.get("depth_channels", 1),
            cnn_stream_output_features=config_params.get("cnn_stream_output_features", 128),
            final_fused_features=config_params.get("final_fused_features", 256),
            resnet_model_name=config_params.get("resnet_model_name", "resnet18")
        ).to(device)

        weights_path = config_params.get("cnn_weights_path")
        if weights_path:
            try:
                model_to_cache.load_state_dict(torch.load(weights_path, map_location=device))
                print(f"Successfully loaded weights for {model_key} from {weights_path}")
            except FileNotFoundError:
                print(f"Warning: Weight file not found for {model_key} at {weights_path}. Using randomly initialized weights.")
            except Exception as e:
                print(f"Warning: Error loading {model_key} weights from {weights_path}: {e}. Using randomly initialized weights.")
        else:
            print(f"No weights_path provided for {model_key}. Using randomly initialized weights.")


        model_to_cache.eval() # Set to evaluation mode
        cache["model_type"] = model_key
        cache["model"] = model_to_cache
        cache["config_params"] = config_params.copy() # Store a copy for future comparisons
        cache["device"] = device
    
    returned_model = cache["model"]
    if not isinstance(returned_model, StandardTwoStreamCNN): # Ensure correct type
        raise RuntimeError(f"{model_key} in cache is not of the correct type or is None after initialization attempt.")
    return returned_model


def multi_camera_rgbd_cnn_features(
    env: ManagerBasedRLEnv,
    camera_names: List[str],
    cnn_stream_output_features: int = 128,
    final_fused_features: int = 256,
    multi_camera_fusion_method: str = "concat", # "concat" or "mean"
    cnn_weights_path: Optional[str] = None,
    resnet_model_name: str = "resnet18",
    depth_scale: float = 1.0, # Optional scaling for depth data
    depth_max_value: Optional[float] = None, # Optional clipping for depth data
) -> torch.Tensor:
    """
    Processes RGB-D data from multiple cameras using a StandardTwoStreamCNN and fuses features.

    Args:
        env: The environment instance.
        camera_names: List of camera entity names.
        cnn_stream_output_features: Number of features from each stream (RGB/Depth) before fusion.
        final_fused_features: Number of features after fusing RGB and Depth streams for one camera.
        multi_camera_fusion_method: How to fuse features from multiple cameras ("concat" or "mean").
        cnn_weights_path: Optional path to pre-trained weights for the StandardTwoStreamCNN.
        resnet_model_name: Name of the ResNet model to use as backbone (e.g., "resnet18", "resnet34").
        depth_scale: Factor to scale depth data by (e.g., if depth is in mm, scale by 0.001 to get meters).
        depth_max_value: Maximum value to clip depth data to before normalization/scaling.

    Returns:
        A tensor containing the fused features from all cameras.
    """
    batch_size = env.num_envs
    device = env.device

    cnn_config_params = {
        "rgb_channels": 3,
        "depth_channels": 1,
        "cnn_stream_output_features": cnn_stream_output_features,
        "final_fused_features": final_fused_features,
        "cnn_weights_path": cnn_weights_path,
        "resnet_model_name": resnet_model_name,
    }
    
    two_stream_cnn_model = _get_or_initialize_rgbd_cnn(cnn_config_params, device)

    all_camera_features_list: List[torch.Tensor] = []

    for cam_name in camera_names:
        if cam_name not in env.scene.keys():
            print(f"Warning: Camera '{cam_name}' not found in scene. Skipping.")
            continue
        
        camera_entity: CameraSensor = env.scene[cam_name] # Type hint for clarity

        # 1. Get RGB data
        if "rgb" not in camera_entity.data.output:
            print(f"Warning: Camera '{cam_name}' does not provide 'rgb' data. Skipping.")
            continue
        rgb_data = camera_entity.data.output["rgb"]  # Expected shape: (B, H, W, 3)
        # Permute to (B, 3, H, W) and normalize to [0, 1]
        rgb_data_processed = rgb_data.permute(0, 3, 1, 2).contiguous().float() / 255.0

        # 2. Get Depth data
        # Prefer 'distance_to_camera' for true depth, fallback to 'distance_to_image_plane'
        depth_key = None
        if "distance_to_camera" in camera_entity.data.output:
            depth_key = "distance_to_camera"
        elif "distance_to_image_plane" in camera_entity.data.output:
            depth_key = "distance_to_image_plane"
            print(f"Warning: Camera '{cam_name}' using 'distance_to_image_plane' for depth. "
                  f"'distance_to_camera' is preferred for true depth.")
        
        if depth_key is None:
            print(f"Warning: Camera '{cam_name}' does not provide recognized depth data ('distance_to_camera' or 'distance_to_image_plane'). Skipping.")
            continue
        
        depth_data = camera_entity.data.output[depth_key] # Expected shape: (B, H, W, 1) or (B, H, W)
        
        if depth_data.ndim == 3: # If shape is (B, H, W), add channel dimension
            depth_data = depth_data.unsqueeze(-1) # Shape: (B, H, W, 1)
        
        # Permute to (B, 1, H, W)
        depth_data_processed = depth_data.permute(0, 3, 1, 2).contiguous().float()

        # Optional: Clip depth data
        if depth_max_value is not None:
            depth_data_processed = torch.clamp(depth_data_processed, max=depth_max_value)
        
        # Optional: Scale depth data (e.g., convert mm to m, or normalize)
        depth_data_processed = depth_data_processed * depth_scale
        
        # Ensure RGB and Depth images have the same H, W dimensions for the CNN
        if rgb_data_processed.shape[2:] != depth_data_processed.shape[2:]:
            print(f"Warning: Camera '{cam_name}' RGB shape {rgb_data_processed.shape} and Depth shape {depth_data_processed.shape} "
                  f"have mismatched H, W dimensions. Skipping. Consider resizing/padding.")
            continue

        # 3. Pass through TwoStreamCNN
        with torch.no_grad(): # Ensure no gradients are computed during inference
            features_this_camera = two_stream_cnn_model(rgb_data_processed, depth_data_processed) # (B, final_fused_features)
        all_camera_features_list.append(features_this_camera)

    if not all_camera_features_list:
        print("Warning: No camera features were processed from any camera.")
        # Determine expected output size for zeros tensor
        # If concat, it's num_cameras * final_fused_features. If mean, it's final_fused_features.
        # This is tricky if camera_names is dynamic or all fail.
        # For robustness, return a tensor of zeros with the 'final_fused_features' dim,
        # assuming the policy might expect at least that base feature size or handle variable input.
        # A more robust solution would be to ensure a fixed number of cameras or pad.
        num_expected_cameras = len(camera_names) if len(camera_names) > 0 else 1 # Avoid 0 multiplier
        if multi_camera_fusion_method == "concat":
            output_dim = num_expected_cameras * final_fused_features
        else: # "mean" or other fixed-size fusion
            output_dim = final_fused_features
        return torch.zeros(batch_size, output_dim, device=device, dtype=torch.float32)

    # 4. Fuse features from multiple cameras
    if len(all_camera_features_list) == 1: # Single camera, no fusion needed beyond its own streams
        fused_all_cameras = all_camera_features_list[0]
    elif multi_camera_fusion_method == "concat":
        fused_all_cameras = torch.cat(all_camera_features_list, dim=1) # (B, num_processed_cameras * final_fused_features)
    elif multi_camera_fusion_method == "mean":
        # Stack to (num_processed_cameras, B, final_fused_features) then mean over dim 0, or (B, num_processed_cameras, final_fused_features) then mean over dim 1
        stacked_features = torch.stack(all_camera_features_list, dim=1) # (B, num_processed_cameras, final_fused_features)
        fused_all_cameras = torch.mean(stacked_features, dim=1) # (B, final_fused_features)
    else:
        raise ValueError(f"Unsupported multi_camera_fusion_method: {multi_camera_fusion_method}")

    return fused_all_cameras
    



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



class I2PMAEEncoder(nn.Module):
    def __init__(self, output_features=384):
        """
        I2P-MAE encoder wrapper
        
        Args:
            output_features: Size of the output feature vector (384 is default for I2P-MAE)
        """
        super().__init__()
        
        # Add the project root to sys.path before importing from I2PMAE
        import sys
        import pathlib
        
        # Determine the root directory of the I2P-MAE project
        project_root = pathlib.Path('/home/chris/IsaacLab')  # Adjust as needed
        
        # Add project root to sys.path if not already there
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
            
        # Now import I2P-MAE modules
        from I2PMAE.models.I2P_MAE import I2P_MAE
        from I2PMAE.utils import config
        
        i2p_mae_project_root = project_root / "I2PMAE"
        
        # Load configuration
        cfg_yaml_path = i2p_mae_project_root / "cfgs" / "pre-training" / "i2p-mae.yaml"
        cfg = config.cfg_from_yaml_file(str(cfg_yaml_path), str(i2p_mae_project_root))
        
        # Initialize model
        self.model = I2P_MAE(cfg.model)
        
        # Load pre-trained weights
        ckpt_path = i2p_mae_project_root / "ckpts" / "pre-train.pth"
        checkpoint = torch.load(str(ckpt_path), map_location='cpu')
        if 'model' in checkpoint:
            self.model.load_state_dict(checkpoint['model'], strict=False)
        else:
            self.model.load_state_dict(checkpoint, strict=False)
        
        self.model.eval()
    
    def forward(self, pointcloud, eval=False):
        """
        Forward pass through the network
        
        Args:
            pointcloud: (B, N, 3) tensor with point coordinates
            eval: Whether to run in evaluation mode (passed to underlying model)
        
        Returns:
            (B, 384) tensor of point cloud features
        """
        # Ensure model is in eval mode
        self.model.eval()
        
        # Forward pass with eval flag
        with torch.no_grad():
            features = self.model(pointcloud, eval=eval)
                
        return features

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

class PVCNNEncoder(nn.Module):
    def __init__(self, output_features=256, input_channels=0, num_points=2048):
        """
        PVCNN encoder using Open3D-ML's implementation
        
        Args:
            output_features: Size of the output feature vector
            input_channels: Number of extra feature channels beyond XYZ (e.g., RGB=3)
            num_points: Number of points the model expects
        """
        super().__init__()
        
        # Import inside the class to avoid global import issues
        import open3d.ml.torch as ml
        
        # Calculate total features (XYZ=3 + extra channels)
        total_channels = 3 + input_channels
        
        # Create the PVCNN model
        self.backbone = ml.models.PVCNN(
            name="PVCNN",
            num_classes=output_features,  # Will use the classifier output as features
            num_points=num_points,
            extra_feature_channels=input_channels
        )
        
        # Replace the final classifier if needed
        if hasattr(self.backbone, 'classifier') and output_features != self.backbone.classifier.out_channels:
            self.backbone.classifier = nn.Sequential(
                nn.Linear(
                    self.backbone.classifier.in_channels, 
                    output_features
                )
            )
        
    def forward(self, pointcloud):
        """
        Forward pass through the network
        
        Args:
            pointcloud: (B, N, 3+C) tensor with point coordinates and optional features
        
        Returns:
            (B, output_features) tensor of point cloud features
        """
        batch_size = pointcloud.shape[0]
        
        # PVCNN expects points in format (B, N, 3+C)
        # Check if we need to reshape
        if pointcloud.shape[-1] >= 3:
            # Already in correct shape
            points = pointcloud
        else:
            # Need to transpose
            points = pointcloud.transpose(1, 2)
            
        # Extract features (without actual class prediction)
        with torch.no_grad():
            # Get features before the final classifier layer
            features = self.backbone.extract_features(points)
            
            # Use global pooling to get a single feature vector per batch
            if features.dim() > 2:
                features = torch.mean(features, dim=-1)  # Global average pooling
            
        return features