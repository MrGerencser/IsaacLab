import numpy as np
import torch
import open3d as o3d
from typing import Optional, Dict, Any, Tuple


# Global cache for visualizers to maintain state across function calls
_global_o3d_visualizer_cache: Dict[str, Dict[str, Any]] = {
    "voxel": {"vis": None, "geom": None, "coord_frame": None, "initialized": False, "first_update": True},
    # Each point cloud will have its own entry based on name
}

def visualize_pointcloud(point_cloud, color=[1, 0, 0], name="debug_pointcloud", show_plot=False, 
                         use_live_vis=True):
    """
    Visualizes point cloud using Open3D, either as a static plot or live updating visualization.
    Creates a separate window for each point cloud name.
    
    Args:
        point_cloud: torch.Tensor of shape (B, N, 3) or (N, 3)
        color: RGB color for points [r, g, b] in range [0, 1]
        name: Name for the point cloud (used as window identifier)
        show_plot: Whether to display the visualization
        use_live_vis: Whether to use a live updating visualization window
    """
    if not show_plot:
        return
        
    # Extract points (handle both batched and non-batched)
    if point_cloud.ndim == 3:
        points = point_cloud[0].detach().cpu().numpy()  # Just use first environment
    else:
        points = point_cloud.detach().cpu().numpy()
    
    # Filter out zero points
    points = points[np.any(points != 0, axis=1)]
    
    # Skip if no points
    if len(points) == 0:
        return
    
    # Log information about the point cloud for debugging
    print(f"Point cloud '{name}' (env {0}): {len(points)} points")
    if len(points) > 0:
        mins = np.min(points, axis=0)
        maxs = np.max(points, axis=0)
        print(f"  Range X: [{mins[0]:.3f}, {maxs[0]:.3f}]")
        print(f"  Range Y: [{mins[1]:.3f}, {maxs[1]:.3f}]")
        print(f"  Range Z: [{mins[2]:.3f}, {maxs[2]:.3f}]")
    
    # Use live visualization if requested
    if use_live_vis:
        # Ensure this cloud has an entry in the cache
        if name not in _global_o3d_visualizer_cache:
            _global_o3d_visualizer_cache[name] = {
                "vis": None, "geom": None, "coord_frame": None, "initialized": False, "first_update": True
            }
        
        vis_cache = _global_o3d_visualizer_cache[name]
        
        # Initialize visualizer if not already done
        if not vis_cache["initialized"]:
            try:
                # Use VisualizerWithKeyCallback for better interactivity
                vis_cache["vis"] = o3d.visualization.VisualizerWithKeyCallback()
                vis_cache["vis"].create_window(
                    window_name=f"Point Cloud: {name}", 
                    width=1024, height=768, visible=True
                )
                
                # Add view control configuration with better defaults
                view_control = vis_cache["vis"].get_view_control()
                view_control.set_zoom(0.8)  # Start slightly zoomed out
                view_control.set_front([0, 0, -1])  # Set initial view direction
                view_control.set_lookat([0, 0, 0])  # Look at center
                view_control.set_up([0, 1, 0])  # Y-axis is up
                
                # Set render options for better visualization
                render_options = vis_cache["vis"].get_render_option()
                render_options.point_size = 3.0
                render_options.point_show_normal = False
                render_options.background_color = np.array([0.1, 0.1, 0.1])  # Dark background
                
                # Add coordinate frame
                vis_cache["coord_frame"] = o3d.geometry.TriangleMesh.create_coordinate_frame(
                    size=0.1, origin=[0, 0, 0]
                )
                vis_cache["vis"].add_geometry(vis_cache["coord_frame"])
                
                # Add key callback to show controls
                def show_help(vis):
                    print("\n--- Point Cloud Controls ---")
                    print("Left-click + drag: Rotate")
                    print("Right-click + drag: Translate")
                    print("Mouse wheel: Zoom in/out")
                    print("H: Show this help")
                    return False
                    
                vis_cache["vis"].register_key_callback(ord("H"), show_help)
                
                # Show initial help
                show_help(None)
                
                vis_cache["initialized"] = True
            except Exception as e:
                print(f"Error initializing Open3D point cloud visualizer for '{name}': {e}")
                vis_cache["initialized"] = False
        
        pc_visualizer = vis_cache["vis"]
        
        if pc_visualizer and vis_cache["initialized"]:
            # Save view parameters before updating geometry
            view_control = pc_visualizer.get_view_control()
            if not vis_cache["first_update"]:
                # Only save params if not the first update
                params = view_control.convert_to_pinhole_camera_parameters()
            
            # Create new point cloud geometry
            new_pcd = o3d.geometry.PointCloud()
            new_pcd.points = o3d.utility.Vector3dVector(points)
            
            # Set point colors
            colors = np.tile(np.array(color, dtype=np.float64), (len(points), 1))
            new_pcd.colors = o3d.utility.Vector3dVector(colors)
            
            # Remove old point cloud geometry if it exists
            if vis_cache["geom"] is not None:
                pc_visualizer.remove_geometry(vis_cache["geom"], reset_bounding_box=False)
                vis_cache["geom"] = None
            
            # Add new geometry - only reset bounding box on first update
            reset_bbox = vis_cache["first_update"]
            pc_visualizer.add_geometry(new_pcd, reset_bounding_box=reset_bbox)
            vis_cache["geom"] = new_pcd
            
            # Restore view parameters after updating geometry
            if not vis_cache["first_update"]:
                view_control.convert_from_pinhole_camera_parameters(params)
            else:
                # Mark that we've done the first update
                vis_cache["first_update"] = False
            
            # Poll events and update renderer
            if not pc_visualizer.poll_events():
                # Window was closed by user, clean up
                if vis_cache["geom"] is not None:
                    pc_visualizer.remove_geometry(vis_cache["geom"], reset_bounding_box=False)
                if vis_cache["coord_frame"] is not None:
                    pc_visualizer.remove_geometry(vis_cache["coord_frame"], reset_bounding_box=False)
                try:
                    pc_visualizer.destroy_window()
                except Exception:
                    pass
                vis_cache["vis"] = None
                vis_cache["geom"] = None
                vis_cache["coord_frame"] = None
                vis_cache["initialized"] = False
                vis_cache["first_update"] = True  # Reset first_update flag
            else:
                # Update the renderer but ensure we don't block interaction
                pc_visualizer.update_renderer()
                # Process viewer events to allow rotation and zoom
                pc_visualizer.poll_events()
    else:
        # Use the original non-live visualization approach
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        colors = np.tile(np.array(color, dtype=np.float64), (len(points), 1))
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.1, origin=[0, 0, 0]
        )
        
        o3d.visualization.draw_geometries(
            [pcd, coordinate_frame],
            window_name=f'Point Cloud: {name}',
            width=1024,
            height=768
        )
        
                     
def pointcloud_to_voxel_grid(points: torch.Tensor, 
                             voxel_size: float | tuple[float, float, float], 
                             grid_range_min: tuple[float, float, float], 
                             grid_range_max: tuple[float, float, float], 
                             mode: str = "binary") -> torch.Tensor:
    """
    Converts a batch of point clouds to a voxel grid using vectorized operations.

    Args:
        points: Input point clouds, shape (B, N, 3).
        voxel_size: Size of each voxel (float or tuple for x, y, z).
        grid_range_min: Minimum coordinates (x, y, z) of the voxelization volume.
        grid_range_max: Maximum coordinates (x, y, z) of the voxelization volume.
        mode: Voxelization mode. "binary" for presence (0 or 1), "density" for point counts.

    Returns:
        Voxel grid, shape (B, 1, Dx, Dy, Dz).
    """
    batch_size = points.shape[0]
    device = points.device
    dtype = points.dtype

    # Ensure voxel_size is a tensor
    if isinstance(voxel_size, (int, float)):
        voxel_size_val = (float(voxel_size), float(voxel_size), float(voxel_size))
    else:
        voxel_size_val = voxel_size
    voxel_size_t = torch.tensor(voxel_size_val, device=device, dtype=dtype).view(1, 3)

    # Ensure grid_range_min/max are tensors
    grid_range_min_t = torch.tensor(grid_range_min, device=device, dtype=dtype).view(1, 3)
    grid_range_max_t = torch.tensor(grid_range_max, device=device, dtype=dtype).view(1, 3)

    # Calculate grid dimensions (Dx, Dy, Dz)
    grid_dims_float = (grid_range_max_t.squeeze() - grid_range_min_t.squeeze()) / voxel_size_t.squeeze()
    # Ensure grid dimensions are at least 1
    grid_dims = torch.max(torch.ceil(grid_dims_float), torch.tensor(1.0, device=device, dtype=dtype)).long().cpu().tolist()
    
    Dx, Dy, Dz = grid_dims[0], grid_dims[1], grid_dims[2]
    voxel_grid = torch.zeros((batch_size, 1, Dx, Dy, Dz), device=device, dtype=dtype)

    # Create mask for points within the grid range
    # points: (B, N, 3), grid_range_min_t.view(1,1,3): (1,1,3)
    valid_mask = (
        (points[..., 0] >= grid_range_min_t[0, 0]) & (points[..., 0] < grid_range_max_t[0, 0]) &
        (points[..., 1] >= grid_range_min_t[0, 1]) & (points[..., 1] < grid_range_max_t[0, 1]) &
        (points[..., 2] >= grid_range_min_t[0, 2]) & (points[..., 2] < grid_range_max_t[0, 2])
    ) # Shape: (B, N)

    # Get batch indices and filtered points
    batch_indices_for_valid_points, point_indices_in_batch = torch.nonzero(valid_mask, as_tuple=True)
    
    if batch_indices_for_valid_points.numel() == 0:
        return voxel_grid # No points fall within the grid

    valid_points = points[batch_indices_for_valid_points, point_indices_in_batch] # Shape: (M, 3), M = total valid points

    # Convert valid points to voxel indices
    # valid_points: (M, 3), grid_range_min_t: (1,3), voxel_size_t: (1,3)
    voxel_indices_float = (valid_points - grid_range_min_t) / voxel_size_t
    voxel_indices = voxel_indices_float.floor().long() # Shape: (M, 3) -> (ix, iy, iz)

    # Clamp indices to be within grid dimensions
    # Note: Dx, Dy, Dz are dimensions, so max index is dim-1
    voxel_indices[:, 0] = torch.clamp(voxel_indices[:, 0], 0, Dx - 1)
    voxel_indices[:, 1] = torch.clamp(voxel_indices[:, 1], 0, Dy - 1)
    voxel_indices[:, 2] = torch.clamp(voxel_indices[:, 2], 0, Dz - 1)

    # Populate the voxel grid
    if mode == "binary":
        # For binary mode, set voxel to 1 if any point falls into it.
        # batch_indices_for_valid_points: (M)
        # voxel_indices: (M, 3) -> (x_idx, y_idx, z_idx)
        # Voxel grid shape: (B, 1, Dx, Dy, Dz)
        voxel_grid[batch_indices_for_valid_points, 0, voxel_indices[:, 0], voxel_indices[:, 1], voxel_indices[:, 2]] = 1.0
    elif mode == "density":
        # For density mode, count the number of points in each voxel.
        # We combine batch index with voxel coordinates to uniquely identify each voxel cell across batches.
        # full_indices shape: (M, 4) -> (batch_idx, x_idx, y_idx, z_idx)
        full_indices = torch.cat((batch_indices_for_valid_points.unsqueeze(1), voxel_indices), dim=1)
        
        # Find unique voxel cells and the number of points mapping to each
        unique_voxel_coords, counts = torch.unique(full_indices, dim=0, return_counts=True)
        
        # unique_voxel_coords: (U, 4) where U is number of unique occupied voxels
        # counts: (U)
        voxel_grid[
            unique_voxel_coords[:, 0],  # Batch indices
            0,                          # Channel index
            unique_voxel_coords[:, 1],  # x_indices
            unique_voxel_coords[:, 2],  # y_indices
            unique_voxel_coords[:, 3]   # z_indices
        ] = counts.to(dtype)
    else:
        raise ValueError(f"Unsupported voxelization mode: {mode}. Choose 'binary' or 'density'.")
    
    return voxel_grid


def crop_pointcloud_to_workspace(
    fused_pc: torch.Tensor,            # (B, N, 3)
    initial_mask: torch.Tensor,        # (B, N), bool
    workspace_min_bounds: Optional[tuple[float, float, float]],
    workspace_max_bounds: Optional[tuple[float, float, float]],
    epsilon: float = 1e-5              # Small tolerance for floating-point comparison
) -> torch.Tensor:
    """
    Returns a mask of shape (B, N) selecting points that are both initially valid
    AND within the box [workspace_min_bounds, workspace_max_bounds].
    """
    if workspace_min_bounds is None or workspace_max_bounds is None:
        return initial_mask

    device = fused_pc.device
    ws_min = torch.tensor(workspace_min_bounds, device=device).view(1,1,3) - epsilon
    ws_max = torch.tensor(workspace_max_bounds, device=device).view(1,1,3) + epsilon
    
    # Calculate per-dimension bounds check
    x_in_bounds = (fused_pc[..., 0] >= ws_min[..., 0].squeeze(-1)) & (fused_pc[..., 0] <= ws_max[..., 0].squeeze(-1))
    y_in_bounds = (fused_pc[..., 1] >= ws_min[..., 1].squeeze(-1)) & (fused_pc[..., 1] <= ws_max[..., 1].squeeze(-1))
    z_in_bounds = (fused_pc[..., 2] >= ws_min[..., 2].squeeze(-1)) & (fused_pc[..., 2] <= ws_max[..., 2].squeeze(-1))
    
    # Apply bounds check (combine all dimensions)
    in_bounds = x_in_bounds & y_in_bounds & z_in_bounds
    
    # Final result combining initial mask and workspace bounds
    return initial_mask & in_bounds



def save_pointcloud_to_file(points, filename="debug_pointcloud", color=[1.0, 0.0, 0.0]):
    """
    Save a point cloud to a PLY file for external viewing
    
    Args:
        points: numpy array or torch tensor of shape (N, 3) 
        filename: filename to save (without extension)
        color: RGB color for the points
    """
    import numpy as np
    import open3d as o3d
    
    # Convert to numpy if it's a torch tensor
    if isinstance(points, torch.Tensor):
        points_np = points.detach().cpu().numpy()
    else:
        points_np = points
        
    # Remove any zero points (these are often invalid/padding points)
    if points_np.ndim == 3:  # If (B, N, 3), take first batch
        points_np = points_np[0]
    
    # Filter out zero points
    non_zero_mask = np.any(points_np != 0, axis=1)
    points_np_filtered = points_np[non_zero_mask]
    
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_np_filtered)
    
    # Set colors
    colors = np.tile(np.array(color, dtype=np.float64), (len(points_np_filtered), 1))
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Save to file
    output_path = f"source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift/mdp/pointclouds/{filename}.ply"
    o3d.io.write_point_cloud(output_path, pcd)
    print(f"Point cloud saved to {output_path}")
    
    return output_path



