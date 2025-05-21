import sys
import pathlib

# Add the project root directory to sys.path
# Assumes test.py is in /home/chris/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift/mdp/
# and I2PMAE is in /home/chris/IsaacLab/I2PMAE/
# So, we need to add /home/chris/IsaacLab to sys.path
current_file_path = pathlib.Path(__file__).resolve()
# Navigate 7 levels up from mdp directory to reach IsaacLab directory
# mdp -> lift -> manipulation -> manager_based -> isaaclab_tasks (module) -> isaaclab_tasks (package) -> source -> IsaacLab
project_root = current_file_path.parents[7] # Ensure this is correct for your structure
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from I2PMAE.models.I2P_MAE import I2P_MAE
# ... rest of your test.py script ...
import torch # Assuming torch is used later
from I2PMAE.utils import config

# Determine the root directory of the I2P-MAE project itself (which is inside IsaacLab project root)
i2p_mae_project_root = project_root / "I2PMAE"

cfg_yaml_path = i2p_mae_project_root / "cfgs" / "pre-training" / "i2p-mae.yaml"
ckpt_path = i2p_mae_project_root / "ckpts" / "pre-train.pth"

if not cfg_yaml_path.is_file():
    raise FileNotFoundError(f"Config file not found: {cfg_yaml_path}")
if not ckpt_path.is_file():
    raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")

# Pass the I2PMAE project root path (as a string) to cfg_from_yaml_file
cfg = config.cfg_from_yaml_file(str(cfg_yaml_path), str(i2p_mae_project_root))

if 'model' not in cfg:
    raise KeyError(f"The configuration file {cfg_yaml_path} does not contain a top-level 'model' key.")


encoder = I2P_MAE(cfg.model)

# Load pre-trained weights
checkpoint = torch.load(str(ckpt_path), map_location='cpu')
if 'model' in checkpoint:
    # If weights are stored under 'model' key (common in training checkpoints)
    encoder.load_state_dict(checkpoint['model'], strict=False)
else:
    # Try loading directly
    encoder.load_state_dict(checkpoint, strict=False)

print(encoder)

# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
encoder = encoder.to(device)
encoder.eval()  # Set to evaluation mode

print(f"Model loaded and ready for inference on {device}")

# Create a dummy point cloud
import numpy as np
def test_with_dummy_pointcloud():
    print("\nTesting model with dummy point cloud...")
    
    # Create random point cloud with specific size 
    dummy_pc = torch.rand(1, 1024, 3).float().to(device)  # [B, N, 3]
    dummy_pc = dummy_pc / dummy_pc.abs().max()  # Normalize to [-1, 1]
    
    print(f"Input shape: {dummy_pc.shape}")
    
    # Use the model with the proper methodology
    with torch.no_grad():
        # The forward method requires a pts argument
        try:
            # Try using the model's built-in processing
            # Passing eval=True to avoid training-specific operations
            features = encoder(dummy_pc, eval=True)
            print("Standard forward call successful!")
            print_features(features)
            return
        except Exception as e:
            print(f"Standard forward call failed: {e}")
        
        # Try using the group dividers to get required centers and indices
        try:
            print("\nTrying with explicit preprocessing...")
            
            # The group_dividers are used to process the point cloud hierarchically
            # This mimics what would happen inside the model's forward method
            points_xyz = [dummy_pc]  # Start with the raw points
            centers_list = []
            idxs_list = []
            
            # Process the point cloud through group dividers to build hierarchy
            for i, group_divider in enumerate(encoder.group_dividers):
                # Get the centers and indices for this level
                centers, idxs = group_divider(points_xyz[-1])
                centers_list.append(centers)
                idxs_list.append(idxs)
                points_xyz.append(centers)
            
            # Now run the h_encoder with the required centers and indices
            encoder_output = encoder.h_encoder(points_xyz, centers_list, idxs_list)
            print("Manual processing successful!")
            print_features(encoder_output)
            return
        except Exception as e:
            print(f"Explicit preprocessing failed: {e}")
            
    print("\nAll attempts failed. I2P-MAE may require specific input formats.")
    print("Consider checking the source code or documentation for examples.")

def print_features(features):
    """Helper function to print feature information."""
    if features is None:
        print("Features is None")
        return
        
    if isinstance(features, list) or isinstance(features, tuple):
        print("Model returned multiple feature levels:")
        for i, feat in enumerate(features):
            if torch.is_tensor(feat):
                print(f"  Level {i} feature shape: {feat.shape}")
            else:
                print(f"  Level {i} feature type: {type(feat)}")
    else:
        print(f"Output feature shape: {features.shape}")
    print("Forward pass successful!")

# Call the test function
test_with_dummy_pointcloud()