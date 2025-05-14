# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg, AssetBaseCfg
from isaaclab.sensors import TiledCameraCfg, CameraCfg, FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR


from isaaclab_tasks.manager_based.manipulation.lift import mdp
from isaaclab_tasks.manager_based.manipulation.lift.lift_rgbd_env_cfg import LiftRGBDEnvCfg

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG  # isort: skip


@configclass
class FrankaCubeLiftRGBDEnvCfg(LiftRGBDEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set Franka as robot
        self.scene.robot = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Set actions for the specific robot type (franka)
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=["panda_joint.*"], scale=0.5, use_default_offset=True
        )
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_finger.*"],
            open_command_expr={"panda_finger_.*": 0.04},
            close_command_expr={"panda_finger_.*": 0.0},
        )
        # Set the body name for the end effector
        self.commands.object_pose.body_name = "panda_hand"
        
        
        # # Set Cone as object
        # self.scene.object = RigidObjectCfg(
        #     prim_path="{ENV_REGEX_NS}/Object",
        #     # Adjust initial position/rotation if needed for the cone's size/origin
        #     init_state=RigidObjectCfg.InitialStateCfg(pos=[0.5, 0, 0.000], rot=      [1, 0, 0, 0]),
        #     spawn=UsdFileCfg(
        #         # Make sure this path points correctly to your converted cone.usd file
        #         usd_path="assets/my_custom_assets/usd/cone.usd",
        #         # Remove or adjust scale if cone.usd is already in meters
        #         # scale=(1.0, 1.0, 1.0), # Example: No scaling if cone.usd is in meters
        #         rigid_props=RigidBodyPropertiesCfg(
        #             solver_position_iteration_count=16,
        #             solver_velocity_iteration_count=1,
        #             max_angular_velocity=1000.0,
        #             max_linear_velocity=1000.0,
        #             max_depenetration_velocity=5.0,
        #             disable_gravity=False,
        #         ),
        #     ),
        # )


        # Set Cube as object
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.5, 0.0, 0.055], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                scale=(0.8, 0.8, 0.8),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
            ),
        )
        
        # # Set table view camera
        # self.scene.table_cam = CameraCfg(
        #     prim_path="{ENV_REGEX_NS}/table_cam",
        #     update_period=0.0333,
        #     height=128,
        #     width=128,
        #     data_types=["rgb", "depth"],
        #     spawn=sim_utils.PinholeCameraCfg(
        #         focal_length=4.0, focus_distance=400.0, horizontal_aperture=6.4, clipping_range=(0.1, 1.0e5)
        #     ),
        #     offset=CameraCfg.OffsetCfg(pos=(1.5, 1.0, 0.5), rot=(-0.336, -0.145, -0.053, 0.929), convention="world"),
        # )
        
        # # Set table view camera
        # self.scene.table_cam_2 = CameraCfg(
        #     prim_path="{ENV_REGEX_NS}/table_cam_2",
        #     update_period=0.0333,
        #     height=128,
        #     width=128,
        #     data_types=["rgb", "depth"],
        #     spawn=sim_utils.PinholeCameraCfg(
        #         focal_length=4.0, focus_distance=400.0, horizontal_aperture=6.4, clipping_range=(0.1, 1.0e5)
        #     ),
        #     offset=CameraCfg.OffsetCfg(pos=(1.5, -1.0, 0.5), rot=(0.336, -0.145, 0.053, 0.929), convention="world"),
        # )
        
    #     # add camera to the scene
    # tiled_camera: TiledCameraCfg = TiledCameraCfg(
    #     prim_path="{ENV_REGEX_NS}/Camera",
    #     offset=TiledCameraCfg.OffsetCfg(pos=(-7.0, 0.0, 3.0), rot=(0.9945, 0.0, 0.1045, 0.0), convention="world"),
    #     data_types=["rgb"],
    #     spawn=sim_utils.PinholeCameraCfg(
    #         focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 20.0)
    #     ),
    #     width=100,
    #     height=100,
    # )
        
        # camera #1
        self.scene.table_cam = TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/table_cam",
            # update_period=0.0333,
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0, focus_distance=2.0,
                horizontal_aperture=20.955, clipping_range=(0.4, 30.0)
            ),
            offset=TiledCameraCfg.OffsetCfg(pos=(-1.5,  1.0, 0.5),
                            rot=(-0.336, -0.145, -0.053, 0.929),
                            convention="world"),
            width=128,
            height=128,
            data_types=["rgb","distance_to_camera"],
        )
        
        # camera #2
        self.scene.table_cam_2 = TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/table_cam_2",
            # update_period=0.0333,
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0, focus_distance=2.0,
                horizontal_aperture=20.955, clipping_range=(0.4, 30.0)
            ),
            offset=TiledCameraCfg.OffsetCfg(pos=(1.5, -1.0, 0.5),
                            rot=( 0.336, -0.145,  0.053, 0.929),
                            convention="world"),
            width=128,
            height=128,
            data_types=["rgb","distance_to_camera"],
        )
        

        # Listens to the required transforms
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        
        marker_cfg_object = marker_cfg.copy()   # or create a new config if needed
        
        # visualize object frame
        self.scene.object_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Object",  # same prim as the object
            debug_vis=True,
            visualizer_cfg=marker_cfg_object,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Object",  # if your object USD has a specific frame, use it here
                    name="object_origin",
                    offset=OffsetCfg(pos=[0.0, 0.0, 0.0]),
                ),
            ],
        )
        
        # viualize ee frame
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
            debug_vis=True,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.0, 0.0, 0.1034],
                    ),
                ),
            ],
        )
        
        
        # visualize robot frame
        self.scene.robot_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
            debug_vis=True,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
                    name="robot_base",
                    offset=OffsetCfg(pos=[0.0, 0.0, 0.0]),
                ),
            ],
        )
        

@configclass
class FrankaCubeLiftRGBDEnvCfg_PLAY(FrankaCubeLiftRGBDEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
