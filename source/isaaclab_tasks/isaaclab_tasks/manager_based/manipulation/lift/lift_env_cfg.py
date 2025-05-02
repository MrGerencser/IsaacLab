# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, DeformableObjectCfg, RigidObjectCfg
from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG, FRANKA_PANDA_STABLE_CFG
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from . import mdp

# Import general reward and penalty functions from core isaaclab
from isaaclab.envs.mdp.rewards import (
    is_terminated,
    joint_torques_l2,
    joint_acc_l2,
    joint_vel_l2,
    action_l2,
    body_lin_acc_l2,
    joint_vel_limits,
    joint_deviation_l1,
    flat_orientation_l2,
    action_rate_l2,
    joint_pos_limits,
    applied_torque_limits,
)

##
# Scene definition
##

@configclass
class ObjectTableSceneCfg(InteractiveSceneCfg):
    """Configuration for the lift scene with a robot and an object."""
    robot: ArticulationCfg = FRANKA_PANDA_STABLE_CFG
    ee_frame: FrameTransformerCfg = MISSING
    object: RigidObjectCfg | DeformableObjectCfg = MISSING

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0, 0], rot=[0.707, 0, 0, 0.707]),
        spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"),
    )

    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -1.05]),
        spawn=GroundPlaneCfg(),
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

##
# MDP settings
##

@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    object_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=MISSING,
        resampling_time_range=(5.0, 5.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.4, 0.6), pos_y=(-0.25, 0.25), pos_z=(0.25, 0.5),
            roll=(0.0, 0.0), pitch=(0.0, 0.0), yaw=(0.0, 0.0)
        ),
    )

@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    arm_action: mdp.JointPositionActionCfg | mdp.DifferentialInverseKinematicsActionCfg = MISSING
    gripper_action: mdp.BinaryJointPositionActionCfg = MISSING

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame)
        object_orientation = ObsTerm(func=mdp.object_orientation_in_robot_frame)
        target_object_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()

@configclass
class EventCfg:
    """Configuration for events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    reset_object_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.1, 0.1),
                "y": (-0.25, 0.25),
                "z": (0.0, 0.0),
                "roll": (-3.14, 3.14),
                "pitch": (-3.14, 3.14),
                "yaw": (-3.14, 3.14),
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object", body_names="Object"),
        },
    )
    
    
@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    reaching_object = RewTerm(func=mdp.object_ee_distance, params={"std": 0.1}, weight=7.0)

    lifting_object = RewTerm(func=mdp.object_is_lifted, params={"minimal_height": 0.04}, weight=15.0)
    
    lifting_object_fine_grained = RewTerm(func=mdp.object_is_lifted, params={"minimal_height": 0.1}, weight=0.0)
    

    object_goal_tracking = RewTerm(
        func=mdp.object_goal_distance,
        params={"std": 0.3, "minimal_height": 0.04, "command_name": "object_pose"},
        weight=16.0,
    )

    object_goal_tracking_fine_grained = RewTerm(
        func=mdp.object_goal_distance,
        params={"std": 0.05, "minimal_height": 0.04, "command_name": "object_pose"},
        weight=5.0,
    )

    # action penalty
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)

    # joint velocity penalty
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    
    # joint acceleration penalty
    joint_acc = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-1e-9, # Example: start with a small weight
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    
    # # multiple grasp attempts penalty
    # multi_grasp_penalty = RewTerm(
    #     func=mdp.multiple_grasp_attempts_penalty, 
    #     params={"penalty_per_attempt": 0.5},  
    #     weight=-1e-4
    # )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("object")}
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""
    
    reaching_object = CurrTerm(
        # num_steps = iterations * num_steps_per_env
        func=mdp.gradually_modify_reward_weight, params={
            "term_name": "reaching_object",
            "start_weight": 7.0,
            "end_weight": 1.0,
            "num_steps": 200 * 24,
            "curve_type": "linear",
            "start_step": 100 * 24,
        }
    )
    
    lifting_object_fine_grained = CurrTerm(
        # num_steps = iterations * num_steps_per_env
        func=mdp.modify_reward_weight, params={
            "term_name": "lifting_object_fine_grained",
            "weight": 15.0,
            "num_steps": 600 * 24,
        }
    )
    
    # # Increase the weight of the multi-grasp penalty term
    # multi_grasp_penalty = CurrTerm(
    #     # num_steps = iterations * num_steps_per_env
    #     func=mdp.modify_reward_weight, params={
    #         "term_name": "multi_grasp_penalty",
    #         "weight": -5.0,
    #         "num_steps": 1000 * 24,
    #     }
    # )
    
    # # Increase the weight of the multi-grasp penalty term
    # multi_grasp_penalty = CurrTerm(
    #     # num_steps = iterations * num_steps_per_env
    #     func=mdp.gradually_modify_reward_weight, params={
    #         "term_name": "multi_grasp_penalty",
    #         "start_weight": -1e-4,
    #         "end_weight": -20.0,
    #         "num_steps": 1000 * 24,
    #         "curve_type": "logarithmic",
    #         "start_step": 650 * 24,
    #     }
    # )
    
    # # Remove the reaching object term
    # reaching_object = CurrTerm(
    #     # num_steps = iterations * num_steps_per_env
    #     func=mdp.modify_reward_weight, params={"term_name": "reaching_object", "weight": 0.0, "num_steps": 2000 * 24}
    # )
    
    
    # Sudden increase the weight of the action rate term
    action_rate = CurrTerm(
        # num_steps = iterations * num_steps_per_env
        func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -1e-1, "num_steps": 1000 * 24}
    )

    joint_vel = CurrTerm(
        # num_steps = iterations * num_steps_per_env
        func=mdp.modify_reward_weight, params={"term_name": "joint_vel", "weight": -1e-1, "num_steps": 1000 * 24} 
    )
    
    # joint_acc = CurrTerm(
    #     # num_steps = iterations * num_steps_per_env
    #     func=mdp.modify_reward_weight, params={"term_name": "joint_acc", "weight": -1e-4, "num_steps": 1000 * 24} 
    # )
    


##
# Environment configuration
##


@configclass
class LiftEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the lifting environment."""

    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        self.decimation = 2
        self.episode_length_s = 5.0
        self.sim.dt = 0.01
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
