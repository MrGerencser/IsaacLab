# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_is_lifted(
    env: ManagerBasedRLEnv, minimal_height: float, object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """Reward the agent for lifting the object above the minimal height."""
    object: RigidObject = env.scene[object_cfg.name]
    return torch.where(object.data.root_pos_w[:, 2] > minimal_height, 1.0, 0.0)


def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward the agent for reaching the object using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    # Target object position: (num_envs, 3)
    cube_pos_w = object.data.root_pos_w
    # End-effector position: (num_envs, 3)
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    # Distance of the end-effector to the object: (num_envs,)
    object_ee_distance = torch.norm(cube_pos_w - ee_w, dim=1)

    return 1 - torch.tanh(object_ee_distance / std)


def object_goal_distance(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward the agent for tracking the goal pose using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    # compute the desired position in the world frame
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], des_pos_b)
    # distance of the end-effector to the object: (num_envs,)
    distance = torch.norm(des_pos_w - object.data.root_pos_w[:, :3], dim=1)
    # rewarded if the object is lifted above the threshold
    return (object.data.root_pos_w[:, 2] > minimal_height) * (1 - torch.tanh(distance / std))


# # Reward agent for moving the object closer to the goal. Use tanh-kernel to reward the agent for reaching the goal.
# def object_goal_distance(
#     env: ManagerBasedRLEnv,
#     std: float,
#     minimal_height: float,
#     command_name: str,
#     robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
#     object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
# ) -> torch.Tensor:
#     # Extract robot & object states
#     robot: RigidObject = env.scene[robot_cfg.name]
#     object: RigidObject = env.scene[object_cfg.name]
#     command = env.command_manager.get_command(command_name)
#     # Compute desired position in the base frame xyz and the quaterion
#     des_pos_b, des_quat_b = command[:, :3], command[:, 3:7]
#     # Convert the goal position and orientation from local frame to base frame
#     des_pos_w, des_quat_w = combine_frame_transforms(
#         robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7],
#         des_pos_b, des_quat_b
#     )
#     # Compute position distance -> Encourage agent to reduce this distance over time
#     distance = torch.norm(des_pos_w - object.data.root_pos_w[:, :3], dim=1)
#     # Compute orientation difference (corrected quaternion distance)
#     object_quat_w = object.data.root_quat_w
#     quat_diff = torch.abs(torch.sum(des_quat_w * object_quat_w, dim=1))
#     # 2*acos(...) converts it into a full angle distance in radiands
#     quat_diff = 2 * torch.acos(torch.clamp(quat_diff, -1, 1))
#     # Compute tanh-based penalties
#     position_penalty = 1 - torch.tanh(distance / std)
#     # Normalize the orientation penalty to be in [0, 1]
#     orientation_penalty = 1 - torch.tanh(0.5 * quat_diff / 3.14)
#     # Final reward calculation (only when object is lifted above minimal height) , too heigh weight for orientation lead to oscillations
#     return (object.data.root_pos_w[:, 2] > minimal_height) * (position_penalty + 1.5*orientation_penalty)


def multiple_grasp_attempts_penalty(
    env: ManagerBasedRLEnv,
    penalty_per_attempt: float = 0.5,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    grasp_dist_thresh: float = 0.08,
) -> torch.Tensor:
    """Penalize multiple open→close attempts only when the gripper is near the object."""
    # init tracking dict
    if not hasattr(env, "_grasp_tracking"):
        env._grasp_tracking = {
            "prev_closed": None,          # will store last gripper_action<0 mask
            "attempts": torch.zeros(env.num_envs, device=env.device, dtype=torch.long),
        }

    # fetch last continuous action (shape [num_envs, action_dim])
    last_act = env.action_manager.prev_action
    if last_act is None or last_act.shape[0] != env.num_envs:
        return torch.zeros(env.num_envs, device=env.device)
    gripper_act = last_act[:, -1]  # assume last dim is gripper: >0=open, <0=close
    closed_mask = gripper_act < 0  # True when closed

    # seed prev_closed on step 1
    if env._grasp_tracking["prev_closed"] is None:
        env._grasp_tracking["prev_closed"] = closed_mask.clone()
        return torch.zeros(env.num_envs, device=env.device)

    # detect open->close transitions
    just_closed = (~env._grasp_tracking["prev_closed"]) & closed_mask

    # only count if near object
    obj = env.scene[object_cfg.name]
    ee = env.scene[ee_frame_cfg.name]
    obj_pos = obj.data.root_pos_w[:, :3]
    ee_pos = ee.data.target_pos_w[..., 0, :]
    near_obj = torch.norm(obj_pos - ee_pos, dim=1) < grasp_dist_thresh

    # increment attempts
    env._grasp_tracking["attempts"] += (just_closed & near_obj)

    # update prev_closed
    env._grasp_tracking["prev_closed"] = closed_mask.clone()

    # penalty = (attempts − 1)×penalty_per_attempt (clamped ≥0)
    counts = env._grasp_tracking["attempts"]
    penalty = torch.clamp(counts - 1, min=0).float() * penalty_per_attempt

    # reset on done
    done_ids = env.reset_buf.nonzero(as_tuple=False).squeeze(-1)
    if len(done_ids) > 0:
        env._grasp_tracking["attempts"][done_ids] = 0
        env._grasp_tracking["prev_closed"][done_ids] = False

    return penalty