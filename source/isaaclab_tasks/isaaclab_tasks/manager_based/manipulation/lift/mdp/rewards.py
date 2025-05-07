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
from isaaclab.utils.math import (
    combine_frame_transforms,
    quat_inv,
    quat_mul,
    matrix_from_quat as quat_to_rotmat,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_is_lifted(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    lift_width: float = 0.01,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """
    Smooth “lift” gate: transitions from 0→1 over ~2*lift_width meters around minimal_height.
    R_lift(z) = sigmoid((z - minimal_height) / lift_width)
    """
    obj = env.scene[object_cfg.name]
    z   = obj.data.root_pos_w[:, 2]
    # lift_width controls the slope: smaller → sharper transition
    return torch.sigmoid((z - minimal_height) / lift_width)


def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    bonus_threshold: float = 0.02,
    bonus_weight: float = 0.3,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """
    Gaussian reach reward + small close-in bonus:
      R_dist = exp(-0.5 * (d/std)^2)  ∈ (0,1]
      R_bonus = bonus_weight if d < bonus_threshold else 0
      R = R_dist + R_bonus
    """
    obj      = env.scene[object_cfg.name]
    ee_frame = env.scene[ee_frame_cfg.name]
    # positions
    cube_pos = obj.data.root_pos_w[:, :3]
    ee_pos   = ee_frame.data.target_pos_w[..., 0, :]
    # Euclidean distance
    dist     = torch.norm(cube_pos - ee_pos, dim=1)
    # Gaussian-shaped term
    r_dist   = torch.exp(-0.5 * (dist / std) ** 2)
    # small bonus when very close
    r_bonus  = (dist < bonus_threshold).float() * bonus_weight
    return r_dist + r_bonus


# def object_goal_distance(
#     env: ManagerBasedRLEnv,
#     std: float,
#     minimal_height: float,
#     command_name: str,
#     robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
#     object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
# ) -> torch.Tensor:
#     """Reward the agent for tracking the goal pose using tanh-kernel."""
#     # extract the used quantities (to enable type-hinting)
#     robot: RigidObject = env.scene[robot_cfg.name]
#     object: RigidObject = env.scene[object_cfg.name]
#     command = env.command_manager.get_command(command_name)
#     # compute the desired position in the world frame
#     des_pos_b = command[:, :3]
#     des_pos_w, _ = combine_frame_transforms(robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], des_pos_b)
#     # distance of the end-effector to the object: (num_envs,)
#     distance = torch.norm(des_pos_w - object.data.root_pos_w[:, :3], dim=1)
#     # rewarded if the object is lifted above the threshold
#     return (object.data.root_pos_w[:, 2] > minimal_height) * (1 - torch.tanh(distance / std))


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


def object_goal_distance_pos(
    env, std, minimal_height, lift_width, command_name, robot_cfg, object_cfg
):
    # Scene handles
    robot = env.scene[robot_cfg.name]
    obj   = env.scene[object_cfg.name]
    cmd   = env.command_manager.get_command(command_name)  # [B,7]

    # World-frame goal position via rotation-matrix
    p_rb   = robot.data.root_state_w[:, :3]       # [B,3]
    q_rb   = robot.data.root_state_w[:, 3:7]      # [B,4]
    p_des_b= cmd[:, :3]                           # [B,3]
    R_rb   = quat_to_rotmat(q_rb)                 # [B,3,3]
    p_des_w= (R_rb @ p_des_b.unsqueeze(-1)).squeeze(-1) + p_rb  # [B,3]

    # Distance and Gaussian reward
    p_obj  = obj.data.root_pos_w[:, :3]
    dist   = torch.norm(p_obj - p_des_w, dim=1)
    r_pos  = torch.exp(-0.5 * (dist / std)**2)

    # Soft lift gate
    z_obj  = p_obj[:, 2]
    gate   = torch.sigmoid((z_obj - minimal_height) / lift_width)
    return gate * r_pos


def ee_goal_distance_ori(
    env, std, command_name, robot_cfg, ee_frame_cfg
):
    """
    Reward end-effector orientation alignment with goal orientation. 
    Uses quaternion log-map to compute orientation error.
    """
    robot = env.scene[robot_cfg.name]
    ee = env.scene[ee_frame_cfg.name]
    cmd = env.command_manager.get_command(command_name)  # [B,7]
    
    # World-frame goal quaternion
    q_des_b = cmd[:, 3:7]  # [B,4]
    p_rb, q_rb = robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7]
    
    # Transform goal quaternion to world frame
    zero_trans = torch.zeros_like(p_rb)  # [B,3]
    _, q_des_w = combine_frame_transforms(
        p_rb, q_rb,
        zero_trans,  # Zero translation vector
        q_des_b
    )
    
    # Get end-effector orientation
    q_all = ee.data.target_quat_w  # [B, M, 4]
    q_ee = q_all[:, 0, :]  # Take first (and only) target frame → [B, 4]
    
    # Compute orientation difference between ee and desired orientation
    q_inv = quat_inv(q_des_w)  # [B,4]
    q_rel = quat_mul(q_inv, q_ee)  # [B,4]
    
    # Convert to axis-angle representation (log map) for error metric
    w, v = q_rel[:, :1], q_rel[:, 1:]  # Real and vector parts
    angle = torch.acos(torch.clamp(w, -1.0, 1.0))  # Extract angle
    sin_ang = torch.sin(angle).clamp(min=1e-8)  # Safe division
    axis = v / sin_ang
    log_map = axis * angle
    err = torch.norm(log_map, dim=1)  # Angular error magnitude
    
    # Convert to Gaussian reward (1.0 when perfectly aligned, decays with error)
    return torch.exp(-0.5 * (err / std)**2)


# def object_goal_distance_ori(
#     env, std, command_name, robot_cfg, object_cfg
# ):
#     robot = env.scene[robot_cfg.name]
#     obj   = env.scene[object_cfg.name]
#     cmd   = env.command_manager.get_command(command_name)  # [B,7]
#     B     = cmd.shape[0]

#     # World-frame goal quaternion
#     q_des_b = cmd[:, 3:7]                                # [B,4]
#     p_rb, q_rb = robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7]

#     # ::: CHANGE HERE :::
#     # use a zero translation of shape [B,3], not zeros_like(q_des_b) which is [B,4]
#     zero_trans = torch.zeros_like(p_rb)                  # [B,3]
#     _, q_des_w = combine_frame_transforms(
#         p_rb, q_rb, 
#         zero_trans,                             # <— was torch.zeros_like(q_des_b)
#         q_des_b
#     )

#     # Object quaternion
#     q_obj = obj.data.root_quat_w                         # [B,4]

#     # rest unchanged...
#     # Batch-safe inverse and multiply
#     q_inv = quat_inv(q_des_w)                            # [B,4]
#     q_rel = quat_mul(q_inv, q_obj)                       # [B,4]

#     w, v = q_rel[:, :1], q_rel[:, 1:]
#     angle   = torch.acos(torch.clamp(w, -1.0, 1.0))
#     sin_ang = torch.sin(angle).clamp(min=1e-8)
#     axis    = v / sin_ang
#     log_map = axis * angle
#     err     = torch.norm(log_map, dim=1)

#     return torch.exp(-0.5 * (err / std)**2)

def object_slip_penalty_norm(
    env: ManagerBasedRLEnv,
    threshold: float = 0.01,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """
    Penalizes tangential slip velocity above `threshold` by:
      1) projecting obj.lin_vel onto the EE’s +Z axis in world frame
      2) measuring the residual (tangential) velocity
      3) clamping any slip > threshold to produce a negative reward
    """
    # 1) object linear velocity in world
    obj   = env.scene[object_cfg.name]
    v_obj = obj.data.root_lin_vel_w               # [B, 3]

    # 2) end-effector world orientation via FrameTransformerData.target_quat_w
    ee        = env.scene[ee_frame_cfg.name]
    q_all     = ee.data.target_quat_w             # [B, M, 4]
    q_ee      = q_all[:, 0, :]                    # take first (and only) target frame → [B, 4]
    R_ee      = quat_to_rotmat(q_ee)              # [B, 3, 3]
    n_grasp   = R_ee[:, :, 2]                     # local +Z axis in world frame → [B, 3]

    # 3) compute slip = ‖v_obj − (v_obj·n_grasp) n_grasp‖
    v_norm    = (v_obj * n_grasp).sum(dim=1, keepdim=True) * n_grasp  # [B,3]
    slip      = torch.norm(v_obj - v_norm, dim=1)                     # [B]

    # 4) thresholded penalty: −max(0, slip − threshold)
    slip_over = -torch.clamp(slip - threshold, min=0.0)
    s_norm   = slip_over / threshold            # dimensionless
    return -s_norm                              # unit‐scale penalty


class LiftPotentialBuffer:
    """Tracks previous potential per environment for ΔΦ computation."""
    def __init__(self, num_envs: int, z_max: float):
        self.prev = torch.zeros(num_envs, device="cpu")  # will move to correct device in compute
        self.z_max = z_max

    def compute(self, z: torch.Tensor):
        # ensure buffer lives on same device
        prev = self.prev.to(z.device)
        phi = torch.clamp(z, max=self.z_max)
        delta = phi - prev
        # store for next step
        self.prev = phi.detach().cpu()
        return delta



def potential_based_lift(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """
    Returns ΔΦ for height shaping where Φ(z)=min(z, z_max).
    Only positive increments (lifting) produce reward.
    """
    # instantiate buffer sized to this env on first call
    if not hasattr(env, "_lift_potential_buffer"):
        # you could pass minimal_height or pick a fixed z_max:
        env._lift_potential_buffer = LiftPotentialBuffer(env.num_envs, z_max=minimal_height * 2.0)
    buf = env._lift_potential_buffer

    obj = env.scene[object_cfg.name]
    z   = obj.data.root_pos_w[:, 2]
    return buf.compute(z)


# def ee_table_penalty(
#     env, table_height, safety_buffer=0.005,
#     ee_frame_cfg=SceneEntityCfg("ee_frame"),
#     object_cfg=SceneEntityCfg("object"),
#     safe_radius=0.1  # Safe zone around object
# ):
#     ee = env.scene[ee_frame_cfg.name]
#     obj = env.scene[object_cfg.name]
    
#     # Get positions
#     ee_pos = ee.data.target_pos_w[..., 0, :]
#     obj_pos = obj.data.root_pos_w
    
#     # Check if EE is close to object horizontally
#     ee_obj_dist_xy = torch.norm(ee_pos[:, :2] - obj_pos[:, :2], dim=1)
#     near_object = ee_obj_dist_xy < safe_radius
    
#     # Only apply table penalty when not near object
#     z_ee = ee_pos[:, 2]
#     close_pen = torch.where(
#         near_object,
#         torch.zeros_like(z_ee),  # No penalty near object
#         -0.5 * ((z_ee < table_height + safety_buffer) & (z_ee >= table_height))
#     )
    
#     # Always apply severe collision penalty
#     collide = -50.0 * (z_ee < table_height)
    
#     return close_pen + collide




###################
# def multiple_grasp_attempts_penalty(
#     env: ManagerBasedRLEnv,
#     penalty_per_attempt: float = 0.5,
#     object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
#     ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
#     grasp_dist_thresh: float = 0.08,
# ) -> torch.Tensor:
#     """Penalize multiple open→close attempts only when the gripper is near the object."""
#     # init tracking dict
#     if not hasattr(env, "_grasp_tracking"):
#         env._grasp_tracking = {
#             "prev_closed": None,          # will store last gripper_action<0 mask
#             "attempts": torch.zeros(env.num_envs, device=env.device, dtype=torch.long),
#         }

#     # fetch last continuous action (shape [num_envs, action_dim])
#     last_act = env.action_manager.prev_action
#     if last_act is None or last_act.shape[0] != env.num_envs:
#         return torch.zeros(env.num_envs, device=env.device)
#     gripper_act = last_act[:, -1]  # assume last dim is gripper: >0=open, <0=close
#     closed_mask = gripper_act < 0  # True when closed

#     # seed prev_closed on step 1
#     if env._grasp_tracking["prev_closed"] is None:
#         env._grasp_tracking["prev_closed"] = closed_mask.clone()
#         return torch.zeros(env.num_envs, device=env.device)

#     # detect open->close transitions
#     just_closed = (~env._grasp_tracking["prev_closed"]) & closed_mask

#     # only count if near object
#     obj = env.scene[object_cfg.name]
#     ee = env.scene[ee_frame_cfg.name]
#     obj_pos = obj.data.root_pos_w[:, :3]
#     ee_pos = ee.data.target_pos_w[..., 0, :]
#     near_obj = torch.norm(obj_pos - ee_pos, dim=1) < grasp_dist_thresh

#     # increment attempts
#     env._grasp_tracking["attempts"] += (just_closed & near_obj)

#     # update prev_closed
#     env._grasp_tracking["prev_closed"] = closed_mask.clone()

#     # penalty = (attempts − 1)×penalty_per_attempt (clamped ≥0)
#     counts = env._grasp_tracking["attempts"]
#     penalty = torch.clamp(counts - 1, min=0).float() * penalty_per_attempt

#     # reset on done
#     done_ids = env.reset_buf.nonzero(as_tuple=False).squeeze(-1)
#     if len(done_ids) > 0:
#         env._grasp_tracking["attempts"][done_ids] = 0
#         env._grasp_tracking["prev_closed"][done_ids] = False

#     return penalty