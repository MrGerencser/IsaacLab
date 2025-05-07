# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to create curriculum for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def modify_reward_weight(env: ManagerBasedRLEnv, env_ids: Sequence[int], term_name: str, weight: float, num_steps: int):
    """Curriculum that modifies a reward weight a given number of steps.

    Args:
        env: The learning environment.
        env_ids: Not used since all environments are affected.
        term_name: The name of the reward term.
        weight: The weight of the reward term.
        num_steps: The number of steps after which the change should be applied.
    """
    if env.common_step_counter > num_steps:
        # obtain term settings
        term_cfg = env.reward_manager.get_term_cfg(term_name)
        # update term settings
        term_cfg.weight = weight
        env.reward_manager.set_term_cfg(term_name, term_cfg)
        
        
def gradually_modify_reward_weight(
    env: ManagerBasedRLEnv, 
    env_ids: Sequence[int], 
    term_name: str, 
    start_weight: float, 
    end_weight: float, 
    num_steps: int,
    start_step: int = 0,
    curve_type: str = "linear"
):
    """Gradually modifies a reward weight over a specified range of steps.
    
    Args:
        env: The learning environment.
        env_ids: Not used since all environments are affected.
        term_name: The name of the reward term.
        start_weight: Starting weight value.
        end_weight: Target weight value to reach.
        num_steps: The number of steps over which the change occurs.
        start_step: When to begin the gradual change (default: 0).
        curve_type: Type of interpolation ("linear", "quadratic", "cubic", "exp").
    """
    current_step = env.common_step_counter

    # Do nothing until we reach the start of the transition
    if current_step < start_step:
        return

    # After the transition period: ensure final weight
    if current_step >= start_step + num_steps:
        term_cfg = env.reward_manager.get_term_cfg(term_name)
        if term_cfg.weight != end_weight:
            term_cfg.weight = end_weight
            env.reward_manager.set_term_cfg(term_name, term_cfg)
        return

    # During the transition period - calculate progress (0 to 1)
    progress = (current_step - start_step) / num_steps

    # Apply different interpolation curves
    if curve_type == "quadratic":
        progress = progress * progress
    elif curve_type == "cubic":
        progress = progress * progress * progress
    elif curve_type == "logarithmic":
        # ease-out style
        progress = 1 - (1 - progress) * (1 - progress)
    elif curve_type == "exp":
        import math
        progress = (math.exp(progress) - 1) / (math.e - 1)
    # else: linear

    # Interpolate weight
    new_weight = start_weight + progress * (end_weight - start_weight)
    term_cfg = env.reward_manager.get_term_cfg(term_name)
    term_cfg.weight = new_weight
    env.reward_manager.set_term_cfg(term_name, term_cfg)

