from __future__ import annotations

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg


def cube_height_above_base(
    env: ManagerBasedRLEnv | DirectRLEnv,
    cube_cfg: SceneEntityCfg,
    robot_cfg: SceneEntityCfg | None = None,
    left_robot_cfg: SceneEntityCfg | None = None,
    right_robot_cfg: SceneEntityCfg | None = None,
    robot_base_name: str = "base",
    height_threshold: float = 0.20,
) -> torch.Tensor:
    """Determine if the cube is above the robot base.

    This function checks whether all success conditions for the task have been met:
    1. cube is above the robot base

    Args:
        env: The RL environment instance.
        cube_cfg: Configuration for the cube entity.
        robot_cfg: Configuration for the single-arm robot entity.
        left_robot_cfg: Configuration for the left robot (bi-arm).
        right_robot_cfg: Configuration for the right robot (bi-arm).
        robot_base_name: Name of the robot base.
        height_threshold: Threshold for the cube height above the robot base.
    Returns:
        Boolean tensor indicating which environments have completed the task.
    """
    done = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    cube: RigidObject = env.scene[cube_cfg.name]
    cube_height = cube.data.root_pos_w[:, 2]

    if robot_cfg is not None:
        robot: Articulation = env.scene[robot_cfg.name]
        base_index = robot.data.body_names.index(robot_base_name)
        robot_base_height = robot.data.body_pos_w[:, base_index, 2]
        done = torch.logical_and(done, cube_height - robot_base_height > height_threshold)
        return done

    if left_robot_cfg is None or right_robot_cfg is None:
        raise ValueError("Provide either robot_cfg or both left_robot_cfg and right_robot_cfg.")

    left_robot: Articulation = env.scene[left_robot_cfg.name]
    right_robot: Articulation = env.scene[right_robot_cfg.name]
    left_base_index = left_robot.data.body_names.index(robot_base_name)
    right_base_index = right_robot.data.body_names.index(robot_base_name)
    left_base_height = left_robot.data.body_pos_w[:, left_base_index, 2]
    right_base_height = right_robot.data.body_pos_w[:, right_base_index, 2]
    base_height = torch.maximum(left_base_height, right_base_height)
    done = torch.logical_and(done, cube_height - base_height > height_threshold)

    return done
