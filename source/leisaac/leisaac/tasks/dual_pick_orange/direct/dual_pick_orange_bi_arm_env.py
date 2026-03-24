import torch
from collections.abc import Sequence
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

# 랩실의 양팔 Direct 템플릿 불러오기
from leisaac.tasks.template import BiArmTaskDirectEnv, BiArmTaskDirectEnvCfg
# 방금 만든 씬 설정 불러오기
from leisaac.tasks.dual_pick_orange.dual_pick_orange_bi_arm_env_cfg import DualPickOrangeBiArmSceneCfg
from leisaac.tasks.pick_orange import mdp as pick_orange_mdp

@configclass
class DualPickOrangeBiArmEnvCfg(BiArmTaskDirectEnvCfg):
    """Direct env configuration for the dual pick orange task."""
    
    scene: DualPickOrangeBiArmSceneCfg = DualPickOrangeBiArmSceneCfg(env_spacing=8.0)
    task_description: str = "Dual-arm orange picking task (Direct)."

    def __post_init__(self) -> None:
        super().__post_init__()

        # 카메라 시점
        self.viewer.eye = (1.5, 0.0, 1.5)
        self.viewer.lookat = (0.0, 0.0, 0.5)

        # 양팔 로봇 28cm 간격으로 배치 (높이는 주방 테이블에 맞게 0.8m로 임의 설정)
        self.scene.left_arm.init_state.pos = (0.0, -0.14, 0.8)
        self.scene.right_arm.init_state.pos = (0.0, 0.14, 0.8)

        # 팔 기준으로 책상/오렌지를 함께 이동시켜 상대 배치를 유지
        self.scene.desk.init_state.pos = (-0.25, 0.0, 0.4)
        self.scene.orange.init_state.pos = (-0.25, 0.0, 0.89)

        self.sim.render.antialiasing_mode = "FXAA"
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.friction_correlation_distance = 0.00625
        self.decimation = 1
        self.dynamic_reset_gripper_effort_limit = True

class DualPickOrangeBiArmEnv(BiArmTaskDirectEnv):
    """Direct env for the dual pick orange task."""
    cfg: DualPickOrangeBiArmEnvCfg

    _MAX_GRIPPER_TARGET_STEP: float = 0.06
    """Max per-step target change for each gripper joint during near-object closing."""

    _CLOSE_CLAMP_DISTANCE: float = 0.10
    """Enable clamp only when jaw is near the orange to keep normal open/close responsiveness."""

    def _setup_scene(self):
        super()._setup_scene()

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        super()._pre_physics_step(actions)

        left_gripper_curr = self.scene["left_arm"].data.joint_pos[:, -1]
        right_gripper_curr = self.scene["right_arm"].data.joint_pos[:, -1]
        left_target = self.actions[:, 5]
        right_target = self.actions[:, 11]

        orange_pos = self.scene["orange"].data.root_pos_w
        left_jaw_pos = self.scene["left_ee_frame"].data.target_pos_w[:, 1, :]
        right_jaw_pos = self.scene["right_ee_frame"].data.target_pos_w[:, 1, :]
        left_near_orange = torch.linalg.vector_norm(orange_pos - left_jaw_pos, dim=1) < self._CLOSE_CLAMP_DISTANCE
        right_near_orange = torch.linalg.vector_norm(orange_pos - right_jaw_pos, dim=1) < self._CLOSE_CLAMP_DISTANCE

        left_is_closing = left_target < left_gripper_curr
        right_is_closing = right_target < right_gripper_curr
        left_near_closing = torch.logical_and(left_near_orange, left_is_closing)
        right_near_closing = torch.logical_and(right_near_orange, right_is_closing)

        step = self._MAX_GRIPPER_TARGET_STEP
        left_limited = torch.clamp(left_target, left_gripper_curr - step, left_gripper_curr + step)
        right_limited = torch.clamp(right_target, right_gripper_curr - step, right_gripper_curr + step)
        self.actions[:, 5] = torch.where(left_near_closing, left_limited, left_target)
        self.actions[:, 11] = torch.where(right_near_closing, right_limited, right_target)

    def _get_observations(self) -> dict:
        obs = super()._get_observations()
        obs["subtask_terms"] = {
            "left_pick_orange": pick_orange_mdp.orange_grasped(
                self,
                robot_cfg=SceneEntityCfg("left_arm"),
                ee_frame_cfg=SceneEntityCfg("left_ee_frame"),
                object_cfg=SceneEntityCfg("orange"),
            ),
            "right_pick_orange": pick_orange_mdp.orange_grasped(
                self,
                robot_cfg=SceneEntityCfg("right_arm"),
                ee_frame_cfg=SceneEntityCfg("right_ee_frame"),
                object_cfg=SceneEntityCfg("orange"),
            ),
        }
        return obs

    def _check_success(self) -> torch.Tensor:
        left_grasped = pick_orange_mdp.orange_grasped(
            self,
            robot_cfg=SceneEntityCfg("left_arm"),
            ee_frame_cfg=SceneEntityCfg("left_ee_frame"),
            object_cfg=SceneEntityCfg("orange"),
        )
        right_grasped = pick_orange_mdp.orange_grasped(
            self,
            robot_cfg=SceneEntityCfg("right_arm"),
            ee_frame_cfg=SceneEntityCfg("right_ee_frame"),
            object_cfg=SceneEntityCfg("orange"),
        )
        return torch.logical_or(left_grasped, right_grasped)

    def _reset_idx(self, env_ids: Sequence[int]):
        super()._reset_idx(env_ids)