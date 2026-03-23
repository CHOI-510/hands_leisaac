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

    _GRIPPER_CLOSE_EPS = 1e-4
    _GRIP_STOP_DISTANCE = 0.040

    def _setup_scene(self):
        super()._setup_scene()

    def _ee_orange_distance(self, ee_frame_name: str) -> torch.Tensor:
        ee_frame = self.scene[ee_frame_name]
        orange = self.scene["orange"]
        end_effector_pos = ee_frame.data.target_pos_w[:, 1, :]
        orange_pos = orange.data.root_pos_w
        return torch.linalg.vector_norm(orange_pos - end_effector_pos, dim=1)

    def _apply_action(self) -> None:
        left_arm_action = self.actions[:, 0:6].clone()
        right_arm_action = self.actions[:, 6:12].clone()

        left_current_gripper = self.scene["left_arm"].data.joint_pos[:, -1]
        right_current_gripper = self.scene["right_arm"].data.joint_pos[:, -1]

        left_closing_cmd = left_arm_action[:, -1] < (left_current_gripper - self._GRIPPER_CLOSE_EPS)
        right_closing_cmd = right_arm_action[:, -1] < (right_current_gripper - self._GRIPPER_CLOSE_EPS)

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

        left_near_orange = self._ee_orange_distance("left_ee_frame") < self._GRIP_STOP_DISTANCE
        right_near_orange = self._ee_orange_distance("right_ee_frame") < self._GRIP_STOP_DISTANCE

        left_stop_close = torch.logical_and(torch.logical_or(left_grasped, left_near_orange), left_closing_cmd)
        right_stop_close = torch.logical_and(torch.logical_or(right_grasped, right_near_orange), right_closing_cmd)

        left_arm_action[:, -1] = torch.where(left_stop_close, left_current_gripper, left_arm_action[:, -1])
        right_arm_action[:, -1] = torch.where(right_stop_close, right_current_gripper, right_arm_action[:, -1])

        self.scene["left_arm"].set_joint_position_target(left_arm_action)
        self.scene["right_arm"].set_joint_position_target(right_arm_action)

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