from pathlib import Path
import torch

from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.sensors import FrameTransformerCfg, OffsetCfg
from isaaclab.utils import configclass
import isaaclab.sim as sim_utils
from isaaclab.envs import DirectRLEnv

from leisaac.tasks.template import BiArmTaskDirectEnvCfg, BiArmTaskSceneCfg
from leisaac.utils.constant import ASSETS_ROOT

# 1. 자산 경로
ORANGE_MESH_USD_PATH = (
    Path(ASSETS_ROOT) / "scenes" / "kitchen_with_orange" / "objects" / "Orange001" / "Orange001.usd"
).as_posix()

@configclass
class TerminationsCfg:
    time_out = None

# 2. 씬 설정 (센서 등록 로직 추가)
@configclass
class DualPickOrangeBiArmSceneCfg(BiArmTaskSceneCfg):
    scene = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Scene",
        spawn=sim_utils.CuboidCfg(size=(0.001, 0.001, 0.001), visual_material=sim_utils.PreviewSurfaceCfg(opacity=0.0))
    )
    terrain = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(prim_path="/World/light", spawn=sim_utils.DomeLightCfg(intensity=2500.0))

    desk = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Desk",
        spawn=sim_utils.CuboidCfg(
            size=(0.6, 1.0, 0.85), 
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True), 
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.15, 0.15)),
            collision_props=sim_utils.CollisionPropertiesCfg()
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-0.3, 0.0, 0.4)) 
    )

    orange = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Orange",
        spawn=sim_utils.UsdFileCfg(
            usd_path=ORANGE_MESH_USD_PATH,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(solver_position_iteration_count=24, solver_velocity_iteration_count=8),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.01, rest_offset=0.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.3, 0.0, 0.89)),
    )

    # 손 위치 추적 센서 설정
    left_ee_frame: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Left_Robot/base",
        target_frames=[
            FrameTransformerCfg.FrameCfg(prim_path="{ENV_REGEX_NS}/Left_Robot/gripper", name="gripper"),
            FrameTransformerCfg.FrameCfg(prim_path="{ENV_REGEX_NS}/Left_Robot/jaw", name="jaw", offset=OffsetCfg(pos=(-0.021, -0.070, 0.02))),
        ],
    )
    right_ee_frame: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Right_Robot/base",
        target_frames=[
            FrameTransformerCfg.FrameCfg(prim_path="{ENV_REGEX_NS}/Right_Robot/gripper", name="gripper"),
            FrameTransformerCfg.FrameCfg(prim_path="{ENV_REGEX_NS}/Right_Robot/jaw", name="jaw", offset=OffsetCfg(pos=(-0.021, -0.070, 0.02))),
        ],
    )

    def __post_init__(self):
        super().__post_init__()
        
        # ⭐ 핵심 해결 포인트: 센서 이름을 시스템에 강제로 등록 ⭐
        self.left_ee_frame.prim_path = "{ENV_REGEX_NS}/Left_Robot/base"
        self.right_ee_frame.prim_path = "{ENV_REGEX_NS}/Right_Robot/base"

        ROTATION_FORWARD = (0.7071, 0.0, 0.0, -0.7071)
        self.left_arm.init_state.pos = (0.0, 0.14, 0.3)
        self.left_arm.init_state.rot = ROTATION_FORWARD
        self.right_arm.init_state.pos = (0.0, -0.14, 0.3)
        self.right_arm.init_state.rot = ROTATION_FORWARD

# 3. 환경 설정
@configclass
class DualPickOrangeBiArmEnvCfg(BiArmTaskDirectEnvCfg):
    scene: DualPickOrangeBiArmSceneCfg = DualPickOrangeBiArmSceneCfg(env_spacing=8.0)
    task_description: str = "Black desk with an orange."
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        self.viewer.eye = (-0.6, 0.0, 1.2)
        self.viewer.lookat = (0.3, 0.0, 0.8)
        self.sim.render.antialiasing_mode = "FXAA"
        self.decimation = 1
        self.dynamic_reset_gripper_effort_limit = True

# 4. 환경 실행 클래스
class DualPickOrangeBiArmEnv(DirectRLEnv):
    def __init__(self, cfg: DualPickOrangeBiArmEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        # 센서 데이터 접근을 위한 인덱싱
        self.left_ee_sensor = self.scene["left_ee_frame"]
        self.right_ee_sensor = self.scene["right_ee_frame"]
        self.left_gripper_ids, _ = self.scene["left_arm"].find_joints(".*_finger_.*")
        self.right_gripper_ids, _ = self.scene["right_arm"].find_joints(".*_finger_.*")

    def step(self, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        obs, rew, terminated, truncated, info = super().step(action)
        if self.cfg.dynamic_reset_gripper_effort_limit:
            limit_val = torch.tensor([10.0], device=self.device)
            self.scene["left_arm"].write_joint_effort_limit(limit_val.repeat(len(self.left_gripper_ids)), self.left_gripper_ids)
            self.scene["right_arm"].write_joint_effort_limit(limit_val.repeat(len(self.right_gripper_ids)), self.right_gripper_ids)
        return obs, rew, terminated, truncated, info

DualPickOrangeBiArmEnvCfg_PLAY = DualPickOrangeBiArmEnvCfg