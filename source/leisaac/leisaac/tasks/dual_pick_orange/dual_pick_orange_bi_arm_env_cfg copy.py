from pathlib import Path

import torch
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.sensors import FrameTransformerCfg, OffsetCfg
from isaaclab.utils import configclass
import isaaclab.sim as sim_utils

from leisaac.tasks.template import BiArmTaskDirectEnvCfg, BiArmTaskSceneCfg
from leisaac.utils.constant import ASSETS_ROOT


ORANGE_MESH_USD_PATH = (
    Path(ASSETS_ROOT) / "scenes" / "kitchen_with_orange" / "objects" / "Orange001" / "Orange001.usd"
).as_posix()

@configclass
class DualPickOrangeBiArmSceneCfg(BiArmTaskSceneCfg):
    """까만 책상과 귤이 있는 1인칭 양팔 씬"""
    
    # 빈 공간 꼼수
    scene = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Scene",
        spawn=sim_utils.CuboidCfg(
            size=(0.001, 0.001, 0.001), 
            visual_material=sim_utils.PreviewSurfaceCfg(opacity=0.0) 
        )
    )

    terrain = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg()
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(1.0, 1.0, 1.0)) # 까만 책상을 위해 조명 살짝 업
    )

    # 1. 시크한 까만 책상 (diffuse_color 변경)
    desk = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Desk",
        spawn=sim_utils.CuboidCfg(
            size=(0.6, 1.0, 0.85), 
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True), 
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.15, 0.15)), # 까만색
            collision_props=sim_utils.CollisionPropertiesCfg()
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-0.3, 0.0, 0.4)) 
        # init_state=AssetBaseCfg.InitialStateCfg(pos=(0.3, 0.0, 0.4)) 
    )

    # 2. 귤 (deterministic mesh path)
    orange = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Orange",
        spawn=sim_utils.UsdFileCfg(
            usd_path=ORANGE_MESH_USD_PATH,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=24,
                solver_velocity_iteration_count=8,
                max_depenetration_velocity=0.5,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.01,
                rest_offset=0.0,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.3, 0.0, 0.89)),
    )

    left_ee_frame: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Left_Robot/base",
        debug_vis=False,
        target_frames=[
            FrameTransformerCfg.FrameCfg(prim_path="{ENV_REGEX_NS}/Left_Robot/gripper", name="gripper"),
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Left_Robot/jaw",
                name="jaw",
                offset=OffsetCfg(pos=(-0.021, -0.070, 0.02)),
            ),
        ],
    )

    right_ee_frame: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Right_Robot/base",
        debug_vis=False,
        target_frames=[
            FrameTransformerCfg.FrameCfg(prim_path="{ENV_REGEX_NS}/Right_Robot/gripper", name="gripper"),
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Right_Robot/jaw",
                name="jaw",
                offset=OffsetCfg(pos=(-0.021, -0.070, 0.02)),
            ),
        ],
    )



    # # 2. 귤
    # orange = RigidObjectCfg(
    #     prim_path="{ENV_REGEX_NS}/Orange",
    #     spawn=sim_utils.SphereCfg(
    #         radius=0.035, 
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(),
    #         mass_props=sim_utils.MassPropertiesCfg(mass=0.1), 
    #         visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.4, 0.4)), 
    #         collision_props=sim_utils.CollisionPropertiesCfg()
    #     ),
    #     init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.3, 0.0, 0.85)) 
    # )

    def __post_init__(self):
        super().__post_init__()
        
        ROTATION_FORWARD = (0.7071, 0.0, 0.0, -0.7071)  # 90도 회전 (Z축 기준)
        # ROTATION_FORWARD = (0.7071, 0.0, 0.0, -0.7071)
        
        # 3. 로봇 위치 수정: 내 쪽(책상 앞단 X=0.0)으로 당기고, 좌우(Y축)를 1인칭에 맞게 변경
        self.left_arm.init_state.pos = (0.0, 0.14, 0.3)  # Y+가 왼쪽
        self.left_arm.init_state.rot = ROTATION_FORWARD

        self.right_arm.init_state.pos = (0.0, -0.14, 0.3) # Y-가 오른쪽
        self.right_arm.init_state.rot = ROTATION_FORWARD


@configclass
class DualPickOrangeBiArmEnvCfg(BiArmTaskDirectEnvCfg):
    scene: DualPickOrangeBiArmSceneCfg = DualPickOrangeBiArmSceneCfg(env_spacing=8.0)
    task_description: str = "Black desk with an orange."

    def __post_init__(self) -> None:
        super().__post_init__()

        orange_mesh_ok = Path(ORANGE_MESH_USD_PATH).is_file()
        if orange_mesh_ok:
            print(f"[INFO] Orange mesh configured: {ORANGE_MESH_USD_PATH}")
        else:
            print(f"[WARN] Orange mesh path not found: {ORANGE_MESH_USD_PATH}")
        print(
            "[INFO] Orange collision tuning: "
            "solver_pos_iter=24, solver_vel_iter=8, max_depenetration_velocity=0.5, "
            "contact_offset=0.01, rest_offset=0.0"
        )

        # 4. 카메라 시점: 로봇 등 뒤에서(X=-0.6) 귤(X=0.3)을 내려다보도록 세팅!
        self.viewer.eye = (-0.6, 0.0, 1.2)
        self.viewer.lookat = (0.3, 0.0, 0.8)

        self.sim.render.antialiasing_mode = "FXAA"
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.friction_correlation_distance = 0.00625
        self.decimation = 1
        self.dynamic_reset_gripper_effort_limit = True

# 엔트리 포인트
DualPickOrangeBiArmEnvCfg_PLAY = DualPickOrangeBiArmEnvCfg