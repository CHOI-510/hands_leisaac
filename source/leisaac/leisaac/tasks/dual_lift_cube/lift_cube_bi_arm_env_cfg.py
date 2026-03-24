import isaaclab.sim as sim_utils
import torch
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass
from leisaac.enhance.envs.manager_based_rl_digital_twin_env_cfg import (
    ManagerBasedRLDigitalTwinEnvCfg,
)
from leisaac.utils.domain_randomization import (
    domain_randomization,
    randomize_camera_uniform,
    randomize_object_uniform,
)
from leisaac.utils.env_utils import delete_attribute

from ..template import (
    BiArmTaskEnvCfg,
    BiArmTaskSceneCfg,
    BiArmObservationsCfg,
    BiArmActionsCfg,
    BiArmTerminationsCfg,
)
from . import mdp


@configclass
class LiftCubeBiArmSceneCfg(BiArmTaskSceneCfg):
    """Scene configuration for the lift cube bi-arm task."""

    # Use procedural assets so this task does not depend on external USD files.
    scene: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Scene",
        spawn=sim_utils.CuboidCfg(
            size=(0.001, 0.001, 0.001),
            visual_material=sim_utils.PreviewSurfaceCfg(opacity=0.0),
        ),
    )

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.CuboidCfg(
            size=(0.7, 0.7, 0.08),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.25, 0.25, 0.25)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.4, 0.0, 0.77)),
    )

    cube: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.05, 0.05, 0.05),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.08),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.1, 0.1)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.4, 0.0, 0.845)),
    )

    front: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Right_Robot/base/front_camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(-0.6, -0.75, 0.38), rot=(0.77337, 0.55078, -0.2374, -0.20537), convention="opengl"
        ),  # wxyz
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=40.6,
            focus_distance=400.0,
            horizontal_aperture=38.11,
            clipping_range=(0.01, 50.0),
            lock_camera=True,
        ),
        width=640,
        height=480,
        update_period=1 / 30.0,  # 30FPS
    )

    light = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=1000.0),
    )

    def __post_init__(self):
        super().__post_init__()
        delete_attribute(self, "top")


@configclass
class ObservationsCfg(BiArmObservationsCfg):

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        left_joint_pos = ObsTerm(func=mdp.joint_pos, params={"asset_cfg": SceneEntityCfg("left_arm")})
        left_joint_vel = ObsTerm(func=mdp.joint_vel, params={"asset_cfg": SceneEntityCfg("left_arm")})
        left_joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("left_arm")})
        left_joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("left_arm")})
        left_joint_pos_target = ObsTerm(func=mdp.joint_pos_target, params={"asset_cfg": SceneEntityCfg("left_arm")})

        right_joint_pos = ObsTerm(func=mdp.joint_pos, params={"asset_cfg": SceneEntityCfg("right_arm")})
        right_joint_vel = ObsTerm(func=mdp.joint_vel, params={"asset_cfg": SceneEntityCfg("right_arm")})
        right_joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("right_arm")})
        right_joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("right_arm")})
        right_joint_pos_target = ObsTerm(func=mdp.joint_pos_target, params={"asset_cfg": SceneEntityCfg("right_arm")})

        actions = ObsTerm(func=mdp.last_action)
        left_wrist = ObsTerm(
            func=mdp.image, params={"sensor_cfg": SceneEntityCfg("left_wrist"), "data_type": "rgb", "normalize": False}
        )
        right_wrist = ObsTerm(
            func=mdp.image, params={"sensor_cfg": SceneEntityCfg("right_wrist"), "data_type": "rgb", "normalize": False}
        )
        front = ObsTerm(
            func=mdp.image, params={"sensor_cfg": SceneEntityCfg("front"), "data_type": "rgb", "normalize": False}
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class SubtaskCfg(ObsGroup):
        """Observations for subtask group."""

        pick_cube_left = ObsTerm(
            func=mdp.object_grasped,
            params={
                "robot_cfg": SceneEntityCfg("left_arm"),
                "ee_frame_cfg": SceneEntityCfg("left_ee_frame"),
                "object_cfg": SceneEntityCfg("cube"),
            },
        )

        pick_cube_right = ObsTerm(
            func=mdp.object_grasped,
            params={
                "robot_cfg": SceneEntityCfg("right_arm"),
                "ee_frame_cfg": SceneEntityCfg("right_ee_frame"),
                "object_cfg": SceneEntityCfg("cube"),
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    subtask_terms: SubtaskCfg = SubtaskCfg()

    def __post_init__(self):
        super().__post_init__()
        delete_attribute(self.policy, "top")


@configclass
class ActionsCfg(BiArmActionsCfg):
    """Configuration for the bi-arm actions."""

    left_arm_action = mdp.JointPositionActionCfg(
        asset_name="left_arm", joint_names=[f"left_arm_joint_{i}" for i in range(6)], scale=1.0, offset=0.0
    )
    left_gripper_action = mdp.BinaryJointActionCfg(asset_name="left_arm", joint_names=["left_gripper_joint"])
    
    right_arm_action = mdp.JointPositionActionCfg(
        asset_name="right_arm", joint_names=[f"right_arm_joint_{i}" for i in range(6)], scale=1.0, offset=0.0
    )
    right_gripper_action = mdp.BinaryJointActionCfg(asset_name="right_arm", joint_names=["right_gripper_joint"])


@configclass
class TerminationsCfg(BiArmTerminationsCfg):

    success = DoneTerm(
        func=mdp.cube_height_above_base,
        params={
            "cube_cfg": SceneEntityCfg("cube"),
            "left_robot_cfg": SceneEntityCfg("left_arm"),
            "right_robot_cfg": SceneEntityCfg("right_arm"),
            "robot_base_name": "base",
            "height_threshold": 0.20,
        },
    )


@configclass
class LiftCubeBiArmEnvCfg(BiArmTaskEnvCfg):
    """Configuration for the lift cube bi-arm environment."""

    scene: LiftCubeBiArmSceneCfg = LiftCubeBiArmSceneCfg(env_spacing=8.0)

    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    task_description: str = "Lift the red cube up with bi-arm robot."

    def __post_init__(self) -> None:
        super().__post_init__()

        self.viewer.eye = (-0.4, -0.6, 0.5)
        self.viewer.lookat = (0.9, 0.0, -0.3)

        self.scene.left_arm.init_state.pos = (0.35, -0.64, 0.01)
        self.scene.right_arm.init_state.pos = (0.45, -0.64, 0.01)

        domain_randomization(
            self,
            random_options=[
                randomize_object_uniform(
                    "cube",
                    pose_range={
                        "x": (-0.075, 0.075),
                        "y": (-0.075, 0.075),
                        "z": (0.0, 0.0),
                        "yaw": (-30 * torch.pi / 180, 30 * torch.pi / 180),
                    },
                ),
                randomize_camera_uniform(
                    "front",
                    pose_range={
                        "x": (-0.005, 0.005),
                        "y": (-0.005, 0.005),
                        "z": (-0.005, 0.005),
                        "roll": (-0.05 * torch.pi / 180, 0.05 * torch.pi / 180),
                        "pitch": (-0.05 * torch.pi / 180, 0.05 * torch.pi / 180),
                        "yaw": (-0.05 * torch.pi / 180, 0.05 * torch.pi / 180),
                    },
                    convention="opengl",
                ),
            ],
        )


@configclass
class LiftCubeBiArmDigitalTwinEnvCfg(LiftCubeBiArmEnvCfg, ManagerBasedRLDigitalTwinEnvCfg):
    """Configuration for the lift cube bi-arm digital twin environment."""

    rgb_overlay_mode: str = "background"

    rgb_overlay_paths: dict[str, str] = {"front": "greenscreen/background-lift-cube.png"}

    render_objects: list[SceneEntityCfg] = [
        SceneEntityCfg("cube"),
        SceneEntityCfg("left_arm"),
        SceneEntityCfg("right_arm"),
    ]
