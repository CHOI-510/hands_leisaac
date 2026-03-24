import gymnasium as gym

gym.register(
    id="LeIsaac-SO101-DualLiftCube-BiArm-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.lift_cube_bi_arm_env_cfg:LiftCubeBiArmEnvCfg",
    },
)

gym.register(
    id="LeIsaac-SO101-DualLiftCube-BiArm-DigitalTwin-v0",
    entry_point="leisaac.enhance.envs:ManagerBasedRLDigitalTwinEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.lift_cube_bi_arm_env_cfg:LiftCubeBiArmDigitalTwinEnvCfg",
    },
)

gym.register(
    id="LeIsaac-SO101-DualLiftCube-BiArm-Mimic-v0",
    entry_point=f"leisaac.enhance.envs:ManagerBasedRLLeIsaacMimicEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.lift_cube_mimic_env_cfg:LiftCubeMimicEnvCfg",
    },
)
