import gymnasium as gym

gym.register(
    id="LeIsaac-SO101-FoldCloth-BiArm-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.fold_cloth_bi_arm_env_cfg:FoldClothBiArmEnvCfg",
    },
)

gym.register(
    id="LeIsaac-SO101-FoldCloth-BiArm-Direct-v0",
    entry_point=f"{__name__}.direct.fold_cloth_bi_arm_env:FoldClothBiArmEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.direct.fold_cloth_bi_arm_env:FoldClothBiArmEnvCfg",
    },
)

gym.register(
    id="LeIsaac-SO101-DualPickOrange-Direct-v0",
    entry_point="leisaac.tasks.dual_pick_orange.direct.dual_pick_orange_bi_arm_env:DualPickOrangeBiArmEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "leisaac.tasks.dual_pick_orange.direct.dual_pick_orange_bi_arm_env:DualPickOrangeBiArmEnvCfg",
    },
)

gym.register(
    id="LeIsaac-SO101-DualPickOrange-Mimic-v0",
    # 👇 이 부분을 위와 동일한 Direct 환경 클래스로 변경합니다! 👇
    entry_point="leisaac.tasks.dual_pick_orange.direct.dual_pick_orange_bi_arm_env:DualPickOrangeBiArmEnv", 
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dual_pick_orange_env_cfg:DualPickOrangeBiArmEnvCfg",
    },
)