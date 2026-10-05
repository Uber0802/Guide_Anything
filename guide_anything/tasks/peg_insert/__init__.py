import gymnasium as gym

gym.register(
    id="GuideAnything-PegInsert-v0",
    entry_point=f"{__name__}.env:PegInsertEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.env_cfg:PegInsertEnvCfg"},
)
