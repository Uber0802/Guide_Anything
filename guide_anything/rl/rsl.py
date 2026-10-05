"""rsl_rl glue: a VecEnv adapter for DirectRLEnv and the PPO configuration."""

from __future__ import annotations

import torch
from rsl_rl.env import VecEnv
from tensordict import TensorDict

from ..tasks.peg_insert.env import PegInsertEnv


class RslRlEnv(VecEnv):
    def __init__(self, env: PegInsertEnv):
        self.env = env
        self.cfg = env.cfg
        self.num_envs = env.num_envs
        self.num_actions = env.cfg.action_space
        self.max_episode_length = env.max_episode_length
        self.device = env.device
        obs, _ = env.reset()
        self._obs = self._wrap(obs)

    @property
    def episode_length_buf(self) -> torch.Tensor:
        return self.env.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor):
        self.env.episode_length_buf = value

    def _wrap(self, obs: dict) -> TensorDict:
        return TensorDict(obs, batch_size=[self.num_envs])

    def get_observations(self) -> TensorDict:
        return self._obs

    def reset(self) -> TensorDict:
        obs, _ = self.env.reset()
        self._obs = self._wrap(obs)
        return self._obs

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        obs, reward, terminated, truncated, extras = self.env.step(actions)
        self._obs = self._wrap(obs)
        extras["time_outs"] = truncated
        return self._obs, reward, (terminated | truncated).long(), extras


def ppo_config(
    seed: int,
    gamma: float,
    lam: float = 0.95,
    init_noise_std: float = 1.0,
    entropy_coef: float = 0.005,
    learning_rate: float = 1e-3,
    desired_kl: float = 0.01,
    schedule: str = "adaptive",
    num_steps_per_env: int = 32,
    save_interval: int = 25,
) -> dict:
    return {
        "seed": seed,
        "num_steps_per_env": num_steps_per_env,
        "save_interval": save_interval,
        "obs_groups": {"policy": ["policy"], "critic": ["policy", "critic"]},
        "policy": {
            "class_name": "ActorCritic",
            "init_noise_std": init_noise_std,
            "noise_std_type": "scalar",
            "actor_obs_normalization": True,
            # Off: the privileged flags are constant during a never-press warm start, so a running
            # normalizer would blow them up the first time they change.
            "critic_obs_normalization": False,
            "actor_hidden_dims": [256, 128, 64],
            "critic_hidden_dims": [256, 128, 64],
            "activation": "elu",
        },
        "algorithm": {
            "class_name": "PPO",
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "clip_param": 0.2,
            "gamma": gamma,
            "lam": lam,
            "value_loss_coef": 1.0,
            "entropy_coef": entropy_coef,
            "learning_rate": learning_rate,
            "max_grad_norm": 1.0,
            "use_clipped_value_loss": True,
            "schedule": schedule,
            "desired_kl": desired_kl,
        },
    }
