"""Warm-start an rsl_rl actor-critic by imitating a scripted teacher (DAgger).

The teacher only supplies motor skill. Round 0 rolls out the teacher; later
rounds roll out the student and label its states with the teacher's action.
To cover the states after a press without teaching when to press, a random
fraction of episodes executes commit = +1 regardless of the policy, while the
commit label stays the teacher's.
The actor is regressed onto the aggregated labels and the critic onto the
discounted returns of the latest student rollout, so PPO starts from a policy
whose value estimate already matches it.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from rsl_rl.modules import ActorCritic

from .rsl import RslRlEnv


def _rollout(
    env: RslRlEnv,
    policy: ActorCritic,
    teacher: Callable[[], torch.Tensor],
    student_acts: bool,
    gamma: float,
    commit_dim: int | None,
    press_fraction: float,
):
    obs = env.reset()
    forced = torch.rand(env.num_envs, device=env.device) < press_fraction
    observations, critic_observations, labels, rewards = [], [], [], []
    while True:
        policy.update_normalization(obs)
        target = teacher()
        with torch.no_grad():
            action = policy.act_inference(obs) if student_acts else target.clone()
        if commit_dim is not None:
            action[forced, commit_dim] = 1.0
        observations.append(policy.get_actor_obs(obs).clone())
        critic_observations.append(policy.get_critic_obs(obs).clone())
        labels.append(target.clone())
        obs, reward, dones, _ = env.step(action)
        rewards.append(reward.clone())
        if dones.any():
            assert dones.all(), "warm start assumes episodes of equal length"
            break
    returns = torch.zeros_like(rewards[-1])
    targets = []
    for r in reversed(rewards):
        returns = r + gamma * returns
        targets.append(returns.clone())
    targets.reverse()
    return (
        torch.cat(observations),
        torch.cat(critic_observations),
        torch.cat(labels),
        torch.cat(targets),
        torch.stack(rewards).sum(0).mean().item(),
    )


def _regress(net: torch.nn.Module, normalizer: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, epochs: int):
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    for _ in range(epochs):
        for idx in torch.randperm(len(x), device=x.device).split(8192):
            opt.zero_grad()
            loss = torch.nn.functional.mse_loss(net(normalizer(x[idx])), y[idx])
            loss.backward()
            opt.step()
    return loss.item()


def warm_start(
    env: RslRlEnv,
    policy: ActorCritic,
    teacher: Callable[[], torch.Tensor],
    gamma: float,
    rounds: int = 4,
    epochs: int = 3,
    action_std: float | list[float] = 0.2,
    commit_dim: int | None = None,
    press_fraction: float = 0.0,
):
    data_x, data_y = [], []
    for r in range(rounds):
        x, x_critic, y, returns, episode_return = _rollout(
            env, policy, teacher, r > 0, gamma, commit_dim, press_fraction
        )
        data_x.append(x)
        data_y.append(y)
        actor_loss = _regress(policy.actor, policy.actor_obs_normalizer, torch.cat(data_x), torch.cat(data_y), epochs)
        critic_loss = _regress(policy.critic, policy.critic_obs_normalizer, x_critic, returns.unsqueeze(-1), epochs)
        who = "student" if r > 0 else "teacher"
        print(
            f"[warm start] round {r}: {who} return {episode_return:.3f}"
            f" | actor mse {actor_loss:.4f} | critic mse {critic_loss:.4f} | {len(torch.cat(data_x))} samples"
        )
    with torch.no_grad():
        policy.std.copy_(torch.as_tensor(action_std, dtype=policy.std.dtype, device=policy.std.device))
