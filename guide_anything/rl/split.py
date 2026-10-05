"""Freeze the motor dimensions of a warm-started rsl_rl actor and leave the decision to RL.

PPO's gradient is driven by the press-or-stop outcome. Through a shared actor it
also perturbs the motor outputs, and sub-millimetre insertion does not survive
that. After the split, the motor dimensions come from a frozen copy of the
warm-started actor with a frozen tiny std, and only the decision dimensions
(and the critic) are trained. The actor's observation normalizer is frozen too:
the motor was fit to its current statistics, and PPO data would shift them.
"""

from __future__ import annotations

import copy

import torch
from rsl_rl.modules import ActorCritic
from rsl_rl.networks import EmpiricalNormalization


class SplitActor(torch.nn.Module):
    def __init__(self, actor: torch.nn.Module, decision_dims: list[int]):
        super().__init__()
        self.motor = copy.deepcopy(actor).requires_grad_(False)
        self.decision = actor
        self.decision_dims = decision_dims

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.motor(x).clone()
        out[..., self.decision_dims] = self.decision(x)[..., self.decision_dims]
        return out


def freeze_motor(runner, decision_dims: list[int], motor_std: float = 1e-3):
    """Split ``runner.alg.policy.actor`` and rebuild the PPO optimizer over the trainable parameters."""
    policy: ActorCritic = runner.alg.policy
    policy.actor = SplitActor(policy.actor, decision_dims)
    if isinstance(policy.actor_obs_normalizer, EmpiricalNormalization):
        policy.actor_obs_normalizer.until = int(policy.actor_obs_normalizer.count)
    motor_dims = [i for i in range(policy.std.numel()) if i not in decision_dims]
    with torch.no_grad():
        policy.std[motor_dims] = motor_std
    grad_mask = torch.ones_like(policy.std)
    grad_mask[motor_dims] = 0.0
    policy.std.register_hook(lambda g: g * grad_mask)
    trainable = [p for p in policy.parameters() if p.requires_grad]
    runner.alg.optimizer = torch.optim.Adam(trainable, lr=runner.alg.learning_rate)
