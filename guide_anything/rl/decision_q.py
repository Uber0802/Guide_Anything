"""A Q-learner for the one-shot press-or-stop decision (checklist 1.2, the non-PPO arm).

The env reads the commit once, at first contact, and the motor is fixed, so the
decision is a contextual bandit: one observation, two actions, one outcome.
Q(o, a) is regressed onto the episode outcome of the action taken; the policy
presses iff Q(o, press) > Q(o, stop).
"""

from __future__ import annotations

import torch

STOP, PRESS = 0, 1


class DecisionQ(torch.nn.Module):
    def __init__(self, obs_dim: int, hidden: int = 64):
        super().__init__()
        self.register_buffer("mean", torch.zeros(obs_dim))
        self.register_buffer("std", torch.ones(obs_dim))
        self.net = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, hidden),
            torch.nn.ELU(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.ELU(),
            torch.nn.Linear(hidden, 2),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net((obs - self.mean) / self.std)

    def press(self, obs: torch.Tensor) -> torch.Tensor:
        q = self(obs)
        return q[:, PRESS] > q[:, STOP]

    def fit(self, obs: torch.Tensor, action: torch.Tensor, outcome: torch.Tensor, epochs: int = 200):
        self.mean.copy_(obs.mean(0))
        self.std.copy_(obs.std(0).clamp(min=1e-3))
        opt = torch.optim.Adam(self.net.parameters(), lr=1e-3)
        for _ in range(epochs):
            for idx in torch.randperm(len(obs), device=obs.device).split(4096):
                opt.zero_grad()
                q = self(obs[idx]).gather(1, action[idx].unsqueeze(1)).squeeze(1)
                torch.nn.functional.mse_loss(q, outcome[idx]).backward()
                opt.step()
