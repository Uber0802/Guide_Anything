"""Held-out classifiers that try to tell the two worlds apart from observation histories.

The aliasing certificate passes when none of them beats chance on held-out
episodes. Pure torch, so it runs without Isaac Sim.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass
class HoldoutResult:
    accuracy: float
    n_test: int

    @property
    def stderr(self) -> float:
        return math.sqrt(0.25 / self.n_test)  # under the null of chance accuracy

    def within(self, tol: float) -> bool:
        return abs(self.accuracy - 0.5) <= tol


def split(labels: torch.Tensor, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Half of each class to train, half to test."""
    g = torch.Generator().manual_seed(seed)
    train, test = [], []
    for c in labels.unique():
        idx = (labels == c).nonzero().flatten()
        idx = idx[torch.randperm(len(idx), generator=g).to(idx.device)]
        train.append(idx[: len(idx) // 2])
        test.append(idx[len(idx) // 2 :])
    return torch.cat(train), torch.cat(test)


def _standardize(train: torch.Tensor, test: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean, std = train.mean(0), train.std(0).clamp(min=1e-8)
    return (train - mean) / std, (test - mean) / std


def _fit(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, weight_decay: float, epochs: int, seed: int):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=weight_decay)
    for _ in range(epochs):
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(model(x).squeeze(-1), y.float())
        loss.backward()
        opt.step()


def holdout_accuracy(
    features: torch.Tensor, labels: torch.Tensor, kind: str, seed: int = 0, epochs: int = 1000
) -> HoldoutResult:
    """Train on half the episodes, report accuracy on the other half.

    Args:
        features: (episodes, dim) flattened histories.
        labels: (episodes,) world index, 0 or 1.
        kind: "logistic" or "mlp".
    """
    train, test = split(labels, seed)
    x_train, x_test = _standardize(features[train].float(), features[test].float())
    dim = features.shape[1]
    if kind == "logistic":
        model = torch.nn.Linear(dim, 1)
        weight_decay = 1e-2
    elif kind == "mlp":
        model = torch.nn.Sequential(torch.nn.Linear(dim, 64), torch.nn.ReLU(), torch.nn.Linear(64, 1))
        weight_decay = 1e-3
    else:
        raise ValueError(f"unknown classifier {kind}")
    model.to(features.device)
    _fit(model, x_train, labels[train], weight_decay, epochs, seed)
    with torch.no_grad():
        pred = model(x_test).squeeze(-1) > 0
    acc = (pred == labels[test].bool()).float().mean().item()
    return HoldoutResult(accuracy=acc, n_test=len(test))
