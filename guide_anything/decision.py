"""Summaries of the press-or-stop decision as a function of the belief p = P(LATCH)."""

from __future__ import annotations

import torch


def optimal_threshold(success: float, damage: float) -> float:
    """Press iff p * success - (1 - p) * damage > (1 - p) * success."""
    return (success + damage) / (2 * success + damage)


def optimal_return(belief: torch.Tensor, success: float, damage: float) -> torch.Tensor:
    """Expected outcome of the Bayes decision with a perfect motor, per episode."""
    press = belief * success - (1 - belief) * damage
    stop = (1 - belief) * success
    return torch.maximum(press, stop)


def fit_threshold(belief: torch.Tensor, pressed: torch.Tensor, resolution: int = 1001) -> tuple[float, float]:
    """Best step function: the t minimizing the disagreement between ``pressed`` and ``p > t``.

    Returns (t, disagreement rate). Ties resolve to the middle of the minimizing range.
    """
    grid = torch.linspace(0.0, 1.0, resolution, device=belief.device)
    disagree = ((belief.unsqueeze(0) > grid.unsqueeze(1)) != pressed.bool().unsqueeze(0)).float().mean(1)
    best = torch.nonzero(disagree == disagree.min()).flatten()
    t = 0.5 * (grid[best[0]] + grid[best[-1]])
    return t.item(), disagree.min().item()


def binned(belief: torch.Tensor, values: torch.Tensor, bins: int = 10) -> list[tuple[float, float, int]]:
    """(bin center, mean value, count) over equal-width bins of p in [0, 1]."""
    idx = (belief * bins).long().clamp(max=bins - 1)
    out = []
    for b in range(bins):
        mask = idx == b
        n = int(mask.sum())
        out.append(((b + 0.5) / bins, values[mask].float().mean().item() if n else float("nan"), n))
    return out
