import pytest
import torch

from guide_anything.decision import fit_threshold, optimal_return, optimal_threshold


def test_threshold_matches_stakes():
    assert optimal_threshold(1.0, 3.0) == pytest.approx(0.8)
    assert optimal_threshold(1.0, 1.0) == pytest.approx(2 / 3)


def test_optimal_return_uniform_belief():
    p = torch.linspace(0, 1, 100001)
    assert optimal_return(p, 1.0, 3.0).mean().item() == pytest.approx(0.6, abs=1e-3)


def test_fit_recovers_step():
    g = torch.Generator().manual_seed(0)
    p = torch.rand(4000, generator=g)
    t, err = fit_threshold(p, p > 0.73)
    assert t == pytest.approx(0.73, abs=0.005) and err == 0.0


def test_fit_with_label_noise():
    g = torch.Generator().manual_seed(0)
    p = torch.rand(4000, generator=g)
    flip = torch.rand(4000, generator=g) < 0.1
    t, err = fit_threshold(p, (p > 0.8) ^ flip)
    assert t == pytest.approx(0.8, abs=0.03) and err == pytest.approx(0.1, abs=0.02)
