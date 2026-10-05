import pytest
import torch

from guide_anything.certificate import holdout_accuracy, split


def data(signal: float, n: int = 1000, dim: int = 50, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    y = torch.arange(n) % 2
    x = torch.randn(n, dim, generator=g)
    x[:, 0] += signal * (2 * y - 1)
    return x, y


def test_split_is_balanced_and_disjoint():
    y = torch.arange(100) % 2
    train, test = split(y, seed=0)
    assert len(set(train.tolist()) & set(test.tolist())) == 0
    assert y[train].sum() == 25 and y[test].sum() == 25


@pytest.mark.parametrize("kind", ["logistic", "mlp"])
def test_chance_without_signal(kind):
    x, y = data(signal=0.0)
    assert holdout_accuracy(x, y, kind, epochs=300).within(0.06)


@pytest.mark.parametrize("kind", ["logistic", "mlp"])
def test_detects_signal(kind):
    x, y = data(signal=1.5)
    assert holdout_accuracy(x, y, kind, epochs=300).accuracy > 0.85
