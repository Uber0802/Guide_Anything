import pytest
import torch

from guide_anything.tasks.peg_insert.latch import FLOOR, LATCH, LatchParams, VirtualLatch


def make(worlds):
    latch = VirtualLatch(len(worlds), LatchParams(), "cpu")
    latch.reset(torch.arange(len(worlds)), torch.tensor(worlds))
    return latch


def press(latch, depths):
    """Push both envs through the same depths quasi-statically and record forces."""
    forces = []
    for d in depths:
        pen = torch.full((latch.world.numel(),), d)
        forces.append(latch.step(pen, torch.zeros_like(pen)).clone())
    return torch.stack(forces)


def test_no_force_above_latch_plane():
    latch = make([LATCH, FLOOR])
    assert torch.all(press(latch, [-0.01, -0.001, 0.0]) == 0)


def test_worlds_identical_below_damage_force():
    latch = make([LATCH, FLOOR])
    k, dmg = latch.params.stiffness, latch.params.damage_force
    forces = press(latch, torch.linspace(0, dmg / k, 20).tolist())
    assert torch.equal(forces[:, 0], forces[:, 1])
    assert not latch.damaged.any() and not latch.released.any()


def test_floor_damages_before_latch_releases():
    latch = make([LATCH, FLOOR])
    k = latch.params.stiffness
    press(latch, [11.0 / k])
    assert latch.damaged.tolist() == [False, True]
    assert latch.released.tolist() == [False, False]


def test_latch_releases_and_stays_released():
    latch = make([LATCH, FLOOR])
    k = latch.params.stiffness
    forces = press(latch, [15.0 / k, 0.02, 0.001])
    assert latch.released.tolist() == [True, False]
    assert torch.all(forces[:, 0] == 0)
    assert latch.damaged.tolist() == [False, True]


def test_damping_only_when_moving_down():
    latch = make([LATCH])
    pen = torch.tensor([1e-4])
    up = latch.step(pen, torch.tensor([-1.0])).item()
    down = latch.step(pen, torch.tensor([0.05])).item()
    assert up == pytest.approx(1.0)
    assert down == pytest.approx(1.0 + 2.5)


def test_reset_clears_state():
    latch = make([LATCH, FLOOR])
    press(latch, [0.01])
    latch.reset(torch.tensor([0, 1]), torch.tensor([FLOOR, LATCH]))
    assert not latch.released.any() and not latch.damaged.any()
    assert latch.world.tolist() == [FLOOR, LATCH]


def test_params_must_alias():
    with pytest.raises(ValueError):
        LatchParams(release_force=10.0, damage_force=12.0)


def test_pressed_tracks_peak_force_in_both_worlds():
    latch = make([LATCH, FLOOR])
    k = latch.params.stiffness
    press(latch, [5.0 / k, 0.0])
    assert latch.pressed.tolist() == [False, False]
    press(latch, [12.0 / k])
    assert latch.pressed.tolist() == [True, True]
