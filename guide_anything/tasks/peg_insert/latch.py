"""Virtual latch on the insertion axis.

Two worlds share one contact model up to the damage threshold, so the
force/position history before that point carries no information about which
world the robot is in.

    world LATCH: a detent at depth d that releases once the resisting force
                 reaches ``release_force``; the peg then slides to the bottom.
    world FLOOR: a hard stop at depth d; the part is damaged once the
                 resisting force exceeds ``damage_force``.

``damage_force < release_force`` is what makes the pair aliased: any history
that could tell the worlds apart in FLOOR has already damaged the part.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

LATCH = 0
FLOOR = 1


@dataclass
class LatchParams:
    stiffness: float = 1.0e4  # N/m
    damping: float = 50.0  # N s/m, only while engaged and moving down
    release_force: float = 15.0  # N
    damage_force: float = 10.0  # N

    def __post_init__(self):
        if not self.damage_force < self.release_force:
            raise ValueError("damage_force must be below release_force, or the worlds are not aliased")


class VirtualLatch:
    """Batched latch state. All tensors have shape (num_envs,)."""

    def __init__(self, num_envs: int, params: LatchParams, device: str | torch.device):
        self.params = params
        self.world = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.released = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.damaged = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.force = torch.zeros(num_envs, device=device)
        self.peak_force = torch.zeros(num_envs, device=device)  # largest resisting force this episode

    @property
    def pressed(self) -> torch.Tensor:
        """Pushed past the damage force: the irreversible commitment, whichever the world."""
        return self.peak_force > self.params.damage_force

    def reset(self, env_ids: torch.Tensor, world: torch.Tensor):
        self.world[env_ids] = world
        self.released[env_ids] = False
        self.damaged[env_ids] = False
        self.force[env_ids] = 0.0
        self.peak_force[env_ids] = 0.0

    def step(self, penetration: torch.Tensor, down_speed: torch.Tensor) -> torch.Tensor:
        """Advance one physics step and return the upward force on the peg.

        Args:
            penetration: how far the peg tip is past the latch plane, in m. Positive below it.
            down_speed: peg tip speed along the insertion direction, in m/s. Positive moving down.
        """
        p = self.params
        engaged = (penetration > 0.0) & ~self.released
        force = p.stiffness * penetration.clamp(min=0.0) + p.damping * down_speed.clamp(min=0.0)
        force = torch.where(engaged, force, torch.zeros_like(force))
        self.peak_force = torch.maximum(self.peak_force, force)

        is_floor = self.world == FLOOR
        self.damaged |= is_floor & (force > p.damage_force)
        self.released |= ~is_floor & (force >= p.release_force)
        # The detent stops pushing back on the step it gives way.
        self.force = torch.where(self.released, torch.zeros_like(force), force)
        return self.force
