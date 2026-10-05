"""Hand-written controllers that read privileged state from the env.

They are measuring instruments, not baselines: they bound what the motor
layer can do (oracle) and generate pre-commit histories (blind press).

The controller is an impedance around ``tip + action * action_scale``. In free
space an action moves the tip by roughly that offset per step; in contact it
commands a force of ``stiffness_z * action_scale * action`` into the surface.
"""

from __future__ import annotations

import torch

from .tasks.peg_insert.env import PegInsertEnv
from .tasks.peg_insert.latch import FLOOR


class ScriptedController:
    def __init__(
        self,
        env: PegInsertEnv,
        fast_step: float = 0.008,  # m per policy step, far from the latch plane
        slow_step: float = 0.0015,  # m per policy step, within ``slow_zone`` of it
        slow_zone: float = 0.006,
        xy_tol: float = 0.001,
        contact_force: float = 0.3,  # N; the slow approach alone presses with ~0.6 N
    ):
        self.env = env
        ctrl = env.cfg.control
        self.scale = ctrl.action_scale
        self.newton_per_action = ctrl.stiffness[2] * ctrl.action_scale
        self.fast_step, self.slow_step, self.slow_zone = fast_step, slow_step, slow_zone
        self.xy_tol = xy_tol
        self.contact_force = contact_force
        self._press_force = torch.zeros(env.num_envs, device=env.device)

    def reset(self):
        self._press_force.zero_()

    def _approach(self) -> torch.Tensor:
        """Center over the hole, then descend; slow down before the latch plane so the impact stays small."""
        env, geo = self.env, self.env.cfg.geometry
        rel = env.tip_pos_w - env.hole_pos_w
        action = torch.zeros(env.num_envs, 3, device=env.device)
        action[:, :2] = (-rel[:, :2] / self.scale).clamp(-1, 1)

        above_latch = rel[:, 2] - geo.latch_height
        step = torch.where(above_latch < self.slow_zone, self.slow_step, self.fast_step)
        step = torch.minimum(step, above_latch.clamp(min=self.slow_step))
        centered = torch.linalg.vector_norm(rel[:, :2], dim=-1) < self.xy_tol
        above_socket = rel[:, 2] > geo.hole_depth + 0.003
        # Hold height until centered; never enter the hole off-center.
        action[:, 2] = torch.where(above_socket | centered, -step / self.scale, torch.zeros_like(step))
        return action

    def _in_contact(self) -> torch.Tensor:
        return self.env.wrist_force_w[:, 2] > self.contact_force

    def oracle(self, hold_force: float = 4.0, release_force: float = 18.0) -> torch.Tensor:
        """Knows the world: push through a latch, rest gently on a floor."""
        action = self._approach()
        floor = self.env.latch.world == FLOOR
        force = torch.where(floor, hold_force, release_force)
        action[:, 2] = torch.where(self._in_contact(), -force / self.newton_per_action, action[:, 2])
        return action.clamp(-1, 1)

    def blind_press(self, stop_force: float, ramp: float = 0.5) -> torch.Tensor:
        """Does not know the world: on contact, ramp the commanded force by ``ramp`` N/step up to ``stop_force``."""
        action = self._approach()
        contact = self._in_contact()
        ramped = (self._press_force + ramp).clamp(max=stop_force)
        self._press_force = torch.where(contact, ramped, torch.full_like(ramped, self.contact_force))
        action[:, 2] = torch.where(contact, -self._press_force / self.newton_per_action, action[:, 2])
        return action.clamp(-1, 1)
