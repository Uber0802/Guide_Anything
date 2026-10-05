"""Hand-written controllers that read privileged state from the env.

They are measuring instruments, not baselines: they bound what the motor
layer can do (oracle) and generate pre-commit histories (blind press).

The controller is an impedance around ``tip + action * action_scale``. In free
space an action moves the tip by roughly that offset per step; in contact it
commands a force of ``stiffness_z * action_scale * action`` into the surface.
The fourth action is the commit the env reads at first contact; these
controllers set it from what they intend to do, before they touch.
"""

from __future__ import annotations

import torch

from .tasks.peg_insert.env import PegInsertEnv
from .tasks.peg_insert.latch import FLOOR

# Commit value when not pressing. Any negative value means "no". The env samples the decision
# once per episode, so this sits under one policy standard deviation from zero: a warm-started
# policy still presses in a fair share of episodes.
NO_COMMIT = -0.2


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

    def _press(self, action: torch.Tensor, force: torch.Tensor | float, commit: torch.Tensor | bool) -> torch.Tensor:
        """In contact, replace the z action by a force command; append the commit decision."""
        n = action.shape[0]
        force = torch.as_tensor(force, device=action.device).expand(n)
        commit = torch.as_tensor(commit, device=action.device).expand(n)
        action[:, 2] = torch.where(self._in_contact(), -force / self.newton_per_action, action[:, 2])
        commit_value = torch.where(commit, 1.0, NO_COMMIT)
        return torch.cat([action, commit_value.unsqueeze(-1)], dim=-1).clamp(-1, 1)

    def _in_contact(self) -> torch.Tensor:
        return self.env.wrist_force_w[:, 2] > self.contact_force

    def oracle(self, hold_force: float = 4.0, release_force: float = 18.0) -> torch.Tensor:
        """Knows the world: push through a latch, rest gently on a floor."""
        press = self.env.latch.world != FLOOR
        return self._press(self._approach(), torch.where(press, release_force, hold_force), press)

    def by_belief(self, threshold: float, hold_force: float = 4.0, release_force: float = 18.0) -> torch.Tensor:
        """Oracle motor, but the decision comes from the belief: press through iff p > threshold."""
        press = self.env.belief > threshold
        return self._press(self._approach(), torch.where(press, release_force, hold_force), press)

    def blind_press(self, stop_force: float, ramp: float = 0.5) -> torch.Tensor:
        """Does not know the world: on contact, ramp the commanded force by ``ramp`` N/step up to ``stop_force``."""
        ramped = (self._press_force + ramp).clamp(max=stop_force)
        self._press_force = torch.where(self._in_contact(), ramped, torch.full_like(ramped, self.contact_force))
        return self._press(self._approach(), self._press_force, stop_force > self.env.cfg.control.safe_force)
