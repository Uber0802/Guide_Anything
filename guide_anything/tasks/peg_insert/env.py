"""Square peg-in-hole with a virtual latch, driven by task-space impedance control.

The peg is a collider rigidly attached to the Franka hand, so the incoming
joint wrench of the hand link is an ideal wrist F/T reading and grasp slip is
out of the picture. The socket is a kinematic body built from four walls; the
ground is the hole bottom. The latch acts on the peg tip along +z.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
import torch
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv
from isaaclab.utils.math import compute_pose_error, quat_apply, sample_uniform

from ...control import impedance_torques
from .env_cfg import PegInsertEnvCfg
from .latch import FLOOR, LATCH, VirtualLatch

# Hand pointing down, fingers along world x.
HAND_DOWN_QUAT = (0.0, 1.0, 0.0, 0.0)


class PegInsertEnv(DirectRLEnv):
    cfg: PegInsertEnvCfg

    def __init__(self, cfg: PegInsertEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._hand_idx = self._robot.find_bodies("panda_hand")[0][0]
        self._arm_ids = self._robot.find_joints("panda_joint[1-7]")[0]
        self._home_q = self._robot.data.default_joint_pos[:, self._arm_ids].clone()

        geo = self.cfg.geometry
        self._hand_quat = torch.tensor(HAND_DOWN_QUAT, device=self.device).repeat(self.num_envs, 1)
        self._tip_offset = torch.tensor([0.0, 0.0, geo.peg_tip_offset], device=self.device).repeat(self.num_envs, 1)

        ctrl = self.cfg.control
        self._stiffness = torch.tensor([*ctrl.stiffness, *ctrl.rot_stiffness], device=self.device)
        self._hand_target_pos = torch.zeros(self.num_envs, 3, device=self.device)  # set every policy step

        self.latch = VirtualLatch(self.num_envs, self.cfg.latch, self.device)
        self.actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self.hole_pos_w = torch.zeros(self.num_envs, 3, device=self.device)  # center of the hole bottom
        self._hole_obs_noise = torch.zeros(self.num_envs, 3, device=self.device)
        self.belief = torch.full((self.num_envs,), 0.5, device=self.device)  # P(LATCH) shown to the policy
        self._potential = torch.zeros(self.num_envs, device=self.device)
        self._last_step = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # Outcome of each env's last finished episode, written just before it resets.
        self.final_world = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.final_belief = torch.zeros(self.num_envs, device=self.device)
        self.final_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.final_damaged = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.final_pressed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.final_tip_error = torch.zeros(self.num_envs, 3, device=self.device)  # tip minus goal, m

    # ------------------------------------------------------------------ scene

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self._spawn_peg()
        self._socket = self._spawn_socket()
        ground = sim_utils.GroundPlaneCfg(physics_material=self._material())
        ground.func("/World/ground", ground)
        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["robot"] = self._robot
        self.scene.rigid_objects["socket"] = self._socket
        light = sim_utils.DomeLightCfg(intensity=2000.0)
        light.func("/World/Light", light)

    def _material(self):
        mu = self.cfg.geometry.friction
        return sim_utils.RigidBodyMaterialCfg(static_friction=mu, dynamic_friction=mu)

    def _collision(self):
        return sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0)

    def _spawn_peg(self):
        geo = self.cfg.geometry
        cfg = sim_utils.CuboidCfg(
            size=(geo.peg_size, geo.peg_size, geo.peg_length),
            collision_props=self._collision(),
            physics_material=self._material(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.5, 0.1)),
        )
        cfg.func(
            f"{self.scene.env_prim_paths[0]}/Robot/panda_hand/Peg",
            cfg,
            translation=(0.0, 0.0, geo.peg_tip_offset - geo.peg_length / 2),
        )

    def _spawn_socket(self) -> RigidObject:
        geo = self.cfg.geometry
        root = f"{self.scene.env_prim_paths[0]}/Socket"
        sim_utils.create_prim(root, "Xform", translation=(*geo.socket_pos, 0.0))
        sim_utils.define_rigid_body_properties(root, sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True))

        h = (geo.peg_size + geo.clearance) / 2  # inner half width
        t = geo.socket_width / 2 - h  # wall thickness
        d = geo.hole_depth
        walls = {
            "WallPosX": ((t, geo.socket_width, d), (h + t / 2, 0.0)),
            "WallNegX": ((t, geo.socket_width, d), (-(h + t / 2), 0.0)),
            "WallPosY": ((2 * h, t, d), (0.0, h + t / 2)),
            "WallNegY": ((2 * h, t, d), (0.0, -(h + t / 2))),
        }
        for name, (size, (x, y)) in walls.items():
            cfg = sim_utils.CuboidCfg(
                size=size,
                collision_props=self._collision(),
                physics_material=self._material(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3, 0.3, 0.35)),
            )
            cfg.func(f"{root}/{name}", cfg, translation=(x, y, d / 2))
        return RigidObject(RigidObjectCfg(prim_path="/World/envs/env_.*/Socket", spawn=None))

    # ------------------------------------------------------------- kinematics

    @property
    def hand_pose_w(self) -> tuple[torch.Tensor, torch.Tensor]:
        pose = self._robot.data.body_link_pose_w[:, self._hand_idx]
        return pose[:, :3], pose[:, 3:7]

    @property
    def tip_pos_w(self) -> torch.Tensor:
        pos, quat = self.hand_pose_w
        return pos + quat_apply(quat, self._tip_offset)

    @property
    def tip_vel_w(self) -> torch.Tensor:
        vel = self._robot.data.body_link_vel_w[:, self._hand_idx]
        _, quat = self.hand_pose_w
        return vel[:, :3] + torch.cross(vel[:, 3:6], quat_apply(quat, self._tip_offset), dim=-1)

    @property
    def wrist_force_w(self) -> torch.Tensor:
        """Force the environment exerts on the tool, in the world frame (N)."""
        incoming = self._robot.data.body_incoming_joint_wrench_b[:, self._hand_idx, :3]
        _, quat = self.hand_pose_w
        return -quat_apply(quat, incoming)

    @property
    def latch_z(self) -> torch.Tensor:
        return self.hole_pos_w[:, 2] + self.cfg.geometry.latch_height

    @property
    def goal_pos_w(self) -> torch.Tensor:
        """Where the tip should end up: the hole bottom past a latch, or the stop at a floor."""
        goal = self.hole_pos_w.clone()
        goal[:, 2] = torch.where(self.latch.world == FLOOR, self.latch_z, goal[:, 2])
        return goal

    @property
    def success(self) -> torch.Tensor:
        err = self.tip_pos_w - self.goal_pos_w
        centered = torch.linalg.vector_norm(err[:, :2], dim=-1) < self.cfg.success_xy_tol
        seated = err[:, 2] < self.cfg.success_depth_tol
        return centered & seated & ~self.latch.damaged

    # ----------------------------------------------------------------- control

    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions.clamp(-1.0, 1.0)
        ctrl = self.cfg.control
        offset = self.actions[:, :3] * ctrl.action_scale
        safe_depth = -ctrl.safe_force / ctrl.stiffness[2]
        pressing = (self.actions[:, 3] > 0) & (self.wrist_force_w[:, 2] > ctrl.commit_contact_force)
        offset[:, 2] = torch.where(pressing, -ctrl.action_scale, offset[:, 2].clamp(min=safe_depth))
        tip_target = self.tip_pos_w + offset
        self._hand_target_pos = tip_target - quat_apply(self._hand_quat, self._tip_offset)

    def _apply_action(self):
        self._apply_latch_force()
        self._robot.set_joint_effort_target(self._arm_torques(), joint_ids=self._arm_ids)

    def _apply_latch_force(self):
        tip, vel = self.tip_pos_w, self.tip_vel_w
        force = self.latch.step(penetration=self.latch_z - tip[:, 2], down_speed=-vel[:, 2])
        forces = torch.zeros(self.num_envs, 1, 3, device=self.device)
        forces[:, 0, 2] = force
        self._robot.instantaneous_wrench_composer.set_forces_and_torques(
            forces=forces, positions=tip.unsqueeze(1), body_ids=[self._hand_idx], is_global=True
        )

    def _arm_torques(self) -> torch.Tensor:
        data = self._robot.data
        pos, quat = self.hand_pose_w
        pos_err, rot_err = compute_pose_error(
            pos, quat, self._hand_target_pos, self._hand_quat, rot_error_type="axis_angle"
        )
        # Fixed base: the Jacobian of body i is at index i - 1. The base frame is aligned with the world.
        jacobian = self._robot.root_physx_view.get_jacobians()[:, self._hand_idx - 1, :, self._arm_ids]
        mass = self._robot.root_physx_view.get_generalized_mass_matrices()[:, self._arm_ids, :][:, :, self._arm_ids]
        ctrl = self.cfg.control
        return impedance_torques(
            jacobian=jacobian,
            mass_matrix=mass,
            pose_error=torch.cat([pos_err, rot_err], dim=-1),
            ee_vel=data.body_link_vel_w[:, self._hand_idx],
            stiffness=self._stiffness,
            damping_ratio=ctrl.damping_ratio,
            joint_pos=data.joint_pos[:, self._arm_ids],
            joint_vel=data.joint_vel[:, self._arm_ids],
            nullspace_target=self._home_q,
            nullspace_stiffness=ctrl.nullspace_stiffness,
        )

    # -------------------------------------------------------------------- MDP

    def _get_observations(self) -> dict:
        hole_obs = self.hole_pos_w + self._hole_obs_noise
        obs = [
            self.tip_pos_w - hole_obs,
            self.tip_vel_w,
            self.wrist_force_w / 10.0,
            self.actions[:, :3],  # motor part only, so the motor's inputs do not depend on the decision
            self.belief.unsqueeze(-1),
            (self.episode_length_buf / self.max_episode_length).unsqueeze(-1),
        ]
        # Privileged state for an asymmetric critic only. A broken FLOOR part looks like an intact one to
        # the wrist sensor, so without this the critic cannot see a damaging press until the episode ends.
        privileged = [
            torch.nn.functional.one_hot(self.latch.world, 2).float(),
            self.latch.damaged.float().unsqueeze(-1),
            self.latch.released.float().unsqueeze(-1),
        ]
        return {"policy": torch.cat(obs, dim=-1), "critic": torch.cat(privileged, dim=-1)}

    def _shaping_potential(self) -> torch.Tensor:
        """Minus the distance to the hole axis at the latch plane. Zero anywhere on the axis below it."""
        d = self.tip_pos_w - self.hole_pos_w
        d[:, 2] = (d[:, 2] - self.cfg.geometry.latch_height).clamp(min=0.0)
        return -self.cfg.reward.shaping * torch.linalg.vector_norm(d, dim=-1)

    def _get_rewards(self) -> torch.Tensor:
        r = self.cfg.reward
        last = self._last_step
        # Potential-based shaping, with the potential of the terminal state taken as zero.
        potential = torch.where(last, torch.zeros_like(self._potential), self._shaping_potential())
        reward = r.gamma * potential - self._potential
        self._potential = potential
        outcome = r.success * self.success.float() - r.damage * self.latch.damaged.float()
        return reward + torch.where(last, outcome, torch.zeros_like(outcome))

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.extras.pop("log", None)  # only present on steps where episodes end; see _reset_idx
        # Every episode runs the full horizon and ends as a true termination: the outcome is paid on
        # the last step, so the learner must not bootstrap past it.
        self._last_step = self.episode_length_buf >= self.max_episode_length - 1
        return self._last_step.clone(), torch.zeros_like(self._last_step)

    def _reset_idx(self, env_ids: torch.Tensor):
        self.final_world[env_ids] = self.latch.world[env_ids]
        self.final_belief[env_ids] = self.belief[env_ids]
        self.final_success[env_ids] = self.success[env_ids]
        self.final_damaged[env_ids] = self.latch.damaged[env_ids]
        self.final_pressed[env_ids] = self.latch.pressed[env_ids]
        self.final_tip_error[env_ids] = (self.tip_pos_w - self.goal_pos_w)[env_ids]
        self.extras["log"] = {
            "success": self.final_success[env_ids].float().mean(),
            "damaged": self.final_damaged[env_ids].float().mean(),
            "pressed": self.final_pressed[env_ids].float().mean(),
        }
        super()._reset_idx(env_ids)
        n = len(env_ids)

        q = self._robot.data.default_joint_pos[env_ids]
        self._robot.write_joint_state_to_sim(q, torch.zeros_like(q), env_ids=env_ids)
        self._robot.set_joint_position_target(q, env_ids=env_ids)  # holds the fingers; arm gains are zero
        self._robot.set_joint_effort_target(
            torch.zeros(n, len(self._arm_ids), device=self.device), self._arm_ids, env_ids
        )

        geo, noise = self.cfg.geometry, self.cfg.reset.tip_xy_noise
        hole = self.scene.env_origins[env_ids].clone()
        hole[:, 0] += geo.socket_pos[0]
        hole[:, 1] += geo.socket_pos[1]
        hole[:, :2] += sample_uniform(-noise, noise, (n, 2), self.device)
        self.hole_pos_w[env_ids] = hole
        socket_pose = torch.cat([hole, torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(n, 1)], dim=-1)
        self._socket.write_root_pose_to_sim(socket_pose, env_ids=env_ids)

        obs_noise = self.cfg.reset.hole_obs_noise
        self._hole_obs_noise[env_ids] = sample_uniform(-obs_noise, obs_noise, (n, 3), self.device)
        self._hole_obs_noise[env_ids, 2] = 0.0

        belief, world = self._sample_belief_and_world(n)
        self.belief[env_ids] = belief
        self.latch.reset(env_ids, world)
        self.actions[env_ids] = 0.0
        self._potential[env_ids] = self._shaping_potential()[env_ids]

    def _sample_belief_and_world(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        mode = self.cfg.belief.mode
        if mode == "calibrated":
            belief = torch.rand(n, device=self.device)
            is_latch = torch.rand(n, device=self.device) < belief
        elif mode == "oracle":
            is_latch = torch.rand(n, device=self.device) < 0.5
            belief = is_latch.float()
        elif mode == "prior":
            belief = torch.full((n,), 0.5, device=self.device)
            is_latch = torch.rand(n, device=self.device) < 0.5
        else:
            raise ValueError(f"unknown belief mode {mode}")
        return belief, torch.where(is_latch, LATCH, FLOOR)
