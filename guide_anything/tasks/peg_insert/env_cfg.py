"""Configuration for the square peg-in-hole task with a virtual latch.

Geometry is built from primitives so every dimension that may later become a
hidden factor (clearance, depth, friction, latch depth) is a config field.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

from .latch import LatchParams


@configclass
class GeometryCfg:
    peg_size: float = 0.008  # side of the square peg, m
    peg_length: float = 0.06
    peg_tip_offset: float = 0.15  # hand frame origin to peg tip, along the hand z axis
    clearance: float = 0.001  # hole side minus peg side; generous, the study is about the decision, not precision
    hole_depth: float = 0.025
    socket_width: float = 0.05  # outer side of the socket block
    socket_pos: tuple[float, float] = (0.55, 0.0)  # in the env frame, on the ground
    latch_height: float = 0.012  # latch plane above the hole bottom
    friction: float = 0.5


@configclass
class ControlCfg:
    """Task-space impedance control of the peg tip position. Orientation is held fixed."""

    stiffness: tuple[float, float, float] = (400.0, 400.0, 400.0)  # N/m
    # Nm/rad about x, y, z. Torques are explicit at the physics rate, so each axis needs
    # sqrt(K / Lambda) * dt well below 1. Yaw inertia at the hand is only ~0.003 kg m^2.
    rot_stiffness: tuple[float, float, float] = (20.0, 20.0, 5.0)
    damping_ratio: float = 1.0
    action_scale: float = 0.05  # m, max offset of the target from the current tip per policy step
    # Without commitment the downward offset is clamped so the commanded force stays below this, under
    # the damage force. The commit (action[3] > 0) is read once, when the wrist first reads more than
    # commit_contact_force upward; if set, the arm then presses with the full stiffness * action_scale.
    safe_force: float = 8.0  # N
    commit_contact_force: float = 0.3  # N
    nullspace_stiffness: float = 10.0


@configclass
class ResetCfg:
    tip_height: float = 0.04  # initial tip height above the socket top
    tip_xy_noise: float = 0.01  # uniform, m
    hole_obs_noise: float = 0.0  # uniform noise on the observed hole position, m


@configclass
class RewardCfg:
    """Outcome on the last step, plus potential-based shaping toward the hole entrance.

    Every episode runs the full horizon and the outcome is paid only on its last
    step, so discounting scales all outcomes alike and the optimal decision is
    to press iff p > (success + damage) / (2 * success + damage), 0.8 for L = 3.
    """

    success: float = 1.0
    damage: float = 3.0  # L, the stakes
    shaping: float = 10.0  # per m of distance to the hole axis at the latch plane
    gamma: float = 0.99  # must equal the learner's discount for the shaping to leave the optimum unchanged


@configclass
class BeliefCfg:
    """How the belief p = P(LATCH) shown to the policy is generated at reset.

    calibrated: p ~ U(0, 1) and the world is drawn with P(LATCH) = p. This is the joint law of a
        50/50 world and a message whose accuracy is uniform on [0.5, 1].
    oracle: p is 1 in LATCH and 0 in FLOOR.
    prior: p = 0.5 and the world is a fair coin. Nothing about the world is revealed.
    """

    mode: str = "calibrated"


@configclass
class PegInsertEnvCfg(DirectRLEnvCfg):
    decimation: int = 8
    episode_length_s: float = 10.0
    # tip offset (3), commit (1): read at first contact; if > 0, press down at full force from then on
    action_space: int = 4
    # tip minus hole (3), tip velocity (3), wrist force / 10 (3), previous tip offset action (3), belief (1),
    # time (1)
    observation_space: int = 14
    state_space: int = 0

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=8,
        gravity=(0.0, 0.0, -9.81),
        physx=PhysxCfg(
            solver_type=1,
            max_position_iteration_count=192,
            max_velocity_iteration_count=1,
            bounce_threshold_velocity=0.2,
            friction_offset_threshold=0.01,
            friction_correlation_distance=0.00625,
            gpu_max_rigid_contact_count=2**23,
            gpu_max_rigid_patch_count=2**23,
        ),
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.5, dynamic_friction=0.5),
    )
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=128, env_spacing=1.5, replicate_physics=True)

    robot: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/FrankaEmika/panda_instanceable.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True, max_depenetration_velocity=5.0),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False, solver_position_iteration_count=192, solver_velocity_iteration_count=1
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                # Tip 4 cm above the socket, hand pointing down. Solved by scripts/solve_home_pose.py.
                "panda_joint1": 0.0046,
                "panda_joint2": 0.3506,
                "panda_joint3": -0.0045,
                "panda_joint4": -2.2233,
                "panda_joint5": 0.0029,
                "panda_joint6": 2.5741,
                "panda_joint7": 0.7835,
                "panda_finger_joint.*": 0.004,
            },
        ),
        actuators={
            # Arm joints are torque controlled by the task-space controller.
            "arm": ImplicitActuatorCfg(joint_names_expr=["panda_joint[1-7]"], stiffness=0.0, damping=0.0),
            "fingers": ImplicitActuatorCfg(
                joint_names_expr=["panda_finger_joint.*"], effort_limit_sim=200.0, stiffness=2e3, damping=1e2
            ),
        },
    )

    geometry: GeometryCfg = GeometryCfg()
    control: ControlCfg = ControlCfg()
    reset: ResetCfg = ResetCfg()
    reward: RewardCfg = RewardCfg()
    belief: BeliefCfg = BeliefCfg()
    latch: LatchParams = LatchParams()

    success_xy_tol: float = 0.0025
    success_depth_tol: float = 0.002
