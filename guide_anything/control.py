"""Task-space impedance control for a redundant arm.

    tau = J^T (K e - D v) + N^T M (k_n (q_n - q) - d_n qd)

K is a true stiffness (N/m, Nm/rad), so an offset of the target from the
current pose maps to a known contact force. D is chosen per axis for the
requested damping ratio given the task-space inertia. The null-space term uses
the dynamically consistent projector, so it does not push the end effector.
"""

from __future__ import annotations

import torch


def impedance_torques(
    jacobian: torch.Tensor,  # (N, 6, n)
    mass_matrix: torch.Tensor,  # (N, n, n)
    pose_error: torch.Tensor,  # (N, 6) position error and axis-angle error, target minus current
    ee_vel: torch.Tensor,  # (N, 6) linear and angular velocity
    stiffness: torch.Tensor,  # (6,)
    damping_ratio: float,
    joint_pos: torch.Tensor,  # (N, n)
    joint_vel: torch.Tensor,  # (N, n)
    nullspace_target: torch.Tensor,  # (N, n)
    nullspace_stiffness: float,
) -> torch.Tensor:
    m_inv = torch.linalg.inv(mass_matrix)
    jt = jacobian.mT
    task_inertia = torch.linalg.inv(jacobian @ m_inv @ jt)  # (N, 6, 6)

    damping = 2.0 * damping_ratio * torch.sqrt(stiffness * torch.diagonal(task_inertia, dim1=-2, dim2=-1))
    wrench = stiffness * pose_error - damping * ee_vel
    tau = (jt @ wrench.unsqueeze(-1)).squeeze(-1)

    # Dynamically consistent null-space projector N^T = I - J^T Jbar^T, with Jbar = M^-1 J^T Lambda.
    n = joint_pos.shape[-1]
    jbar_t = task_inertia @ jacobian @ m_inv  # (N, 6, n)
    projector = torch.eye(n, device=jacobian.device) - jt @ jbar_t
    nullspace_damping = 2.0 * nullspace_stiffness**0.5
    joint_acc = nullspace_stiffness * (nullspace_target - joint_pos) - nullspace_damping * joint_vel
    tau_null = (projector @ mass_matrix @ joint_acc.unsqueeze(-1)).squeeze(-1)
    return tau + tau_null
