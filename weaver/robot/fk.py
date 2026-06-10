"""Franka Panda forward kinematics (Denavit-Hartenberg convention)."""

import numpy as np
from scipy.spatial.transform import Rotation


def compute_fk_cartesian(joint_positions: np.ndarray) -> np.ndarray:
    """(T, 7) joint angles → (T, 6) [x, y, z, rx, ry, rz] end-effector pose."""
    out = []
    for joints in joint_positions:
        T = _fk(joints)
        out.append(np.concatenate([T[:3, 3], Rotation.from_matrix(T[:3, :3]).as_euler("xyz")]))
    return np.array(out, dtype=np.float32)


def _fk(joints: np.ndarray) -> np.ndarray:
    """Compute 4×4 end-effector transform for 7-DOF Panda joints."""
    dh = [
        [0,       0.333,  0,        joints[0]],
        [0,       0,     -np.pi/2,  joints[1]],
        [0,       0.316,  np.pi/2,  joints[2]],
        [0.0825,  0,      np.pi/2,  joints[3]],
        [-0.0825, 0.384, -np.pi/2,  joints[4]],
        [0,       0,      np.pi/2,  joints[5]],
        [0.088,   0,      np.pi/2,  joints[6]],
        [0,       0.107,  0,        0         ],
        [0,       0,      0,       -np.pi/4   ],
        [0.0,     0.1034, 0,        0         ],
    ]
    T = np.eye(4)
    for i in range(8):
        a, d, alpha, q = dh[i]
        T = T @ np.array([
            [np.cos(q), -np.sin(q), 0, a],
            [np.sin(q)*np.cos(alpha), np.cos(q)*np.cos(alpha), -np.sin(alpha), -np.sin(alpha)*d],
            [np.sin(q)*np.sin(alpha), np.cos(q)*np.sin(alpha),  np.cos(alpha),  np.cos(alpha)*d],
            [0, 0, 0, 1],
        ])
    return T
