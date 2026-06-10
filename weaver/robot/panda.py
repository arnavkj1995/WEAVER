"""Franka Panda robot control helpers for WEAVER deploy scripts.

Extracted from openpi/examples/droid/panda_log.py to remove the hard-coded
sys.path dependency on that repo.
"""

from __future__ import annotations

import contextlib
import signal

import numpy as np


@contextlib.contextmanager
def prevent_keyboard_interrupt():
    """Defer SIGINT until after the protected block completes.

    Prevents Ctrl+C from killing the policy-server connection mid-inference.
    """
    interrupted = False
    original_handler = signal.getsignal(signal.SIGINT)

    def handler(signum, frame):
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGINT, handler)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, original_handler)
        if interrupted:
            raise KeyboardInterrupt


def _joint_velocity_to_delta(joint_velocity: np.ndarray) -> np.ndarray:
    relative_max_joint_delta = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2])
    max_joint_delta = relative_max_joint_delta.max()
    if isinstance(joint_velocity, list):
        joint_velocity = np.array(joint_velocity)
    relative_max_joint_vel = relative_max_joint_delta / max_joint_delta
    max_joint_vel_norm = (np.abs(joint_velocity) / relative_max_joint_vel).max()
    if max_joint_vel_norm > 1:
        joint_velocity = joint_velocity / max_joint_vel_norm
    return joint_velocity * max_joint_delta


def step(action: np.ndarray, robot_interface, controller_cfg,
         controller_type: str = "JOINT_IMPEDANCE",
         step_count: int = 0,
         last_joint_position: np.ndarray | None = None) -> None:
    """Send one action to the Franka robot via deoxys.

    Converts 7-D joint-velocity + 1-D gripper action to a joint-position
    command and forwards it to robot_interface.control().

    Only JOINT_POSITION and JOINT_IMPEDANCE controller types are supported.
    """
    if controller_type not in ("JOINT_POSITION", "JOINT_IMPEDANCE"):
        raise NotImplementedError(f"Unsupported controller type: {controller_type}")

    joint_velocity = action[:7]
    delta = _joint_velocity_to_delta(joint_velocity)
    joint_position = robot_interface.last_q if last_joint_position is None else last_joint_position
    robot_action = (joint_position + delta).tolist() + [action[-1]]
    assert len(robot_action) == 8

    robot_interface.control(
        controller_type=controller_type,
        action=robot_action,
        controller_cfg=controller_cfg,
    )
