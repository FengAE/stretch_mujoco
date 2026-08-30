import numpy as np
from unittest.mock import Mock

from examples.evaluate_openpi_policy import (
    ACTION_LIMITS,
    _applied_action,
    _execute_action,
    _safe_action,
)
from stretch_mujoco.enums.actuators import Actuators
from stretch_mujoco.robots.stretch3.config import GRIPPER_MIN_MAX


def test_safe_action_clips_policy_outliers():
    action = _safe_action(np.array([9, -9, 9, -9, 9, -9, 9, 9], dtype=np.float32))
    np.testing.assert_allclose(np.abs(action[:7]), ACTION_LIMITS)
    assert np.isclose(action[7], GRIPPER_MIN_MAX[1])


def test_execute_action_ignores_rotation_drift():
    sim = Mock()
    targets = np.zeros(5)

    _execute_action(sim, np.array([0, 4e-4, 0, 0, 0, 0, 0, 0], dtype=np.float32), targets)

    sim.set_base_velocity.assert_called_once_with(0.0, 0.0)
    sim.reset_mock()

    _execute_action(sim, np.array([0, 6e-4, 0, 0, 0, 0, 0, 0], dtype=np.float32), targets)

    np.testing.assert_allclose(sim.set_base_velocity.call_args.args, (0.0, 0.006))


def test_execute_action_ignores_translation_drift():
    sim = Mock()
    targets = np.zeros(5)

    _execute_action(sim, np.array([2e-3, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32), targets)

    sim.set_base_velocity.assert_called_once_with(0.0, 0.0)
    sim.reset_mock()

    _execute_action(sim, np.array([4e-3, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32), targets)

    np.testing.assert_allclose(sim.set_base_velocity.call_args.args, (0.04, 0.0))


def test_execute_action_accumulates_joint_targets():
    sim = Mock()
    targets = np.array([0.5, 0.1, 0.0, -0.2, 0.0])
    action = np.array([0, 0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.2], dtype=np.float32)

    _execute_action(sim, action, targets)
    _execute_action(sim, action, targets)

    np.testing.assert_allclose(targets, [0.52, 0.14, 0.06, -0.12, 0.1])
    arm_targets = [
        call.args[1] for call in sim.move_to.call_args_list if call.args[0] == Actuators.arm
    ]
    np.testing.assert_allclose(arm_targets, [0.12, 0.14])


def test_applied_action_distinguishes_clipping_from_deadbands():
    raw = np.array([9, 4e-4, 1e-6, 0.02, 0, 0, 0, 9], dtype=np.float32)

    clipped = _safe_action(raw)
    applied = _applied_action(raw)

    assert clipped[0] == ACTION_LIMITS[0]
    assert clipped[1] == np.float32(4e-4)
    assert clipped[2] == np.float32(1e-6)
    np.testing.assert_allclose(applied[:4], [ACTION_LIMITS[0], 0, 0, 0.02])
    assert np.isclose(applied[7], GRIPPER_MIN_MAX[1])
