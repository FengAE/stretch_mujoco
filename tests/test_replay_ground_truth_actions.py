from unittest.mock import Mock

import numpy as np

from examples.replay_ground_truth_actions import _execute_target_action, _execute_velocity_action
from stretch_mujoco.enums.actuators import Actuators


def test_velocity_executor_converts_frame_deltas_and_filters_drift():
    sim = Mock()
    action = np.array([0.004, 0.0004, 0.01, 0, 0, 0, 0, 0.2], dtype=np.float32)

    _execute_velocity_action(sim, action)

    sim.set_base_velocity.assert_called_once_with(0.04000000189989805, 0.0)
    sim.move_by.assert_called_once_with(Actuators.lift, 0.009999999776482582)
    sim.move_to.assert_called_once_with(Actuators.gripper, 0.20000000298023224)


def test_target_executor_accumulates_joint_deltas():
    sim = Mock()
    targets = np.array([0.50, 0.10, 0.0, -0.20, 0.0], dtype=np.float64)
    first = np.array([0.004, 0.002, 0.01, 0.02, 0.03, 0.04, 0.05, 0.2])
    second = np.array([0.0, 0.0, 0.02, 0.03, 0.04, 0.05, 0.06, 0.3])

    _execute_target_action(sim, first, targets)
    _execute_target_action(sim, second, targets)

    np.testing.assert_allclose(targets, [0.53, 0.15, 0.07, -0.11, 0.11])
    np.testing.assert_allclose(sim.set_base_velocity.call_args_list[0].args, (0.04, 0.02))
    arm_calls = [call.args for call in sim.move_to.call_args_list if call.args[0] == Actuators.arm]
    assert [call[0] for call in arm_calls] == [Actuators.arm, Actuators.arm]
    np.testing.assert_allclose([call[1] for call in arm_calls], (0.12, 0.15))
