from types import SimpleNamespace

import numpy as np

from examples.openpi_auto_grasp import EpisodeCapture, _blue_grasp_model
from stretch_mujoco.enums.actuators import Actuators


class FakeSimulator:
    def __init__(self):
        self.arm_position = 0.0
        self.commands = []

    def move_to(self, actuator, target):
        assert actuator == Actuators.arm
        self.arm_position = target
        self.commands.append(target)

    def move_by(self, actuator, delta):
        self.commands.append((actuator, delta))

    def pull_status(self):
        return SimpleNamespace(arm=SimpleNamespace(pos=self.arm_position))


def test_small_base_move_is_skipped_and_arm_move_is_rate_limited():
    sim = FakeSimulator()
    recorder = EpisodeCapture(sim, "pick")
    recorder.capture = lambda: None

    recorder.move_base_by(0.001)
    assert sim.commands == []
    assert not recorder.base_was_commanded

    recorder.move({Actuators.arm: 0.1}, max_step=0.02)
    np.testing.assert_allclose(sim.commands, [0.02, 0.04, 0.06, 0.08, 0.1, 0.1])

    model = _blue_grasp_model()
    body = model.body("object2")
    qpos = int(model.jnt_qposadr[int(body.jntadr[0])])
    np.testing.assert_allclose(model.qpos0[qpos : qpos + 3], [0.30, -0.80, 0.60])
