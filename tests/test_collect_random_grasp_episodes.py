import numpy as np
import random

from examples.collect_random_grasp_episodes import (
    SEEDS,
    _close_gripper_until_contact,
    _move_base_slow,
    _move_target,
)
from stretch_mujoco.enums.actuators import Actuators
from stretch_mujoco.grasp_task import (
    GRASP_FRICTION,
    VALIDATION_EXCLUSION_RADIUS,
    apply_scenario,
    random_training_scenario,
    validation_scenarios,
)
import mujoco
from examples.collect_random_grasp_episodes import SCENE


class FakeRecorder:
    def __init__(self):
        self.base_moves = []
        self.moves = []
        self.move_kwargs = []

    def move_base_by(self, dx):
        self.base_moves.append(dx)

    def move(self, targets, **kwargs):
        self.moves.append(targets)
        self.move_kwargs.append(kwargs)

    def hold(self, seconds):
        pass


class FakeBaseRecorder:
    def __init__(self):
        self.base_was_commanded = False
        self.sim = self
        self.x = 0.0
        self.velocity = 0.0
        self.commands = []

    def set_base_velocity(self, velocity, omega):
        self.velocity = velocity
        self.commands.append((velocity, omega))

    def get_base_pose(self):
        return self.x, 0.0, 0.0

    def capture(self):
        self.x += self.velocity / 10.0

    def pull_status(self):
        return type("Status", (), {"base": type("Base", (), {"x_vel": self.velocity})()})()


def test_base_move_is_slow_and_stops():
    recorder = FakeBaseRecorder()

    _move_base_slow(recorder, 0.04)

    assert recorder.base_was_commanded
    assert recorder.commands == [(0.08, 0.0), (0.0, 0.0)]
    assert recorder.x <= 0.04


def test_red_uses_six_dof_ik_order():
    assert all(seed.shape == (6,) for seed in SEEDS.values())
    joints = np.array([0.1, 0.4, 0.3, 0.2, -0.5, 0.6])
    recorder = FakeRecorder()

    _move_target(recorder, joints, "red")

    wrist_move = recorder.moves[3]
    assert wrist_move == {
        Actuators.wrist_yaw: 0.2,
        Actuators.wrist_pitch: -0.5,
        Actuators.wrist_roll: 0.6,
    }
    assert recorder.move_kwargs[3]["max_step"] == 0.04
    assert recorder.moves[4] == {Actuators.arm: 0.21999999999999997, Actuators.lift: 0.5}
    assert recorder.move_kwargs[4]["max_step"] == 0.04
    assert recorder.moves[5] == {Actuators.arm: 0.3, Actuators.lift: 0.4}
    assert recorder.move_kwargs[5]["max_step"] == 0.02


def test_close_gripper_stops_after_three_contact_frames():
    class Recorder:
        def __init__(self):
            self.sim = self
            self.captures = 0
            self.commands = []

        def pull_status(self):
            return type("Status", (), {"gripper": type("Gripper", (), {"pos": 0.56})()})()

        def move_to(self, actuator, position):
            self.commands.append((actuator, position))

        def capture(self):
            self.captures += 1

        def pull_grasp_metrics(self):
            return {"bilateral_contact": self.captures >= 2}

    recorder = Recorder()

    _close_gripper_until_contact(recorder)

    assert recorder.commands == [
        (Actuators.gripper, 0.52),
        (Actuators.gripper, 0.48000000000000004),
        (Actuators.gripper, -0.15),
    ]
    assert recorder.captures == 7


def test_training_scenarios_exclude_validation_positions_and_share_friction():
    held_out = validation_scenarios("blue")
    assert len(held_out) == 20
    rng = random.Random(9)
    samples = [random_training_scenario("blue", rng, expanded=True) for _ in range(50)]
    assert all(
        np.hypot(sample.target_x - validation.target_x, sample.target_y - validation.target_y)
        >= VALIDATION_EXCLUSION_RADIUS
        for sample in samples
        for validation in held_out
    )

    model = mujoco.MjModel.from_xml_path(str(SCENE))
    apply_scenario(model, "blue", held_out[0])
    for body_name in ("rubber_tip_left", "rubber_tip_right", "object1", "object2"):
        body = model.body(body_name)
        first = int(body.geomadr[0])
        assert all(
            model.geom_friction[index, 0] == GRASP_FRICTION
            for index in range(first, first + int(body.geomnum[0]))
        )
