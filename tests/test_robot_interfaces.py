import time
from pathlib import Path

import mujoco
import numpy as np

from stretch_mujoco.robots import RobotType, create_simulator
from stretch_mujoco.robots.google_robot import (
    GoogleRobotCameras,
    GoogleRobotSensors,
    StatusGoogleRobotCameraData,
    StatusGoogleRobotJoints,
)
from stretch_mujoco.robots.tidybot import StatusTidyBotJoints, TidyBotCameras


ASSETS = Path(__file__).resolve().parents[1] / "stretch_mujoco" / "models" / "assets"


def test_google_robot_model_includes_simpler_env_head_and_sensors() -> None:
    model = mujoco.MjModel.from_xml_path(str(ASSETS / "google_robot" / "scene.xml"))

    assert model.nu == 14
    assert model.ncam == 1
    assert model.nsensor == 14
    assert GoogleRobotCameras.rgb() == [GoogleRobotCameras.overhead_camera]
    assert GoogleRobotCameras.depth() == [GoogleRobotCameras.overhead_depth]


def test_robot_joint_defaults_are_not_shared() -> None:
    first = StatusGoogleRobotJoints.default()
    second = StatusGoogleRobotJoints.default()

    first.base_x.pos = 1.0

    assert second.base_x.pos == 0.0


def test_google_robot_backend_uses_robot_specific_data_and_planar_pose() -> None:
    half_yaw = np.pi / 4.0
    simulator = create_simulator(
        RobotType.GOOGLE_ROBOT,
        start_translation=[1.0, 2.0, 99.0],
        start_rotation_quat=[np.cos(half_yaw), 0.0, 0.0, np.sin(half_yaw)],
    )
    try:
        simulator.start(headless=True)

        status = simulator.pull_status()
        camera_data = simulator.pull_camera_data()
        x, y, heading = simulator.get_base_pose()

        assert isinstance(status, StatusGoogleRobotJoints)
        assert np.isclose(status.joint_head_pan.pos, -0.00285961, atol=1e-3)
        assert np.isclose(status.joint_head_tilt.pos, 0.9351361, atol=1e-3)
        assert isinstance(camera_data, StatusGoogleRobotCameraData)
        assert camera_data.get_all() == {}
        assert np.allclose((x, y, heading), (1.0, 2.0, np.pi / 2.0), atol=1e-3)
        assert simulator.get_ee_pose().shape == (4, 4)
        assert len(simulator.pull_joint_limits()) == 11
        assert simulator.Sites.HEAD_CAMERA in simulator.Sites.camera_sites()
        assert simulator.available_sensors == GoogleRobotSensors.all()
        sensor_data = simulator.pull_sensor_data()
        assert sensor_data.get_data(GoogleRobotSensors.TIME_OF_FLIGHT).shape == (8,)
        assert sensor_data.get_data(GoogleRobotSensors.CLIFF).shape == (2,)
    finally:
        simulator.stop()


def test_google_robot_move_and_omnidirectional_velocity() -> None:
    simulator = create_simulator(RobotType.GOOGLE_ROBOT)
    try:
        simulator.start(headless=True)
        simulator.move_to(simulator.Actuators.joint_shoulder, 0.1)

        assert simulator.wait_until_at_setpoint(simulator.Actuators.joint_shoulder, timeout=2.0)

        simulator.set_base_velocity(0.0, 0.0, v_lateral=0.2)
        time.sleep(0.1)
        simulator.set_base_velocity(0.0, 0.0)

        x, y, _ = simulator.get_base_pose()
        assert abs(x) < 1e-3
        assert y > 0.005
    finally:
        simulator.stop()


def test_tidybot_general_actuator_and_keyframes() -> None:
    simulator = create_simulator(RobotType.TIDYBOT)
    try:
        simulator.start(headless=True)
        assert isinstance(simulator.pull_status(), StatusTidyBotJoints)

        simulator.move_to(simulator.Actuators.fingers_actuator, 300.0)
        time.sleep(0.02)
        assert simulator.pull_status().fingers_actuator.pos == 255.0

        simulator.home()
        time.sleep(0.02)
        assert np.isclose(simulator.pull_status().joint_2.pos, 0.26179939, atol=2e-3)

        simulator.stow()
        time.sleep(0.02)
        assert np.isclose(simulator.pull_status().joint_2.pos, -0.34906585, atol=2e-3)
    finally:
        simulator.stop()


def test_source_camera_calibrations_are_exposed() -> None:
    google_k = GoogleRobotCameras.overhead_camera.initial_camera_settings
    tidy_settings = TidyBotCameras.base.initial_camera_settings

    assert np.allclose(
        np.asarray(google_k.get_intrinsic_params_k()).reshape(3, 3),
        [[425.0, 0.0, 305.0], [0.0, 413.1, 233.0], [0.0, 0.0, 1.0]],
    )
    assert np.allclose(
        np.asarray(tidy_settings.get_intrinsic_params_k()).reshape(3, 3),
        [
            [377.89521770116426, 0.0, 319.9291029848687],
            [0.0, 378.16364930400925, 180.4933473081451],
            [0.0, 0.0, 1.0],
        ],
    )
    assert tidy_settings.distortion_params is not None
    assert TidyBotCameras.base.is_simulated is False
    assert TidyBotCameras.wrist.is_simulated is True
