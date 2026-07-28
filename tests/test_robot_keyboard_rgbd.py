import numpy as np

from examples.robot_keyboard_rgbd import build_profile, colorize_depth
from stretch_mujoco.robots import RobotType, create_simulator


def test_every_rgb_stream_has_matching_depth_stream() -> None:
    for robot_type in (RobotType.GOOGLE_ROBOT, RobotType.TIDYBOT):
        simulator = create_simulator(robot_type)
        rgb_frames = {camera.camera_name_in_mjcf for camera in simulator.Cameras.rgb()}
        depth_frames = {camera.camera_name_in_mjcf for camera in simulator.Cameras.depth()}

        assert rgb_frames
        assert rgb_frames == depth_frames


def test_keyboard_profiles_cover_arm_and_gripper() -> None:
    google = create_simulator(RobotType.GOOGLE_ROBOT)
    tidy = create_simulator(RobotType.TIDYBOT)

    google_profile = build_profile(google, RobotType.GOOGLE_ROBOT)
    tidy_profile = build_profile(tidy, RobotType.TIDYBOT)

    assert set("1234567zxcvbnm").issubset(google_profile.arm_keys)
    assert {"8", "9", ",", "."}.issubset(google_profile.arm_keys)
    assert set("1234567zxcvbnm") == set(tidy_profile.arm_keys)
    assert len(google_profile.gripper_close) == 2
    assert len(tidy_profile.gripper_close) == 1


def test_depth_colorization_handles_invalid_pixels() -> None:
    depth = np.array([[np.nan, -1.0, 0.5], [1.0, 2.0, np.inf]], dtype=np.float32)

    colored = colorize_depth(depth, max_depth=2.0)

    assert colored.shape == (2, 3, 3)
    assert colored.dtype == np.uint8
    assert np.array_equal(colored[0, 0], [0, 0, 0])
    assert np.array_equal(colored[0, 1], [0, 0, 0])
