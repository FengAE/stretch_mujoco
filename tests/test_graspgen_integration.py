import msgpack
import numpy as np

from stretch_mujoco.graspgen.calibration import (
    ROTATED_TO_RAW_OPTICAL,
    d435i_rotated_camera_intrinsics,
    rotated_d435i_optical_pose,
    world_to_base_pose,
)
from stretch_mujoco.graspgen.ik import StretchGraspIK
from stretch_mujoco.graspgen.protocol import PROTOCOL, encode_rgbd_request
from stretch_mujoco.utils import URDFmodel


def test_rgbd_protocol_encodes_rotated_image_and_intrinsics() -> None:
    rgb = np.zeros((424, 240, 3), dtype=np.uint8)
    depth = np.ones((424, 240), dtype=np.float32)
    camera_k = d435i_rotated_camera_intrinsics()

    frames = encode_rgbd_request(rgb, depth, camera_k, "bread")
    header = msgpack.unpackb(frames[0], raw=False)

    assert len(frames) == 3
    assert header["protocol"] == PROTOCOL
    assert header["rgb"]["shape"] == [424, 240, 3]
    assert header["depth"]["shape"] == [424, 240]
    np.testing.assert_allclose(header["camera_K"], camera_k)


def test_rotated_d435i_calibration_updates_intrinsics_and_optical_axes() -> None:
    camera_k = d435i_rotated_camera_intrinsics()
    raw_pose = np.eye(4)
    raw_pose[:3, 3] = (1.0, 2.0, 3.0)

    rotated_pose = rotated_d435i_optical_pose(raw_pose)

    assert camera_k[0, 2] == 120.0
    assert camera_k[1, 2] == 212.0
    np.testing.assert_allclose(rotated_pose[:3, :3], ROTATED_TO_RAW_OPTICAL)
    np.testing.assert_allclose(rotated_pose[:3, 3], raw_pose[:3, 3])


def test_world_to_base_pose_inverts_planar_base_transform() -> None:
    base_from_world = world_to_base_pose((1.0, 2.0, np.pi / 2.0))
    point_world = np.array([1.0, 3.0, 0.5, 1.0])

    np.testing.assert_allclose(base_from_world @ point_world, [1.0, 0.0, 0.5, 1.0], atol=1e-7)


def test_stretch_grasp_ik_recovers_a_reachable_pose() -> None:
    solver = StretchGraspIK()
    expected = np.array([0.65, 0.38, 0.25, -0.6, 0.4])
    target = solver.forward(expected)

    joints, position_error, orientation_error = solver.solve(target)

    assert position_error < 1e-3
    assert orientation_error < 0.02
    np.testing.assert_allclose(solver.forward(joints), target, atol=1e-2)


def test_stretch_grasp_ik_ranks_reachable_candidates() -> None:
    solver = StretchGraspIK()
    poses = np.asarray(
        [
            solver.forward(np.array([0.65, 0.38, 0.25, -0.6, 0.4])),
            solver.forward(np.array([0.70, 0.42, 0.10, -0.4, -0.2])),
        ]
    )

    ranked = solver.rank_candidates(poses, np.array([0.8, 0.9]))

    assert {candidate.index for candidate in ranked} == {0, 1}
    assert all(candidate.position_error < 0.015 for candidate in ranked)


def test_vertical_carry_pitch_points_gripper_downward() -> None:
    urdf = URDFmodel()
    configuration = {
        "lift": 0.75,
        "arm": 0.06,
        "wrist_yaw": 0.0,
        "wrist_pitch": -1.50,
        "wrist_roll": 0.0,
        "head_pan": 0.0,
        "head_tilt": 0.0,
    }
    wrist = urdf.get_transform(configuration, "link_wrist_roll")[:3, 3]
    grasp = urdf.get_transform(configuration, "link_grasp_center")[:3, 3]
    gripper_axis = grasp - wrist

    assert abs(gripper_axis[0]) < 0.01
    assert abs(gripper_axis[1]) < 0.03
    assert gripper_axis[2] < -0.24
