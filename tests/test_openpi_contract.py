import numpy as np
import pytest

from stretch_mujoco.datamodels.status_stretch_joints import StatusStretchJoints
from stretch_mujoco.openpi_contract import (
    ACTION_NAMES,
    HEAD_POSE,
    IMAGE_SIZE,
    STATE_NAMES,
    action_from_keys,
    base_action_deltas,
    grasp_diagnostics,
    make_observation,
    save_episode,
    state_from_status,
    to_lerobot_frame,
    validate_action,
)


def test_openpi_contract_builds_training_frame(tmp_path) -> None:
    status = StatusStretchJoints.default()
    status.base.x_vel = 0.1
    status.base.theta_vel = -0.2
    status.lift.pos = 0.7
    status.gripper.pos = 0.56

    state = state_from_status(status)
    head = np.zeros((240, 424, 3), dtype=np.uint8)
    head[..., 0] = 255
    wrist = np.zeros((270, 270, 3), dtype=np.uint8)
    wrist[..., 1] = 255
    observation = make_observation(
        state=state, head_rgb=head, wrist_rgb=wrist, prompt="  pick up the red cup  "
    )
    action = np.arange(len(ACTION_NAMES), dtype=np.float32)
    frame = to_lerobot_frame(observation, action)

    assert state.shape == (len(STATE_NAMES),)
    assert state.dtype == np.float32
    assert np.allclose(state[[0, 1, 2, 7]], [0.1, -0.2, 0.7, 0.56])
    assert frame["image"].shape == (*IMAGE_SIZE, 3)
    assert frame["wrist_image"].shape == (*IMAGE_SIZE, 3)
    assert frame["image"].dtype == np.uint8
    assert frame["image"][112, 112].tolist() == [255, 0, 0]
    assert frame["wrist_image"][112, 112].tolist() == [0, 255, 0]
    assert frame["task"] == "pick up the red cup"
    np.testing.assert_array_equal(frame["actions"], action)

    path = save_episode(tmp_path / "episode.npz", [frame], [0.0])
    with np.load(path, allow_pickle=False) as episode:
        assert episode["image"].shape == (1, *IMAGE_SIZE, 3)
        assert episode["state"].shape == (1, len(STATE_NAMES))
        assert episode["actions"].shape == (1, len(ACTION_NAMES))
        assert episode["task"].item() == "pick up the red cup"


def test_openpi_contract_rejects_bad_shapes() -> None:
    with pytest.raises(ValueError, match="state must be finite"):
        make_observation(
            state=np.zeros(7),
            head_rgb=np.zeros((2, 2, 3)),
            wrist_rgb=np.zeros((2, 2, 3)),
            prompt="pick",
        )
    with pytest.raises(ValueError, match="action must be finite"):
        validate_action(np.zeros(7))

    observation = make_observation(
        state=np.zeros(8),
        head_rgb=np.zeros((2, 2, 3)),
        wrist_rgb=np.zeros((2, 2, 3)),
        prompt="pick",
    )
    with pytest.raises(ValueError, match="no commanded motion"):
        save_episode(
            "unused.npz",
            [to_lerobot_frame(observation, np.zeros(8))],
            [0.0],
        )


def test_openpi_contract_maps_teleop_keys() -> None:
    action = action_from_keys({"w", "a", "i", "n"}, gripper_position=0.52)

    assert np.allclose(action[:3], [0.07, 0.15, 0.1])
    assert action[-1] == pytest.approx(0.56)
    assert action_from_keys({"w", "s"}, 0.0)[0] == pytest.approx(0.0)
    assert HEAD_POSE == {"head_pan": -1.57, "head_tilt": -0.65}


def test_base_action_deltas_uses_wrap_aware_angle() -> None:
    deltas = base_action_deltas((0.0, -np.pi + 0.01), (0.12, np.pi - 0.01))
    # Shortest signed angular difference across the -pi/pi seam is ~-0.02, not ~2pi.
    assert np.allclose(deltas, [0.12, -0.02], atol=1e-3)

    assert np.allclose(base_action_deltas((0.5, 0.3), (0.52, 0.4)), [0.02, 0.1], atol=1e-3)


def test_episode_saves_grasp_diagnostics_without_changing_training_frame(tmp_path) -> None:
    observation = make_observation(
        state=np.zeros(8),
        head_rgb=np.zeros((2, 2, 3)),
        wrist_rgb=np.zeros((2, 2, 3)),
        prompt="pick",
    )
    frame = to_lerobot_frame(observation, np.ones(8, dtype=np.float32))
    diagnostic = grasp_diagnostics(
        {
            "object_position": [1.0, 2.0, 0.7],
            "grasp_center_position": [1.0, 2.0, 0.6],
            "center_distance_m": 0.1,
            "left_finger_contacts": 1,
            "right_finger_contacts": 0,
            "bilateral_contact": False,
        },
        initial_object_position=np.array([1.0, 2.0, 0.5]),
    )

    path = save_episode(
        tmp_path / "episode.npz",
        [frame],
        [0.0],
        diagnostics=[diagnostic],
        metadata={"success": False},
    )

    with np.load(path, allow_pickle=False) as episode:
        assert episode["diagnostics/object_position"].shape == (1, 3)
        assert episode["diagnostics/object_lift_m"].item() == pytest.approx(0.2)
        assert not bool(episode["diagnostics/bilateral_contact"].item())
        assert '"success": false' in episode["episode_metadata_json"].item()
        assert set(frame) == {"image", "wrist_image", "state", "actions", "task"}
