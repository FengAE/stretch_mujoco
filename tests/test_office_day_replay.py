from pathlib import Path

import numpy as np

from examples.office_day_replay import (
    active_item,
    annotate_frame,
    apply_item,
    load_replay,
    visual_action,
)
from stretch_mujoco.datamodels.status_stretch_camera import StatusStretchCameras
from stretch_mujoco.enums.stretch_cameras import StretchCameras


REPLAY_PATH = (
    Path(__file__).resolve().parents[1] / "stretch_mujoco" / "models" / "office_day_replay.json"
)


class FakeSimulator:
    def __init__(self) -> None:
        self.calls = []

    def set_humanoid_animation(self, animation):
        self.calls.append(("animation", animation))

    def set_humanoid_sit_target(self, target):
        self.calls.append(("sit_target", target))

    def set_humanoid_navigation_target(self, target):
        self.calls.append(("navigation_target", target))


def test_replay_schedule_covers_gaps_without_teleporting_home() -> None:
    start, end, items = load_replay(REPLAY_PATH)

    assert start == 9 * 60
    assert end == 18 * 60
    assert active_item(items, 10 * 60 + 27).item_id == "morning_work"
    assert active_item(items, 11 * 60 + 5).item_id == "team_meeting"


def test_replay_maps_work_to_seated_pose_and_meeting_to_navigation() -> None:
    _, _, items = load_replay(REPLAY_PATH)
    simulator = FakeSimulator()

    apply_item(simulator, items[0])
    assert simulator.calls == [
        ("sit_target", "chair_right"),
        ("animation", "work"),
    ]

    simulator.calls.clear()
    meeting = next(item for item in items if item.activity == "meeting")
    apply_item(simulator, meeting)
    assert simulator.calls == [
        ("navigation_target", "meeting_table"),
        ("animation", "walk"),
    ]


def test_lunch_visual_sequence_walks_then_eats_then_finishes() -> None:
    _, _, items = load_replay(REPLAY_PATH)
    lunch = next(item for item in items if item.location == "snack_counter")
    duration = lunch.end_minute - lunch.start_minute

    assert visual_action(lunch, lunch.start_minute + duration * 0.10) == "walk"
    assert visual_action(lunch, lunch.start_minute + duration * 0.50) == "eat"
    assert visual_action(lunch, lunch.start_minute + duration * 0.90) == "idle"


def test_office_overview_frame_is_available_and_annotated() -> None:
    rgb = np.zeros((540, 960, 3), dtype=np.uint8)
    cameras = StatusStretchCameras.default()
    cameras.set_camera_data(StretchCameras.office_overview_rgb, rgb)
    frame = cameras.get_camera_data(StretchCameras.office_overview_rgb)
    _, _, items = load_replay(REPLAY_PATH)

    annotated = annotate_frame(frame, 9 * 60, items[0])

    assert annotated.shape == (540, 960, 3)
    assert np.count_nonzero(annotated[:110, :570]) > 0
    assert StretchCameras.office_overview_rgb in StretchCameras.rgb()
