"""Tests for MP4 episode recording."""

import json
import time
from types import SimpleNamespace

import cv2
import numpy as np

from strech_codex.episode_logger import EpisodeLogger
from strech_codex.video_recorder import RobotVideoRecorder


def test_video_recorder_writes_readable_mp4(tmp_path):
    frame = np.full((48, 64, 3), 127, dtype=np.uint8)

    class Snapshot:
        def get_camera_data(self, camera, auto_correct_rgb=True):
            return frame

    sim = SimpleNamespace(pull_camera_data=lambda: Snapshot())
    camera = SimpleNamespace(name="test_camera")
    episode = EpisodeLogger("video test", tmp_path)
    path = episode.video_path
    assert episode.path.parent.parent == tmp_path
    assert episode.path.parent.name == episode.path.stem
    assert path.parent == episode.path.parent
    assert path.stem == episode.path.stem
    recorder = RobotVideoRecorder(sim, camera, path, fps=20.0)
    recorder.start()
    deadline = time.monotonic() + 2.0
    while recorder.frames < 3 and time.monotonic() < deadline:
        time.sleep(0.02)
    result = recorder.stop()

    assert result["error"] is None
    assert result["frames"] >= 3
    assert path.stat().st_size > 0
    video = cv2.VideoCapture(str(path))
    readable, decoded = video.read()
    video.release()
    assert readable
    assert decoded.shape[:2] == frame.shape[:2]
    episode.finish("done")
    log = json.loads(episode.path.read_text(encoding="utf-8"))
    assert log["artifacts"][0]["path"] == str(path)
