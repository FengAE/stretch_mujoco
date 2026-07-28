"""RGBD action-boundary evidence regression test."""

import json
from types import SimpleNamespace

import numpy as np

from strech_codex.evidence import capture
from strech_codex.world.state import set_sim


def test_capture_saves_all_rgbd(monkeypatch, tmp_path):
    rgb = SimpleNamespace(name="head_rgb", is_depth=False)
    depth = SimpleNamespace(name="head_depth", is_depth=True)

    class Snapshot:
        def get_camera_data(self, camera, auto_correct_rgb=True):
            if camera.is_depth:
                return np.ones((8, 10), dtype=np.float32)
            return np.zeros((8, 10, 3), dtype=np.uint8)

    sim = SimpleNamespace(
        Cameras=SimpleNamespace(all=lambda: [rgb, depth]),
        pull_camera_data=lambda: Snapshot(),
        pull_status=lambda: SimpleNamespace(to_dict=lambda: {"time": 1.0}),
        get_base_pose=lambda: (1.0, 2.0, 0.5),
        get_ee_pose=lambda: np.eye(4),
    )
    monkeypatch.setenv("STRECH_CODEX_EPISODE_DIR", str(tmp_path))
    set_sim(sim)
    try:
        artifacts = capture("nav_execute_path", "start", 1)
    finally:
        set_sim(None)

    directory = tmp_path / "evidence" / "001_nav_execute_path" / "start"
    assert len(artifacts) == 2
    assert (directory / "head_rgb.png").is_file()
    assert (directory / "head_depth.npy").is_file()
    assert (directory / "head_depth.png").is_file()
    metadata = json.loads((directory / "metadata.json").read_text())
    assert metadata["tool"] == "nav_execute_path"
    assert metadata["phase"] == "start"
    assert metadata["state"]["base_pose"] == [1.0, 2.0, 0.5]
    assert len(metadata["cameras"]) == 2
