"""Offline tests for the QwenS2 (online Qwen2.5-VL + NavDP pixel-goal) module.

No live API, no server and no rendering are required: the selector and the
NavDP client are faked, and the closed loop is exercised with pure geometry.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from stretch_mujoco.navigations import Algorithm, NavigationController
from stretch_mujoco.navigations.NavDP import NavDPClient
from stretch_mujoco.navigations.QwenS2 import QwenPixelGoalSelector, QwenS2Planner


# ---------------------------------------------------------------------------
# QwenPixelGoalSelector — response parsing + pixel scaling
# ---------------------------------------------------------------------------


def test_parse_response_normalized_coords() -> None:
    assert QwenPixelGoalSelector.parse_response("0.5 0.3") == (0.5, 0.3)
    assert QwenPixelGoalSelector.parse_response("0.8, 0.2") == (0.8, 0.2)
    assert QwenPixelGoalSelector.parse_response("I think 0.6 and 0.7") == (0.6, 0.7)


def test_parse_response_stop() -> None:
    assert QwenPixelGoalSelector.parse_response("STOP") is None
    assert QwenPixelGoalSelector.parse_response("stop, reached") is None


def test_parse_response_invalid() -> None:
    with pytest.raises(RuntimeError, match="did not return coordinates"):
        QwenPixelGoalSelector.parse_response("no numbers here")


def test_select_goal_scales_normalized_to_pixel(monkeypatch) -> None:
    selector = QwenPixelGoalSelector(model="x", api_key="k")
    monkeypatch.setattr(selector, "_query_normalized", lambda *a: (0.5, 0.5))
    assert selector.select_goal(b"jpg", "go forward", (424, 240)) == (212, 120)
    monkeypatch.setattr(selector, "_query_normalized", lambda *a: (1.0, 1.0))
    assert selector.select_goal(b"jpg", "go forward", (424, 240)) == (423, 239)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeSelector(QwenPixelGoalSelector):
    """Returns the image centre until *stop_after* calls, then STOP."""

    def __init__(self, stop_after: int = 3) -> None:
        super().__init__(model="x", api_key="k")
        self.stop_after = int(stop_after)
        self.calls = 0

    def select_goal(self, image_bytes, instruction, image_size):
        self.calls += 1
        if self.calls >= self.stop_after:
            return None
        return (image_size[0] // 2, image_size[1] // 2)


class FakeNavDPClient(NavDPClient):
    """In-memory stand-in for the HTTP NavDP client."""

    def __init__(self) -> None:
        super().__init__(base_url="http://127.0.0.1:1")
        self.reset_count = 0

    def reset(self, intrinsic, stop_threshold=-0.5, batch_size=1) -> dict:
        self.reset_count += 1
        return {"algo": "navdp"}

    def step_pixelgoal(self, pixel_xy, rgb_bytes, depth_m) -> np.ndarray:
        xs = np.linspace(0.2, 1.2, 10)
        return np.column_stack([xs, np.zeros(10), np.zeros(10)])


# ---------------------------------------------------------------------------
# QwenS2Planner — closed loop
# ---------------------------------------------------------------------------


def test_qwens2_closed_loop_moves_and_stops_on_vlm_signal() -> None:
    planner = QwenS2Planner(
        FakeNavDPClient(),
        FakeSelector(stop_after=3),
        np.eye(3),
        instruction="go forward",
        image_size=(424, 240),
    )
    robot = np.array([0.0, 0.0], dtype=float)
    yaw = 0.0
    dt = 0.1
    for _ in range(200):
        v, w = planner.step(robot, yaw, b"x", np.full((240, 424), 2.0, np.float32))
        yaw += w * dt
        robot = robot + v * np.array([np.cos(yaw), np.sin(yaw)]) * dt
        if planner.diag.stop_requested:
            break

    assert planner.diag.stop_requested
    assert planner.diag.replan_count >= 2
    assert robot[0] > 0.5  # made forward progress before the VLM stopped it


def test_qwens2_reuses_last_instruction_when_override_none() -> None:
    planner = QwenS2Planner(
        FakeNavDPClient(),
        FakeSelector(stop_after=99),
        np.eye(3),
        instruction="default instruction",
        image_size=(424, 240),
    )
    v, w = planner.step(np.array([0.0, 0.0]), 0.0, b"x", np.full((240, 424), 2.0, np.float32))
    assert planner.diag.last_instruction == "default instruction"
    assert v > 0.0


# ---------------------------------------------------------------------------
# Controller integration
# ---------------------------------------------------------------------------


def _scene_path() -> str:
    import stretch_mujoco

    return str(Path(stretch_mujoco.__file__).resolve().parent / "models" / "office_scene.xml")


def test_controller_registers_qwens2_algorithm() -> None:
    nav = NavigationController.from_scene_xml(
        _scene_path(),
        algorithm=Algorithm.QWENS2,
        planner_kwargs={
            "client": FakeNavDPClient(),
            "selector": FakeSelector(),
            "intrinsic": np.eye(3),
            "instruction": "go",
            "image_size": (424, 240),
        },
    )
    assert nav.algorithm is Algorithm.QWENS2
    assert isinstance(nav.qwens2, QwenS2Planner)
    # Grid-based planning is not available for the closed-loop policy.
    with pytest.raises(RuntimeError, match="closed-loop"):
        nav.plan(start=(0.0, 0.0), goal=(1.0, 1.0))
