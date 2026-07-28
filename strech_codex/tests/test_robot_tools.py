"""Tests for robot control and scene MCP tools."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

# Ensure strech_codex is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from strech_codex.tools.robot import (
    robot_get_status,
    robot_init,
    robot_list_actuators,
    robot_list_cameras,
    robot_stop,
)
from strech_codex.tools.navigation import nav_astar, nav_build_grid, nav_execute_path
from strech_codex.tools.scene import scene_get_object_pose, scene_list_objects
from strech_codex.world.state import has_sim, reset_world
from strech_codex.world.state import (
    get_robot_type,
    get_scene_path,
    get_sim,
    set_scene_path,
    set_sim,
)


def _office_scene_xml() -> str:
    return str(
        Path(__file__).resolve().parents[2] / "stretch_mujoco" / "models" / "office_scene.xml"
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_world():
    """Reset world state before each test and clean up after."""
    reset_world()
    yield
    try:
        robot_stop()
    except Exception:
        pass
    reset_world()


# ---------------------------------------------------------------------------
# Robot lifecycle
# ---------------------------------------------------------------------------


def test_robot_init_stretch3():
    """robot_init should start a Stretch 3 simulator headless."""
    result = robot_init("stretch3", headless=True)
    assert result["success"], result.get("message")
    assert result["robot_type"] == "stretch3"
    assert has_sim()


def test_robot_init_invalid_type():
    """robot_init should fail for an unknown robot type."""
    result = robot_init("invalid_robot", headless=True)
    assert not result["success"]


def test_robot_init_passes_scene_and_cameras(monkeypatch):
    """Configured scene and camera names should reach the simulator factory."""
    captured = {}

    class Actuators:
        @staticmethod
        def all():
            return []

    class FakeSim:
        def __init__(self):
            self.Actuators = Actuators

        def start(self, **kwargs):
            captured["start"] = kwargs

        def stop(self):
            pass

    def fake_create(robot_type, **kwargs):
        captured.update(kwargs)
        return FakeSim()

    monkeypatch.setattr("strech_codex.tools.robot.create_simulator", fake_create)
    result = robot_init(
        "google_robot",
        scene_xml="office_02_cross_axis",
        cameras_to_use=["all_rgb", "overhead_depth"],
    )

    assert result["success"]
    assert captured["scene_xml_path"].endswith("/office_scenes/office_02_cross_axis.xml")
    assert [camera.name for camera in captured["cameras_to_use"]] == [
        "overhead_camera",
        "overhead_depth",
    ]


def test_robot_init_failure_does_not_publish_state(monkeypatch):
    class FailingSim:
        def start(self, **kwargs):
            raise RuntimeError("boom")

        def stop(self):
            pass

    monkeypatch.setattr(
        "strech_codex.tools.robot.create_simulator", lambda *args, **kwargs: FailingSim()
    )
    result = robot_init("google_robot")
    assert not result["success"]
    assert not has_sim()
    assert get_robot_type() is None


def test_robot_stop():
    """robot_stop should shut down the simulator."""
    robot_init("stretch3", headless=True)
    assert has_sim()
    result = robot_stop()
    assert result["success"]
    assert not has_sim()


# ---------------------------------------------------------------------------
# Introspection
# ---------------------------------------------------------------------------


def test_list_actuators():
    """robot_list_actuators should return Stretch 3 actuators."""
    robot_init("stretch3", headless=True)
    result = robot_list_actuators()
    assert result["success"]
    assert len(result["actuators"]) > 0
    # Stretch 3 has well-known actuators like lift, arm, etc.
    names = [a["name"] for a in result["actuators"]]
    assert "LIFT" in names or "lift" in names or any("lift" in n.lower() for n in names)


def test_list_cameras():
    """robot_list_cameras should return camera info."""
    robot_init("stretch3", headless=True)
    result = robot_list_cameras()
    assert result["success"]
    # Stretch 3 has multiple cameras
    assert result["cameras"]


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------


def test_get_status():
    """robot_get_status should return a valid status snapshot."""
    robot_init("stretch3", headless=True)
    result = robot_get_status()
    assert result["success"]
    assert "status" in result


# ---------------------------------------------------------------------------
# Scene tools (without active robot)
# ---------------------------------------------------------------------------


def test_scene_list_objects_no_sim():
    """scene_list_objects should fail gracefully without a simulator."""
    result = scene_list_objects()
    assert not result["success"]


def test_scene_tools_fall_back_to_scene_xml():
    class FakeStretchSim:
        _scene_xml_path = _office_scene_xml()

        def get_link_pose(self, name):
            raise KeyError(name)

    set_sim(FakeStretchSim())
    objects = scene_list_objects()
    assert objects["success"]
    assert objects["objects"]
    pose = scene_get_object_pose(objects["objects"][0])
    assert pose["success"]
    assert len(pose["pose"]) == 4


# ---------------------------------------------------------------------------
# Google Robot (if available; these models take longer to load, so mark them)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_robot_init_google_robot():
    """robot_init should start a Google Robot simulator headless."""
    result = robot_init("google_robot", headless=True)
    # This may fail if the scene XML is not configured correctly
    if result["success"]:
        assert result["robot_type"] == "google_robot"
        assert has_sim()
        robot_stop()
    else:
        pytest.skip(f"Google Robot init failed (may need scene_xml): {result.get('message')}")


@pytest.mark.slow
def test_robot_init_tidybot():
    """robot_init should start a TidyBot simulator headless."""
    result = robot_init("tidybot", headless=True)
    if result["success"]:
        assert result["robot_type"] == "tidybot"
        assert has_sim()
        robot_stop()
    else:
        pytest.skip(f"TidyBot init failed (may need scene_xml): {result.get('message')}")


@pytest.mark.slow
@pytest.mark.parametrize("robot_type", ["stretch3", "google_robot", "tidybot"])
def test_execute_short_path_on_real_robot(robot_type):
    started = robot_init(robot_type, headless=True)
    assert started["success"], started.get("message")
    assert Path(started["scene_xml"]).is_file()
    assert get_scene_path() == started["scene_xml"]
    grid = nav_build_grid(resolution=0.2, agent_radius=0.1)
    assert grid["success"], grid.get("message")
    x, y, yaw = get_sim().get_base_pose()
    goal = [x + 0.15 * math.cos(yaw), y + 0.15 * math.sin(yaw)]
    planned = nav_astar(x, y, goal[0], goal[1])
    assert planned["success"], planned.get("message")
    result = nav_execute_path(
        planned["waypoints"],
        timeout=20.0,
        position_tolerance=0.025,
        max_linear_speed=0.2,
    )
    assert result["success"], {"planned": planned, "result": result}
    assert result["goal_error_m"] <= 0.025
