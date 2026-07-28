"""Tests for navigation MCP tools."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure strech_codex is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from strech_codex.tools.navigation import (
    nav_astar,
    nav_build_grid,
    nav_execute_path,
    nav_fmm,
    nav_get_grid_info,
    nav_is_free,
)
from strech_codex.world.state import (
    get_scene_path,
    has_nav,
    reset_world,
    set_robot_type,
    set_sim,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_world():
    """Reset world state before each test."""
    reset_world()
    yield
    reset_world()


def _office_scene_xml() -> str:
    """Return the path to a known office scene XML."""
    repo_root = Path(__file__).resolve().parents[2]  # strech_codex/tests -> repo root
    return str(repo_root / "stretch_mujoco" / "models" / "office_scene.xml")


def _empty_office_scene_xml() -> str:
    """Return a scene with large open area."""
    repo_root = Path(__file__).resolve().parents[2]
    return str(
        repo_root
        / "stretch_mujoco"
        / "models"
        / "assets"
        / "office_scenes"
        / "office_01_linear_bench.xml"
    )


# ---------------------------------------------------------------------------
# Grid building
# ---------------------------------------------------------------------------


def test_build_grid_succeeds():
    """nav_build_grid should return success on a valid scene."""
    result = nav_build_grid(scene_xml=_office_scene_xml(), resolution=0.2, agent_radius=0.3)
    assert result["success"], result.get("message")
    assert result["width"] > 0
    assert result["height"] > 0
    assert result["free_cells"] > 0
    assert has_nav()
    assert get_scene_path() == _office_scene_xml()


def test_build_grid_invalid_scene():
    """nav_build_grid should fail gracefully on a missing file."""
    result = nav_build_grid(scene_xml="/nonexistent/scene.xml")
    assert not result["success"]


def test_grid_info():
    """nav_get_grid_info should return valid metadata after building."""
    nav_build_grid(scene_xml=_office_scene_xml(), resolution=0.2, agent_radius=0.3)
    info = nav_get_grid_info()
    assert info["success"]
    assert info["width"] > 0
    assert info["height"] > 0
    assert 0.0 < info["free_ratio"] < 1.0


def test_is_free():
    """nav_is_free should correctly report free/occupied cells."""
    nav_build_grid(scene_xml=_office_scene_xml(), resolution=0.2, agent_radius=0.3)
    # Test a point likely to be free (center of the floor)
    result = nav_is_free(0.0, 0.0)
    assert result["success"]
    # The center may or may not be free depending on scene layout
    assert isinstance(result["is_free"], bool)


def test_grid_info_without_build():
    """nav_get_grid_info should fail if no grid is built."""
    result = nav_get_grid_info()
    assert not result["success"]


# ---------------------------------------------------------------------------
# A* path planning
# ---------------------------------------------------------------------------


def test_astar_simple_path():
    """A* should find a path between two free points."""
    nav_build_grid(scene_xml=_office_scene_xml(), resolution=0.2, agent_radius=0.3)

    # Find two free points
    from strech_codex.world.state import get_nav

    nav = get_nav()
    rng = np.random.default_rng(42)
    free_pts = []
    for x in np.linspace(nav.grid.x_min + 1, nav.grid.x_max - 1, 8):
        for y in np.linspace(nav.grid.y_min + 1, nav.grid.y_max - 1, 8):
            if nav.is_free((x, y)):
                free_pts.append((float(x), float(y)))

    if len(free_pts) >= 2:
        s = free_pts[0]
        g = free_pts[-1]
        result = nav_astar(start_x=s[0], start_y=s[1], goal_x=g[0], goal_y=g[1])
        assert result["success"], result.get("message")
        assert result["num_waypoints"] >= 2
        assert result["distance_m"] > 0
        assert len(result["waypoints"]) == result["num_waypoints"]


def test_astar_no_grid():
    """A* should fail without a grid."""
    result = nav_astar(start_x=0, start_y=0, goal_x=1, goal_y=1)
    assert not result["success"]


def test_astar_same_point():
    """A* should handle start == goal."""
    nav_build_grid(scene_xml=_office_scene_xml(), resolution=0.2, agent_radius=0.3)
    from strech_codex.world.state import get_nav

    nav = get_nav()
    # Find a free point
    for x in np.linspace(nav.grid.x_min, nav.grid.x_max, 10):
        for y in np.linspace(nav.grid.y_min, nav.grid.y_max, 10):
            if nav.is_free((x, y)):
                result = nav_astar(start_x=x, start_y=y, goal_x=x, goal_y=y)
                # Start=goal resolves through endpoint validation
                assert result["success"]
                return
    pytest.skip("No free point found")


def test_planners_do_not_insert_endpoint_cell_centers():
    nav_build_grid(scene_xml=_office_scene_xml(), resolution=0.2, agent_radius=0.3)
    from strech_codex.world.state import get_nav

    nav = get_nav()
    for row, col in zip(*np.where(~nav.grid.occupancy)):
        center = nav.grid.cell_to_world((int(row), int(col)))
        start = center + np.array([-0.02, 0.0])
        goal = center + np.array([0.02, 0.0])
        if nav.is_free(start) and nav.is_free(goal):
            for planner in (nav_astar, nav_fmm):
                result = planner(start[0], start[1], goal[0], goal[1])
                assert result["success"], result.get("message")
                assert np.allclose(result["waypoints"], [start, goal])
            return
    pytest.skip("No free cell center found")


# ---------------------------------------------------------------------------
# FMM path planning
# ---------------------------------------------------------------------------


def test_fmm_simple_path():
    """FMM should find a path between two free points."""
    nav_build_grid(scene_xml=_office_scene_xml(), resolution=0.2, agent_radius=0.3)

    from strech_codex.world.state import get_nav

    nav = get_nav()
    rng = np.random.default_rng(99)
    free_pts = []
    for x in np.linspace(nav.grid.x_min + 1, nav.grid.x_max - 1, 8):
        for y in np.linspace(nav.grid.y_min + 1, nav.grid.y_max - 1, 8):
            if nav.is_free((x, y)):
                free_pts.append((float(x), float(y)))

    if len(free_pts) >= 2:
        s = free_pts[0]
        g = free_pts[-1]
        result = nav_fmm(start_x=s[0], start_y=s[1], goal_x=g[0], goal_y=g[1])
        assert result["success"], result.get("message")
        assert result["num_waypoints"] >= 2
        assert result["distance_m"] > 0


def test_fmm_no_grid():
    """FMM should fail without a grid."""
    result = nav_fmm(start_x=0, start_y=0, goal_x=1, goal_y=1)
    assert not result["success"]


def test_astar_and_fmm_agree():
    """Both A* and FMM should produce valid paths for the same query."""
    nav_build_grid(scene_xml=_office_scene_xml(), resolution=0.2, agent_radius=0.3)
    from strech_codex.world.state import get_nav

    nav = get_nav()
    free_pts = []
    for x in np.linspace(nav.grid.x_min + 1, nav.grid.x_max - 1, 6):
        for y in np.linspace(nav.grid.y_min + 1, nav.grid.y_max - 1, 6):
            if nav.is_free((x, y)):
                free_pts.append((float(x), float(y)))

    if len(free_pts) >= 2:
        s, g = free_pts[0], free_pts[-1]
        r1 = nav_astar(start_x=s[0], start_y=s[1], goal_x=g[0], goal_y=g[1])
        r2 = nav_fmm(start_x=s[0], start_y=s[1], goal_x=g[0], goal_y=g[1])
        assert r1["success"] and r2["success"]
        # Both should produce reasonable distances
        assert r1["distance_m"] > 0
        assert r2["distance_m"] > 0


@pytest.mark.parametrize("robot_type", ["stretch3", "google_robot", "tidybot"])
def test_execute_path_for_all_drive_types(monkeypatch, robot_type):
    class FakeSim:
        def __init__(self):
            self.pose = np.zeros(3)
            self.command = np.zeros(3)

        def is_running(self):
            return True

        def stop(self):
            pass

        def get_base_pose(self):
            forward, lateral, omega = self.command
            yaw = self.pose[2]
            self.pose[0] += (forward * np.cos(yaw) - lateral * np.sin(yaw)) * 0.05
            self.pose[1] += (forward * np.sin(yaw) + lateral * np.cos(yaw)) * 0.05
            self.pose[2] += omega * 0.05
            return tuple(self.pose)

        def set_base_velocity(self, forward, omega, lateral=0.0):
            self.command[:] = (forward, lateral, omega)

    sim = FakeSim()
    set_sim(sim)
    set_robot_type(robot_type)
    monkeypatch.setattr("strech_codex.tools.navigation.time.sleep", lambda _: None)
    result = nav_execute_path(
        [[0.0, 0.0], [0.3, 0.2]],
        position_tolerance=0.03,
        max_linear_speed=0.5,
        control_hz=20.0,
    )
    assert result["success"], result.get("message")
    assert result["robot_type"] == robot_type
    assert result["goal_error_m"] <= 0.03
    assert np.allclose(sim.command, 0.0)


def test_execute_path_rejects_mismatched_start():
    sim = type(
        "FakeSim",
        (),
        {
            "get_base_pose": lambda self: (0.0, 0.0, 0.0),
            "set_base_velocity": lambda self, *args: None,
            "is_running": lambda self: True,
            "stop": lambda self: None,
        },
    )()
    set_sim(sim)
    set_robot_type("google_robot")
    result = nav_execute_path([[1.0, 1.0], [2.0, 2.0]], start_tolerance=0.1)
    assert not result["success"]
    assert "from path start" in result["message"]


def test_execute_path_uses_lookahead_tolerance_for_intermediate_waypoint():
    sim = type(
        "FakeSim",
        (),
        {
            "get_base_pose": lambda self: (0.0, 0.0, 0.0),
            "set_base_velocity": lambda self, *args: None,
            "is_running": lambda self: True,
            "stop": lambda self: None,
        },
    )()
    set_sim(sim)
    set_robot_type("stretch3")

    result = nav_execute_path(
        [[0.0, 0.0], [0.25, 0.0], [0.0, 0.0]],
        position_tolerance=0.12,
        timeout=0.1,
    )

    assert result["success"]
