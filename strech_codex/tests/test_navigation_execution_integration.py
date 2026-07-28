"""End-to-end navigation execution on the Google Robot default scene."""

import pytest

from strech_codex.tools.navigation import nav_astar, nav_build_grid, nav_execute_path
from strech_codex.tools.robot import robot_init
from strech_codex.world.state import reset_world


@pytest.mark.slow
def test_google_robot_executes_planned_route():
    reset_world()
    try:
        started = robot_init(
            "google_robot", headless=True, start_translation=[-0.7, -2.7, 0.0]
        )
        assert started["success"], started.get("message")
        grid = nav_build_grid(resolution=0.15, agent_radius=0.25)
        assert grid["success"], grid.get("message")
        planned = nav_astar(-0.7, -2.7, 2.2, -0.5)
        assert planned["success"], planned.get("message")
        executed = nav_execute_path(
            planned["waypoints"], timeout=60.0, max_linear_speed=0.5
        )
        assert executed["success"], executed.get("message")
        assert executed["goal_error_m"] <= 0.12
    finally:
        reset_world()
