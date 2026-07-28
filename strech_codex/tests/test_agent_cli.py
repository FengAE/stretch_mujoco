"""Small regression tests for CLI configuration and offline parsing."""

from pathlib import Path

from strech_codex.agent import plan_offline
from strech_codex.cli import _DEFAULT_OFFICE_SCENE
from strech_codex.world.state import reset_world, set_robot_type, set_scene_path


def setup_function():
    reset_world()


def teardown_function():
    reset_world()


def test_default_scene_exists():
    assert Path(_DEFAULT_OFFICE_SCENE).is_file()


def test_cli_robot_selection_is_used_by_planner():
    set_robot_type("google_robot")
    calls = plan_offline("启动机器人")
    assert calls[0].arguments["robot_type"] == "google_robot"


def test_joint_move_is_not_navigation_or_implicit_lift():
    set_scene_path("/tmp/scene.xml")
    calls = plan_offline("移动 lift 到 0.5")
    assert [(call.name, call.arguments) for call in calls] == [
        ("robot_move_to", {"actuator_name": "lift", "position": 0.5})
    ]


def test_malformed_coordinate_does_not_crash():
    calls = plan_offline("从 (1..2, 0) 到 (1, 2)")
    assert all(call.name != "nav_astar" for call in calls)
