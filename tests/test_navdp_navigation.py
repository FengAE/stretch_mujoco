"""Offline tests for the NavDP closed-loop navigation module.

No HTTP server and no MuJoCo rendering are required: the client is faked and
the tracker is exercised with pure geometry.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from stretch_mujoco.navigations import Algorithm, NavigationController
from stretch_mujoco.navigations.NavDP import (
    NavDPClient,
    NavDPPlanner,
    TrajectoryTracker,
)


# ---------------------------------------------------------------------------
# TrajectoryTracker
# ---------------------------------------------------------------------------


def test_tracker_follows_straight_line_toward_goal() -> None:
    tracker = TrajectoryTracker(lookahead=0.4, kv=1.0, kw=2.0, max_v=0.4, max_w=0.6)
    goal = np.array([2.0, 0.0])
    traj = np.column_stack([np.linspace(0.2, 2.0, 10), np.zeros(10)])
    tracker.update(traj)

    robot = np.array([0.0, 0.0])
    v, w = tracker.step(robot, 0.0)
    assert v > 0.0
    assert abs(w) < 1e-6  # goal straight ahead → no rotation

    # A robot that has almost reached the goal reports reached.
    assert tracker.reached(np.array([1.9, 0.0]), goal, goal_tol=0.3)
    assert not tracker.reached(np.array([0.0, 0.0]), goal, goal_tol=0.3)

    # No trajectory → stationary.
    tracker.clear()
    assert tracker.step(robot, 0.0) == (0.0, 0.0)


def test_tracker_rotates_toward_offset_goal() -> None:
    tracker = TrajectoryTracker(kv=1.0, kw=2.0, max_v=0.4, max_w=0.6)
    # Goal is 45° to the left of the robot heading.
    traj = np.array([[0.2, 0.0], [0.8, 0.6], [1.5, 1.3], [2.0, 2.0]])
    tracker.update(traj)
    v, w = tracker.step(np.array([0.0, 0.0]), 0.0)
    assert w > 0.0  # turn left


def test_tracker_near_end_triggers_replan() -> None:
    tracker = TrajectoryTracker(lookahead=0.4)
    traj = np.array([[0.2, 0.0], [0.4, 0.0], [0.6, 0.0]])
    tracker.update(traj)
    assert tracker.near_end(np.array([0.0, 0.0])) is False
    assert tracker.near_end(np.array([0.45, 0.0])) is True  # within lookahead of end
    tracker.clear()
    assert tracker.near_end(np.array([0.0, 0.0])) is True  # no trajectory → replan


# ---------------------------------------------------------------------------
# NavDPPlanner (faked client, no network)
# ---------------------------------------------------------------------------


class FakeNavDPClient(NavDPClient):
    """In-memory stand-in for the HTTP client."""

    def __init__(self) -> None:
        super().__init__(base_url="http://127.0.0.1:1")
        self.reset_count = 0
        self.last_goal_body: np.ndarray | None = None

    def reset(self, intrinsic, stop_threshold=-0.5, batch_size=1) -> dict:
        self.reset_count += 1
        return {"algo": "navdp"}

    def step_pointgoal(self, goal_xy, rgb_bytes, depth_m) -> np.ndarray:
        goal = np.asarray(goal_xy, dtype=float)
        self.last_goal_body = goal.reshape(-1)[:2]
        # Body-frame trajectory: from 0.2 m ahead straight toward the goal.
        xs = np.linspace(0.2, float(goal.reshape(-1)[0]), 10)
        ys = np.linspace(0.0, float(goal.reshape(-1)[1]), 10)
        return np.column_stack([xs, ys, np.zeros(10)])


def test_navdp_planner_sends_body_frame_goal_and_returns_velocity() -> None:
    client = FakeNavDPClient()
    intrinsic = np.eye(3)
    planner = NavDPPlanner(client, intrinsic)

    rgb = b"\x00\x01\x02"
    depth = np.full((240, 424), 2.0, dtype=np.float32)

    # Robot at world origin facing +x, goal straight ahead.
    v, w = planner.step((0.0, 0.0), 0.0, rgb, depth, (2.0, 0.0))
    assert planner.diag.replan_count == 1
    assert client.reset_count == 1
    assert v > 0.0
    assert abs(w) < 1e-6
    # Body-frame goal seen by the "server" equals the world goal (same frame).
    np.testing.assert_allclose(client.last_goal_body, [2.0, 0.0], atol=1e-9)
    # The returned body trajectory was transformed back into world frame.
    np.testing.assert_allclose(planner.diag.last_traj_world[-1], [2.0, 0.0], atol=1e-6)


def test_navdp_planner_handles_rotated_robot() -> None:
    client = FakeNavDPClient()
    planner = NavDPPlanner(client, np.eye(3))

    rgb = b"\x00\x01\x02"
    depth = np.full((240, 424), 2.0, dtype=np.float32)

    # Robot at (1, 1) facing +y; goal at (2, 1) is 1 m to its right (-x in body).
    v, w = planner.step((1.0, 1.0), np.pi / 2.0, rgb, depth, (2.0, 1.0))
    # Body-frame goal: (0, -1) — directly to the robot's right.
    np.testing.assert_allclose(client.last_goal_body, [0.0, -1.0], atol=1e-9)
    assert w < 0.0  # must turn right (clockwise) to face the goal
    np.testing.assert_allclose(planner.diag.last_traj_world[-1], [2.0, 1.0], atol=1e-6)


def test_navdp_closed_loop_reaches_goal() -> None:
    """Full closed loop (no rendering, no server): the point robot must reach
    the goal, exercising reset → replan (near-end + progress) → tracker."""

    class GoalClient(FakeNavDPClient):
        def step_pointgoal(self, goal_xy, rgb_bytes, depth_m) -> np.ndarray:
            super().step_pointgoal(goal_xy, rgb_bytes, depth_m)
            gx, gy = (float(v) for v in np.asarray(goal_xy).reshape(-1)[:2])
            xs = np.linspace(0.2, gx, 12)
            ys = np.linspace(0.0, gy, 12)
            return np.column_stack([xs, ys, np.zeros(12)])

    planner = NavDPPlanner(GoalClient(), np.eye(3), stop_threshold=-1.5, plan_interval=10)
    robot = np.array([0.0, 0.0], dtype=float)
    yaw = 0.0
    goal = np.array([2.0, 0.5])
    dt = 0.1
    path = [robot.copy()]
    reached = False
    for _ in range(200):
        v, w = planner.step(robot, yaw, b"x", np.full((240, 424), 2.0, np.float32), goal)
        yaw += w * dt
        robot = robot + v * np.array([np.cos(yaw), np.sin(yaw)]) * dt
        path.append(robot.copy())
        if planner.tracker.reached(robot, goal, goal_tol=0.3):
            reached = True
            break
    assert reached, f"closed loop did not reach goal, final pose {robot}, path len {len(path)}"


def test_navdp_planner_keeps_old_trajectory_on_replan_failure() -> None:
    class FlakyClient(FakeNavDPClient):
        def step_pointgoal(self, goal_xy, rgb_bytes, depth_m) -> np.ndarray:
            raise RuntimeError("simulated network error")

    planner = NavDPPlanner(FlakyClient(), np.eye(3))
    v, w = planner.step((0.0, 0.0), 0.0, b"x", np.full((240, 424), 2.0, np.float32), (2.0, 0.0))
    assert planner.diag.last_error != ""
    assert v == 0.0 and w == 0.0  # no trajectory → stationary, no crash


# ---------------------------------------------------------------------------
# Controller integration
# ---------------------------------------------------------------------------


def _scene_path() -> str:
    import stretch_mujoco

    return str(Path(stretch_mujoco.__file__).resolve().parent / "models" / "office_scene.xml")


def test_controller_registers_navdp_algorithm() -> None:
    nav = NavigationController.from_scene_xml(
        _scene_path(),
        algorithm=Algorithm.NAVDP,
        planner_kwargs={"client": FakeNavDPClient(), "intrinsic": np.eye(3)},
    )
    assert nav.algorithm is Algorithm.NAVDP
    assert isinstance(nav.navdp, NavDPPlanner)
    # Grid-based planning is not available for the closed-loop policy.
    with pytest.raises(RuntimeError, match="closed-loop"):
        nav.plan(start=(0.0, 0.0), goal=(1.0, 1.0))
