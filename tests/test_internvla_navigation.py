"""Offline tests for the InternVLA-N1 dual-system navigation module.

No server and no rendering are required: the HTTP client is faked and the two
response branches (trajectory / discrete_action) are exercised with geometry.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from stretch_mujoco.navigations import Algorithm, NavigationController
from stretch_mujoco.navigations.InternVLAS2 import (
    InternVLAClient,
    InternVLAS2Planner,
)


class FakeInternVLAClient(InternVLAClient):
    """In-memory stand-in that replays canned responses."""

    def __init__(self, responses: list | None = None) -> None:
        super().__init__(base_url="http://127.0.0.1:1")
        self.responses = list(responses or [])
        self.calls: list[dict] = []

    def step(self, rgb_bytes, depth_m, *, reset=False, instruction=None) -> dict:
        self.calls.append({"reset": reset, "instruction": instruction})
        if self.responses:
            return self.responses.pop(0)
        return {"trajectory": [[0.2, 0.0], [1.0, 0.0], [1.8, 0.0]]}


def _obs() -> tuple[bytes, np.ndarray]:
    return b"x", np.full((240, 424), 2.0, dtype=np.float32)


# ---------------------------------------------------------------------------
# Trajectory branch
# ---------------------------------------------------------------------------


def test_trajectory_closed_loop_advances_and_resets_once() -> None:
    client = FakeInternVLAClient()
    planner = InternVLAS2Planner(client, instruction="go forward")
    robot = np.array([0.0, 0.0], dtype=float)
    yaw = 0.0
    dt = 0.1
    for _ in range(100):
        v, w = planner.step(robot, yaw, *_obs())
        yaw += w * dt
        robot = robot + v * np.array([np.cos(yaw), np.sin(yaw)]) * dt

    assert robot[0] > 1.0  # made forward progress
    assert client.calls[0]["reset"] is True  # first call resets the server
    assert client.calls[1]["reset"] is False
    assert planner.diag.last_response_type == "trajectory"


def test_trajectory_body_to_world_transform() -> None:
    client = FakeInternVLAClient(
        responses=[{"trajectory": [[0.2, 0.0], [1.0, 0.0]], "pixel_goal": [320, 120]}]
    )
    planner = InternVLAS2Planner(client, instruction="go")
    planner.step((1.0, 1.0), np.pi / 2.0, *_obs())

    traj = planner.tracker._trajectory  # world-frame waypoints
    # Body (1, 0) with robot yaw=+90° → world (1, 2).
    np.testing.assert_allclose(traj[-1], [1.0, 2.0], atol=1e-6)
    assert planner.diag.last_pixel == [320, 120]


# ---------------------------------------------------------------------------
# Discrete action branch
# ---------------------------------------------------------------------------


def test_discrete_forward_moves_goal_then_stop() -> None:
    client = FakeInternVLAClient(responses=[{"discrete_action": [1]}, {"discrete_action": [0]}])
    planner = InternVLAS2Planner(client, instruction="go")
    robot = np.array([0.0, 0.0], dtype=float)

    v, w = planner.step(robot, 0.0, *_obs())
    assert planner.diag.last_discrete_action == [1]
    assert planner._goal_pose[0] > 0.0  # forward 0.25 m in +x
    assert v > 0.0  # drives forward toward the goal
    assert abs(w) < 1e-9  # goal straight ahead, same heading

    # Finish the forward chunk before requesting the next S2 action.
    planner.step(np.array([0.25, 0.0]), 0.0, *_obs())
    assert planner.diag.stop_requested is True
    assert planner.diag.last_discrete_action == [0]


def test_discrete_turn_rotates_in_place() -> None:
    client = FakeInternVLAClient(responses=[{"discrete_action": [2]}])
    planner = InternVLAS2Planner(client, instruction="go")
    robot = np.array([0.0, 0.0], dtype=float)

    v, w = planner.step(robot, 0.0, *_obs())
    # [2] = turn left by 15°; position unchanged → rotate in place.
    assert np.isclose(planner._goal_pose[2], np.radians(15.0))
    assert v == 0.0
    assert w > 0.0


def test_discrete_chunk_finishes_before_requesting_next_action() -> None:
    client = FakeInternVLAClient(
        responses=[{"discrete_action": [2, 2, 2, 2]}, {"discrete_action": [1]}]
    )
    planner = InternVLAS2Planner(client, instruction="go")

    planner.step((0.0, 0.0), 0.0, *_obs())
    assert len(client.calls) == 1

    # The 60-degree turn has not completed, so no new S2 request is made.
    planner.step((0.0, 0.0), np.radians(20.0), *_obs())
    assert len(client.calls) == 1

    # Once the chunk is complete, the next observation is sent to S2.
    v, _ = planner.step((0.0, 0.0), np.radians(60.0), *_obs())
    assert len(client.calls) == 2
    assert v > 0.0


# ---------------------------------------------------------------------------
# Controller integration
# ---------------------------------------------------------------------------


def _scene_path() -> str:
    import stretch_mujoco

    return str(Path(stretch_mujoco.__file__).resolve().parent / "models" / "office_scene.xml")


def test_controller_registers_internvla_algorithm() -> None:
    nav = NavigationController.from_scene_xml(
        _scene_path(),
        algorithm=Algorithm.INTERVLAS2,
        planner_kwargs={"client": FakeInternVLAClient(), "instruction": "go"},
    )
    assert nav.algorithm is Algorithm.INTERVLAS2
    assert isinstance(nav.internvla, InternVLAS2Planner)
    # Grid-based planning is not available for the closed-loop policy.
    with pytest.raises(RuntimeError, match="closed-loop"):
        nav.plan(start=(0.0, 0.0), goal=(1.0, 1.0))
