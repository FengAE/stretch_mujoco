"""Closed-loop NavDP planner.

NavDP (Navigation Diffusion Policy) is an end-to-end mapless local navigation
policy: given a body-frame goal and the current RGB-D view it diffuses a
waypoint trajectory which is then followed with a pure-Python lookahead
tracker.  The heavy model runs on a remote ``navdp_server.py`` process (GPU
workstation); this class only speaks HTTP.

Coordinate convention (verified against the official Isaac/Go2 evaluation):
the goal sent to the server and the returned trajectory are both in the
**robot body frame** at the current pose.  This planner converts a world-frame
goal into the body frame, and converts the returned body-frame trajectory back
into world frame before tracking.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from stretch_mujoco.navigations.NavDP.navdp_client import NavDPClient
from stretch_mujoco.navigations.NavDP.trajectory_tracker import TrajectoryTracker


@dataclass
class NavDPDiagnostics:
    """Diagnostics for one NavDP navigation episode."""

    replan_count: int = 0
    reset_count: int = 0
    last_traj_body: Optional[np.ndarray] = None  # (T, 3) body-frame waypoints
    last_traj_world: Optional[np.ndarray] = None  # (T, 2) world-frame waypoints
    dist_to_goal: float = float("inf")
    last_error: str = ""


class NavDPPlanner:
    """Stateful closed-loop NavDP controller emitting ``(v, w)`` commands."""

    def __init__(
        self,
        client: NavDPClient,
        intrinsic: np.ndarray | list,
        *,
        stop_threshold: float = -0.5,
        batch_size: int = 1,
        tracker: Optional[TrajectoryTracker] = None,
        plan_interval: int = 5,
    ) -> None:
        if not isinstance(client, NavDPClient):
            raise TypeError("client must be a NavDPClient")
        intrinsic_arr = np.asarray(intrinsic, dtype=float)
        if intrinsic_arr.shape != (3, 3):
            raise ValueError(f"intrinsic must be a 3x3 matrix, got {intrinsic_arr.shape}")
        if not np.isfinite(stop_threshold):
            raise ValueError("stop_threshold must be finite")
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
            raise ValueError("batch_size must be a positive int")
        if not isinstance(plan_interval, int) or isinstance(plan_interval, bool) or plan_interval <= 0:
            raise ValueError("plan_interval must be a positive int")

        self._client = client
        self._intrinsic = intrinsic_arr
        self._stop_threshold = float(stop_threshold)
        self._batch_size = int(batch_size)
        self._tracker = tracker if tracker is not None else TrajectoryTracker()
        self._plan_interval = int(plan_interval)
        self._steps_since_replan = self._plan_interval  # force a replan on the first step
        self._last_replan_pose: Optional[np.ndarray] = None
        self._reset_done = False
        self.diag = NavDPDiagnostics()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """(Re)initialise the server-side policy and clear local state."""
        self._client.reset(
            self._intrinsic,
            stop_threshold=self._stop_threshold,
            batch_size=self._batch_size,
        )
        self._tracker.clear()
        self._reset_done = True
        self.diag = NavDPDiagnostics(reset_count=1)

    def step(
        self,
        robot_xy: np.ndarray,
        robot_yaw: float,
        rgb_bytes: bytes,
        depth_m: np.ndarray,
        goal_xy: np.ndarray,
        *,
        replan: Optional[bool] = None,
    ) -> tuple[float, float]:
        """Advance the closed loop by one control tick and return ``(v, w)``.

        Parameters
        ----------
        robot_xy:
            World ``(x, y)`` of the robot base.
        robot_yaw:
            World heading in radians.
        rgb_bytes:
            JPEG/PNG bytes of the current egocentric RGB view.
        depth_m:
            ``(H, W)`` float32 depth in metres aligned to the RGB view.
        goal_xy:
            World ``(x, y)`` goal position.
        replan:
            Force a replan.  When ``None``, replan only when the current
            trajectory is nearly consumed.
        """
        robot = np.asarray(robot_xy, dtype=float)[:2]
        goal = np.asarray(goal_xy, dtype=float)[:2]
        yaw = float(robot_yaw)

        if not self._reset_done:
            self.reset()

        # Body-frame goal for the policy.
        goal_body = _rotate(-yaw) @ (goal - robot)

        self._steps_since_replan += 1
        if replan is None:
            # Replan when the current trajectory is nearly consumed, or when
            # the robot has failed to make progress for a while (starvation
            # safety net — a stale/unreachable plan must not stall forever).
            advanced = (
                0.0
                if self._last_replan_pose is None
                else float(np.linalg.norm(robot - self._last_replan_pose))
            )
            replan = self._tracker.near_end(robot) or (
                self._steps_since_replan >= self._plan_interval and advanced < 0.05
            )
        if replan:
            self._steps_since_replan = 0
            self._last_replan_pose = robot.copy()
            try:
                traj_body = np.asarray(
                    self._client.step_pointgoal(goal_body.reshape(1, 2), rgb_bytes, depth_m),
                    dtype=float,
                )
                # The server returns a batched trajectory (batch_size=1).
                if traj_body.ndim == 3 and traj_body.shape[0] == 1:
                    traj_body = traj_body[0]
                traj_world = robot + traj_body[:, :2] @ _rotate(yaw).T
                self._tracker.update(traj_world)
                self.diag.last_traj_body = traj_body
                self.diag.last_traj_world = traj_world
                self.diag.replan_count += 1
                self.diag.last_error = ""
            except Exception as exc:  # keep following the previous trajectory
                self.diag.last_error = f"{type(exc).__name__}: {exc}"

        self.diag.dist_to_goal = float(np.linalg.norm(robot - goal))
        return self._tracker.step(robot, yaw)

    @property
    def tracker(self) -> TrajectoryTracker:
        return self._tracker


def _rotate(angle: float) -> np.ndarray:
    """2-D rotation matrix ``Rz(angle)``."""
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s], [s, c]], dtype=float)
