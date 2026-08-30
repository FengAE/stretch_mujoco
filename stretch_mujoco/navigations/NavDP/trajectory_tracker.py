"""Pure-NumPy lookahead trajectory tracker for a differential-drive base.

Follows a world-frame waypoint trajectory with a simple PD law: forward
speed from the along-path remaining distance, angular speed from the bearing
error to a lookahead point.  No external dependencies (no casadi).
"""

from __future__ import annotations

import math

import numpy as np


class TrajectoryTracker:
    """Follow a world-frame ``(N, 2)`` trajectory with ``(v, w)`` commands."""

    def __init__(
        self,
        lookahead: float = 0.4,
        kv: float = 1.0,
        kw: float = 2.0,
        max_v: float = 0.4,
        max_w: float = 0.6,
        goal_tol: float = 0.3,
    ) -> None:
        if not np.isfinite(lookahead) or lookahead <= 0:
            raise ValueError("lookahead must be finite and > 0")
        if kv <= 0 or kw <= 0:
            raise ValueError("kv and kw must be > 0")
        if max_v <= 0 or max_w <= 0:
            raise ValueError("max_v and max_w must be > 0")
        if not np.isfinite(goal_tol) or goal_tol <= 0:
            raise ValueError("goal_tol must be finite and > 0")

        self.lookahead = float(lookahead)
        self.kv = float(kv)
        self.kw = float(kw)
        self.max_v = float(max_v)
        self.max_w = float(max_w)
        self.goal_tol = float(goal_tol)

        self._trajectory: np.ndarray = np.zeros((0, 2))  # world-frame waypoints

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, trajectory_world: np.ndarray) -> None:
        """Replace the current trajectory with new world-frame waypoints."""
        traj = np.asarray(trajectory_world, dtype=float)
        if traj.ndim == 1:
            traj = traj.reshape(-1, 2)
        if traj.ndim != 2 or traj.shape[1] != 2:
            raise ValueError(f"trajectory_world must have shape (N, 2), got {traj.shape}")
        if not np.all(np.isfinite(traj)):
            raise ValueError("trajectory_world must contain only finite values")
        self._trajectory = traj

    def clear(self) -> None:
        """Drop the current trajectory (robot should stop)."""
        self._trajectory = np.zeros((0, 2))

    def has_trajectory(self) -> bool:
        return self._trajectory.shape[0] >= 2

    def near_end(self, robot_xy: np.ndarray) -> bool:
        """Return True when the robot is near the end of the trajectory."""
        if not self.has_trajectory():
            return True
        end = self._trajectory[-1]
        return float(np.linalg.norm(np.asarray(robot_xy)[:2] - end)) <= self.lookahead

    def reached(self, robot_xy: np.ndarray, goal_xy: np.ndarray, goal_tol: float | None = None) -> bool:
        """Return True when the robot is within *goal_tol* of the goal."""
        tol = self.goal_tol if goal_tol is None else float(goal_tol)
        return float(np.linalg.norm(np.asarray(robot_xy)[:2] - np.asarray(goal_xy)[:2])) <= tol

    def step(self, robot_xy: np.ndarray, robot_yaw: float) -> tuple[float, float]:
        """Compute one ``(v, w)`` velocity command from the current pose."""
        robot = np.asarray(robot_xy, dtype=float)[:2]
        if not self.has_trajectory():
            return 0.0, 0.0

        target = self._lookahead_point(robot)
        if target is None:
            return 0.0, 0.0

        dx, dy = target - robot
        dist = float(np.hypot(dx, dy))
        heading_error = _wrap_angle(math.atan2(dy, dx) - float(robot_yaw))

        v = float(np.clip(self.kv * dist, 0.0, self.max_v))
        w = float(np.clip(self.kw * heading_error, -self.max_w, self.max_w))
        # Don't drive forward while facing far from the target.
        if abs(heading_error) > math.radians(90):
            v = 0.0
        return v, w

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _lookahead_point(self, robot_xy: np.ndarray) -> np.ndarray | None:
        """Return the waypoint ~*lookahead* metres ahead along the path."""
        pts = self._trajectory
        dists = np.linalg.norm(pts - robot_xy, axis=1)
        nearest = int(np.argmin(dists))

        # Walk forward along the path by arc length until we reach lookahead.
        travelled = 0.0
        for i in range(nearest, len(pts) - 1):
            segment = float(np.linalg.norm(pts[i + 1] - pts[i]))
            if travelled + segment >= self.lookahead:
                fraction = (self.lookahead - travelled) / max(segment, 1e-9)
                return pts[i] + fraction * (pts[i + 1] - pts[i])
            travelled += segment
        # Lookahead point falls past the end — use the final waypoint.
        return pts[-1]


def _wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))
