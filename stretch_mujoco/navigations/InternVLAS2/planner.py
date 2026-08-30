"""InternVLAS2Planner — closed-loop navigation with the InternVLA-N1 dual system.

The server (``http_internvla_server.py``) runs System 2 (Qwen2.5-VL, language →
goal) + System 1 (NavDP-style diffusion policy) in one process.  Each call
returns either:

- a **trajectory** (body-frame ``(N, 2)`` waypoints) that S1 re-plans while the
  S2 latent is active, or
- a **discrete_action** (``[0]=STOP, [1]=forward, [2]=left, [3]=right``) when
  S2 emits an action word.

This planner feeds the current RGB-D frame every control tick, transforms
trajectories to the world frame, applies discrete actions to an accumulated
goal pose, and follows everything with the shared lookahead :class:`TrajectoryTracker`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from stretch_mujoco.navigations.InternVLAS2.internvla_client import InternVLAClient
from stretch_mujoco.navigations.NavDP.trajectory_tracker import TrajectoryTracker


@dataclass
class InternVLAS2Diagnostics:
    """Diagnostics for one dual-system navigation episode."""

    replan_count: int = 0
    last_response_type: str = ""  # "trajectory" | "discrete" | ""
    last_discrete_action: Optional[list] = None
    last_pixel: Optional[list] = None
    stop_requested: bool = False
    last_error: str = ""


class InternVLAS2Planner:
    """Stateful closed-loop controller emitting ``(v, w)`` for an instruction."""

    def __init__(
        self,
        client: InternVLAClient,
        *,
        instruction: str = "",
        tracker: Optional[TrajectoryTracker] = None,
        discrete_step: float = 0.25,
        turn_deg: float = 15.0,
        discrete_position_tol: float = 0.04,
        discrete_yaw_tol_deg: float = 3.0,
    ) -> None:
        if not isinstance(client, InternVLAClient):
            raise TypeError("client must be an InternVLAClient")
        if not isinstance(instruction, str):
            raise TypeError("instruction must be a string")
        if discrete_step <= 0:
            raise ValueError("discrete_step must be > 0")
        if not np.isfinite(turn_deg) or turn_deg <= 0:
            raise ValueError("turn_deg must be finite and > 0")
        if not np.isfinite(discrete_position_tol) or discrete_position_tol <= 0:
            raise ValueError("discrete_position_tol must be finite and > 0")
        if not np.isfinite(discrete_yaw_tol_deg) or discrete_yaw_tol_deg <= 0:
            raise ValueError("discrete_yaw_tol_deg must be finite and > 0")

        self._client = client
        self._instruction = instruction
        self._tracker = tracker if tracker is not None else TrajectoryTracker()
        self._discrete_step = float(discrete_step)
        self._turn_deg = float(turn_deg)
        self._discrete_position_tol = float(discrete_position_tol)
        self._discrete_yaw_tol = math.radians(float(discrete_yaw_tol_deg))
        self._reset_done = False
        self._goal_pose: Optional[np.ndarray] = None  # [x, y, yaw] world
        self.diag = InternVLAS2Diagnostics()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Mark the episode as needing a server-side reset on the next step."""
        self._reset_done = False
        self._goal_pose = None
        self._tracker.clear()
        self.diag = InternVLAS2Diagnostics()

    def step(
        self,
        robot_xy: np.ndarray,
        robot_yaw: float,
        rgb_bytes: bytes,
        depth_m: np.ndarray,
        *,
        instruction: Optional[str] = None,
    ) -> tuple[float, float]:
        """Advance one control tick and return ``(v, w)``."""
        robot = np.asarray(robot_xy, dtype=float)[:2]
        yaw = float(robot_yaw)
        instruction = instruction if instruction is not None else self._instruction

        reset_flag = not self._reset_done
        if self.diag.stop_requested:
            return 0.0, 0.0
        if self.diag.last_response_type == "discrete" and self._goal_pose is not None:
            if not self._discrete_goal_reached(robot, yaw):
                return self._drive_to_goal_pose(robot, yaw)
            self._goal_pose = None

        try:
            response = self._client.step(
                rgb_bytes, depth_m, reset=reset_flag, instruction=instruction
            )
        except Exception as exc:
            self.diag.last_error = f"{type(exc).__name__}: {exc}"
            return self._tracker.step(robot, yaw)

        if reset_flag:
            self._reset_done = True
            self._goal_pose = np.array([robot[0], robot[1], yaw], dtype=float)

        if not isinstance(response, dict):
            self.diag.last_error = f"ValueError: response must be an object, got {type(response).__name__}"
            return 0.0, 0.0
        has_trajectory = "trajectory" in response
        has_discrete = "discrete_action" in response
        if has_trajectory and has_discrete:
            self.diag.last_error = "ValueError: response contains both trajectory and discrete_action"
            return 0.0, 0.0

        if has_trajectory:
            try:
                self.diag.last_response_type = "trajectory"
                self.diag.last_pixel = response.get("pixel_goal")
                self._handle_trajectory(robot, yaw, response["trajectory"])
                self.diag.replan_count += 1
                return self._tracker.step(robot, yaw)
            except Exception as exc:
                self.diag.last_error = f"{type(exc).__name__}: {exc}"
                return 0.0, 0.0
        if has_discrete:
            try:
                self.diag.last_response_type = "discrete"
                self.diag.last_discrete_action = list(response["discrete_action"])
                self._handle_discrete(robot, yaw, self.diag.last_discrete_action)
                self.diag.replan_count += 1
                return self._drive_to_goal_pose(robot, yaw)
            except Exception as exc:
                self.diag.last_error = f"{type(exc).__name__}: {exc}"
                return 0.0, 0.0

        self.diag.last_error = "ValueError: response contains neither trajectory nor discrete_action"
        return 0.0, 0.0

    @property
    def tracker(self) -> TrajectoryTracker:
        return self._tracker

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _handle_trajectory(self, robot: np.ndarray, yaw: float, raw: object) -> None:
        """Follow a body-frame ``(N, 2)`` trajectory in the world frame."""
        traj_body = np.asarray(raw, dtype=float)
        if traj_body.ndim == 3 and traj_body.shape[0] == 1:
            traj_body = traj_body[0]
        if traj_body.ndim == 1:
            traj_body = traj_body.reshape(-1, 2)
        if traj_body.ndim != 2 or traj_body.shape[1] != 2:
            raise ValueError(f"unexpected trajectory shape {traj_body.shape}")
        world = robot + traj_body @ _rotate(yaw).T
        self._tracker.update(world)

    def _handle_discrete(self, robot: np.ndarray, yaw: float, actions: list) -> None:
        """Apply an action chunk relative to the current observed pose."""
        self._goal_pose = np.array([robot[0], robot[1], yaw], dtype=float)
        goal = self._goal_pose
        for action in actions:
            action = int(action)
            if action == 1:  # forward
                goal[:2] += self._discrete_step * np.array(
                    [math.cos(goal[2]), math.sin(goal[2])]
                )
            elif action == 2:  # turn left
                goal[2] += math.radians(self._turn_deg)
            elif action == 3:  # turn right
                goal[2] -= math.radians(self._turn_deg)
            elif action == 0:  # stop
                self.diag.stop_requested = True
                goal[:2] = robot[:2]
                goal[2] = yaw
                break
            elif action == 5:  # look down should have been consumed by the server
                self.diag.last_error = "unexpected discrete action 5"
                break
            else:
                self.diag.last_error = f"unknown discrete action {action}"
                break

    def _drive_to_goal_pose(self, robot: np.ndarray, yaw: float) -> tuple[float, float]:
        """Drive toward the accumulated goal pose (position **and** heading).

        Follows the Go2 client's PID semantics: forward speed from the forward
        component of the goal offset (in the robot frame), angular speed from
        the goal-heading error.  This handles both forward moves and
        turn-only actions (which leave the position unchanged).
        """
        if self._goal_pose is None:
            return 0.0, 0.0
        goal = self._goal_pose
        dx, dy = goal[:2] - robot
        dist = float(np.hypot(dx, dy))
        # Forward component of the goal offset in the robot frame.
        fwd = dx * math.cos(yaw) + dy * math.sin(yaw)
        yaw_err = _wrap(goal[2] - yaw)

        v = float(np.clip(self._tracker.kv * fwd, 0.0, self._tracker.max_v))
        # Do not drive forward while the goal is behind us.
        if dist > 1e-6 and abs(_wrap(math.atan2(dy, dx) - yaw)) > math.pi / 2.0:
            v = 0.0
        w = float(np.clip(self._tracker.kw * yaw_err, -self._tracker.max_w, self._tracker.max_w))
        return v, w

    def _discrete_goal_reached(self, robot: np.ndarray, yaw: float) -> bool:
        if self._goal_pose is None:
            return True
        position_error = float(np.linalg.norm(self._goal_pose[:2] - robot))
        yaw_error = abs(_wrap(self._goal_pose[2] - yaw))
        return position_error <= self._discrete_position_tol and yaw_error <= self._discrete_yaw_tol


def _wrap(angle: float) -> float:
    """Wrap an angle into ``[-pi, pi]``."""
    return math.atan2(math.sin(angle), math.cos(angle))


def _rotate(angle: float) -> np.ndarray:
    """2-D rotation matrix ``Rz(angle)``."""
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s], [s, c]], dtype=float)
