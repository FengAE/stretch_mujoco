"""QwenS2Planner — instruction-driven goal navigation.

Composes an online Qwen2.5-VL pixel selector with the NavDP pixel-goal policy:

1. The VLM looks at the current camera image + a natural-language instruction
   and points at the goal **pixel** (``None`` when it judges the goal reached).
2. NavDP's ``/pixelgoal_step`` diffuses a trajectory toward that pixel.
3. A lookahead tracker follows the trajectory and emits ``(v, w)``.

The language model runs online (cloud API); NavDP runs on the GPU workstation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from stretch_mujoco.navigations.NavDP.navdp_client import NavDPClient
from stretch_mujoco.navigations.NavDP.trajectory_tracker import TrajectoryTracker
from stretch_mujoco.navigations.QwenS2.qwen_client import QwenPixelGoalSelector


@dataclass
class QwenS2Diagnostics:
    """Diagnostics for one instruction-driven navigation episode."""

    replan_count: int = 0
    last_pixel: Optional[Tuple[int, int]] = None
    last_instruction: str = ""
    stop_requested: bool = False
    last_error: str = ""


class QwenS2Planner:
    """Stateful closed-loop controller emitting ``(v, w)`` for an instruction."""

    def __init__(
        self,
        client: NavDPClient,
        selector: QwenPixelGoalSelector,
        intrinsic: np.ndarray | list,
        *,
        instruction: str = "",
        image_size: Tuple[int, int] = (424, 240),
        stop_threshold: float = -1.5,
        batch_size: int = 1,
        tracker: Optional[TrajectoryTracker] = None,
        plan_interval: int = 8,
    ) -> None:
        if not isinstance(client, NavDPClient):
            raise TypeError("client must be a NavDPClient")
        if not isinstance(selector, QwenPixelGoalSelector):
            raise TypeError("selector must be a QwenPixelGoalSelector")
        intrinsic_arr = np.asarray(intrinsic, dtype=float)
        if intrinsic_arr.shape != (3, 3):
            raise ValueError(f"intrinsic must be a 3x3 matrix, got {intrinsic_arr.shape}")
        if not isinstance(instruction, str):
            raise TypeError("instruction must be a string")
        if len(image_size) != 2 or int(image_size[0]) <= 0 or int(image_size[1]) <= 0:
            raise ValueError(f"image_size must be (width, height), got {image_size}")
        if not np.isfinite(stop_threshold):
            raise ValueError("stop_threshold must be finite")
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
            raise ValueError("batch_size must be a positive int")
        if not isinstance(plan_interval, int) or isinstance(plan_interval, bool) or plan_interval <= 0:
            raise ValueError("plan_interval must be a positive int")

        self._client = client
        self._selector = selector
        self._intrinsic = intrinsic_arr
        self._instruction = instruction
        self._image_size = (int(image_size[0]), int(image_size[1]))
        self._stop_threshold = float(stop_threshold)
        self._batch_size = int(batch_size)
        self._tracker = tracker if tracker is not None else TrajectoryTracker()
        self._plan_interval = int(plan_interval)
        self._steps_since_replan = self._plan_interval  # force a replan on the first step
        self._reset_done = False
        self.diag = QwenS2Diagnostics(last_instruction=self._instruction)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """(Re)initialise the NavDP server and clear local state."""
        self._client.reset(
            self._intrinsic,
            stop_threshold=self._stop_threshold,
            batch_size=self._batch_size,
        )
        self._tracker.clear()
        self._reset_done = True
        self.diag = QwenS2Diagnostics(last_instruction=self._instruction)

    def step(
        self,
        robot_xy: np.ndarray,
        robot_yaw: float,
        rgb_bytes: bytes,
        depth_m: np.ndarray,
        instruction: Optional[str] = None,
    ) -> Tuple[float, float]:
        """Advance one control tick and return ``(v, w)``.

        ``instruction`` overrides the constructor-level instruction for this
        tick (e.g. when the task prompt changes).
        """
        robot = np.asarray(robot_xy, dtype=float)[:2]
        yaw = float(robot_yaw)
        instruction = instruction if instruction is not None else self._instruction
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("instruction must be a non-empty string")
        self.diag.last_instruction = instruction

        if not self._reset_done:
            self.reset()

        self._steps_since_replan += 1
        replan = self._tracker.near_end(robot) or self._steps_since_replan >= self._plan_interval
        if replan:
            self._steps_since_replan = 0
            try:
                pixel = self._selector.select_goal(rgb_bytes, instruction, self._image_size)
                if pixel is None:
                    # The VLM judged the goal reached.
                    self.diag.stop_requested = True
                    self.diag.last_error = ""
                    self._tracker.clear()
                    return 0.0, 0.0
                self.diag.stop_requested = False
                self.diag.last_pixel = pixel

                traj_body = np.asarray(
                    self._client.step_pixelgoal(pixel, rgb_bytes, depth_m),
                    dtype=float,
                )
                # The server returns a batched trajectory (batch_size=1).
                if traj_body.ndim == 3 and traj_body.shape[0] == 1:
                    traj_body = traj_body[0]
                traj_world = robot + traj_body[:, :2] @ _rotate(yaw).T
                self._tracker.update(traj_world)
                self.diag.replan_count += 1
                self.diag.last_error = ""
            except Exception as exc:  # keep following the previous trajectory
                self.diag.last_error = f"{type(exc).__name__}: {exc}"

        return self._tracker.step(robot, yaw)

    @property
    def tracker(self) -> TrajectoryTracker:
        return self._tracker


def _rotate(angle: float) -> np.ndarray:
    """2-D rotation matrix ``Rz(angle)``."""
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s], [s, c]], dtype=float)
