"""Acyclic frontier enforcement for VLFM exploration.

Aligned with ``vlfm.policy.utils.acyclic_enforcer.AcyclicEnforcer`` from the
upstream `rai-opensource/vlfm <https://github.com/rai-opensource/vlfm>`_
repository.

Without this guard, the robot can oscillate between two high-value frontiers
that trade places in the top ranks from one step to the next.  The enforcer
remembers visited ``(robot_pose, frontier)`` transitions and suppresses
frontiers that would create a short cycle with similar value differentials.
"""

from __future__ import annotations

import numpy as np


class AcyclicEnforcer:
    """Suppress frontier choices that would form a short cycle.

    Each time the robot commits to a frontier, the transition
    ``(robot_xy, frontier_xy, top_two_values)`` is recorded.  Before
    committing to a new frontier, the enforcer checks whether the proposed
    transition mirrors a recent one with a similar value profile.

    Parameters
    ----------
    distance_threshold:
        Two robot positions or frontier positions within this distance (m)
        are considered the same location.
    value_threshold:
        Two value profiles within this absolute difference are considered
        equivalent.
    max_history:
        Maximum number of past transitions to retain (FIFO).
    """

    def __init__(
        self,
        distance_threshold: float = 0.5,
        value_threshold: float = 0.05,
        max_history: int = 100,
    ) -> None:
        if not np.isfinite(distance_threshold) or distance_threshold <= 0:
            raise ValueError("distance_threshold must be finite and > 0")
        if not np.isfinite(value_threshold) or value_threshold < 0:
            raise ValueError("value_threshold must be finite and >= 0")
        if not isinstance(max_history, int) or isinstance(max_history, bool) or max_history <= 0:
            raise ValueError("max_history must be a positive integer")
        self._dist_thresh = float(distance_threshold)
        self._val_thresh = float(value_threshold)
        self._max_history = int(max_history)
        self._history: list[_Transition] = []

    def check_cyclic(
        self,
        robot_xy: np.ndarray,
        frontier_xy: np.ndarray,
        top_values: tuple[float, ...],
    ) -> bool:
        """Return True when this transition looks like a known cycle.

        Parameters
        ----------
        robot_xy:
            Current robot world position ``(x, y)``.
        frontier_xy:
            Proposed frontier world position ``(x, y)``.
        top_values:
            The best value scores from the current decision step, used to
            detect similar value landscapes.
        """
        if not self._history:
            return False

        for prev in reversed(self._history):
            # A cycle requires the robot to revisit a location and propose
            # a similar frontier, or vice versa.
            robot_close = float(np.linalg.norm(robot_xy - prev.robot_xy)) < self._dist_thresh
            frontier_close = (
                float(np.linalg.norm(frontier_xy - prev.frontier_xy)) < self._dist_thresh
            )

            if not (robot_close or frontier_close):
                continue

            # Value profiles must be close too
            values_close = all(
                abs(a - b) < self._val_thresh for a, b in zip(top_values, prev.top_values)
            )

            if values_close:
                # Same location or same frontier with same value landscape
                if robot_close and frontier_close:
                    return True
                # Revisiting the same robot location proposing a different
                # frontier with the same value differential → probably a
                # flip-flop.
                if robot_close or frontier_close:
                    return True

        return False

    def add_state_action(
        self,
        robot_xy: np.ndarray,
        frontier_xy: np.ndarray,
        top_values: tuple[float, ...],
    ) -> None:
        """Record a committed transition for future cycle checks."""
        self._history.append(
            _Transition(
                robot_xy=robot_xy.copy(),
                frontier_xy=frontier_xy.copy(),
                top_values=top_values,
            )
        )
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history :]

    def reset(self) -> None:
        """Clear all remembered transitions."""
        self._history.clear()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


class _Transition:
    """A single committed (robot → frontier) choice."""

    __slots__ = ("robot_xy", "frontier_xy", "top_values")

    def __init__(
        self,
        robot_xy: np.ndarray,
        frontier_xy: np.ndarray,
        top_values: tuple[float, ...],
    ) -> None:
        self.robot_xy = robot_xy
        self.frontier_xy = frontier_xy
        self.top_values = top_values
