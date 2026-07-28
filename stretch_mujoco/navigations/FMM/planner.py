"""Fast Marching Method (FMM) planner.

Solves the isotropic Eikonal equation |∇T| = 1 on a 2-D occupancy grid
using the standard first-order FMM scheme, then extracts a path via
gradient descent.
"""

from __future__ import annotations

import heapq
import math
from typing import Optional

import numpy as np

from stretch_mujoco.navigations.base import BasePlanner, NavigationPathError, OccupancyGrid


# Eikonal solver status tags
_FAR = 0  # unvisited
_FRONT = 1  # in the narrow band (heap)
_ACCEPTED = 2  # arrival time is final


class FMMPlanner(BasePlanner):
    """Fast Marching Method on a uniform 2-D grid.

    The planner solves the Eikonal equation from *start* outward, producing
    a time-of-arrival field.  A path is extracted by gradient descent from
    *goal* back to *start*.
    """

    # 8-connected neighbourhood for the Eikonal solver
    _NEIGHBOURS = (
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
        (-1, -1),
        (-1, 1),
        (1, -1),
        (1, 1),
    )

    def __init__(self, *, smoothing: bool = True):
        """
        Parameters
        ----------
        smoothing:
            If True, apply path shortening after gradient descent.
        """
        self.smoothing = smoothing

    # ------------------------------------------------------------------
    def plan(
        self,
        grid: OccupancyGrid,
        start: np.ndarray,
        goal: np.ndarray,
    ) -> list[np.ndarray]:
        """Compute an FMM path on *grid* from *start* to *goal*."""
        start_xy, start_cell = self._resolve_endpoint(grid, start, "Start")
        goal_xy, goal_cell = self._resolve_endpoint(grid, goal, "Goal")

        # Run FMM to build the arrival-time field
        times, status = self._fmm_solve(grid, start_cell, goal_cell)

        # Gradient-descent path extraction
        cell_path = self._extract_path(grid, times, start_cell, goal_cell)

        # Optional smoothing
        if self.smoothing:
            cell_path = self._smooth(grid, cell_path)

        return self._world_waypoints(grid, cell_path, start_xy, goal_xy)

    # ------------------------------------------------------------------
    # FMM Eikonal solver
    # ------------------------------------------------------------------
    def _fmm_solve(
        self,
        grid: OccupancyGrid,
        start: tuple[int, int],
        goal: tuple[int, int],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run FMM outward from *start* until *goal* is accepted.

        Returns
        -------
        times : np.ndarray (float, shape = grid.occupancy.shape)
            Arrival time for every cell.  Unreachable cells hold ``inf``.
        status : np.ndarray (int, shape = grid.occupancy.shape)
            Solver tag per cell (*_FAR* / *_FRONT* / *_ACCEPTED*).
        """
        rows, cols = grid.occupancy.shape
        times = np.full((rows, cols), np.inf, dtype=float)
        status = np.full((rows, cols), _FAR, dtype=np.uint8)

        # Initialise start
        sr, sc = start
        times[sr, sc] = 0.0
        status[sr, sc] = _ACCEPTED

        # Narrow-band heap: (time, row, col)
        heap: list[tuple[float, int, int]] = []

        # Seed the narrow band with neighbours of start
        for dr, dc in self._NEIGHBOURS:
            nr, nc = sr + dr, sc + dc
            if not grid.is_cell_free((nr, nc)):
                continue
            new_time = self._update_eikonal(grid, times, status, nr, nc)
            if np.isfinite(new_time):
                heapq.heappush(heap, (new_time, nr, nc))
                status[nr, nc] = _FRONT

        # Main loop
        while heap:
            t, r, c = heapq.heappop(heap)
            if status[r, c] == _ACCEPTED:
                continue  # stale entry
            status[r, c] = _ACCEPTED
            times[r, c] = t

            if (r, c) == goal:
                break

            # Update neighbours
            for dr, dc in self._NEIGHBOURS:
                nr, nc = r + dr, c + dc
                if not grid.is_cell_free((nr, nc)):
                    continue
                if status[nr, nc] == _ACCEPTED:
                    continue
                new_time = self._update_eikonal(grid, times, status, nr, nc)
                if not np.isfinite(new_time):
                    continue
                # In FMM we push the updated value regardless; stale entries
                # are skipped when they pop (see above).
                status[nr, nc] = _FRONT
                heapq.heappush(heap, (new_time, nr, nc))

        return times, status

    @staticmethod
    def _update_eikonal(
        grid: OccupancyGrid,
        times: np.ndarray,
        status: np.ndarray,
        r: int,
        c: int,
    ) -> float:
        """Compute the FMM arrival-time candidate for cell (r, c).

        Uses the first-order upwind discretisation of |∇T| = 1 on a
        uniform grid with unit spacing (cell scale).  At least one
        cardinal neighbour must already be accepted.
        """
        # Collect accepted cardinal-neighbour times (axis-aligned only
        # for the standard first-order scheme – diagonals are handled
        # implicitly via the two-axis minimum).
        neighbours: list[float] = []
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < times.shape[0] and 0 <= nc < times.shape[1]:
                if status[nr, nc] == _ACCEPTED:
                    neighbours.append(times[nr, nc])

        if not neighbours:
            return np.inf

        # Solve the quadratic for each axis pair (x, y) and take min.
        # For a single accepted cardinal neighbour, T = T_min + 1.
        candidates = [min(neighbours) + 1.0]

        # Pairwise horizontal + vertical
        h_vals: list[float] = []
        v_vals: list[float] = []
        for dr, dc in ((-1, 0), (1, 0)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < times.shape[0] and 0 <= nc < times.shape[1]:
                if status[nr, nc] == _ACCEPTED:
                    h_vals.append(times[nr, nc])
        for dr, dc in ((0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < times.shape[0] and 0 <= nc < times.shape[1]:
                if status[nr, nc] == _ACCEPTED:
                    v_vals.append(times[nr, nc])

        # Best horizontal and vertical
        best_h = min(h_vals) if h_vals else np.inf
        best_v = min(v_vals) if v_vals else np.inf

        if np.isfinite(best_h) and np.isfinite(best_v):
            t1, t2 = sorted([best_h, best_v])
            # Quadratic solution for |∇T| = 1 with two sides accepted:
            # (T - t1)² + (T - t2)² = 1  →  2T² - 2(t1+t2)T + t1²+t2²-1 = 0
            if t2 - t1 >= 1.0:
                candidates.append(t1 + 1.0)
            else:
                disc = 2.0 - (t2 - t1) ** 2
                if disc >= 0:
                    candidates.append((t1 + t2 + math.sqrt(disc)) / 2.0)

        return min(candidates)

    # ------------------------------------------------------------------
    # Gradient-descent path extraction
    # ------------------------------------------------------------------
    def _extract_path(
        self,
        grid: OccupancyGrid,
        times: np.ndarray,
        start: tuple[int, int],
        goal: tuple[int, int],
    ) -> list[tuple[int, int]]:
        """Follow the negative gradient of the arrival-time field from
        *goal* back to *start*."""
        path: list[tuple[int, int]] = []
        current = goal
        rows, cols = times.shape

        # Limit iterations to avoid infinite loops
        max_steps = rows * cols
        for _ in range(max_steps):
            path.append(current)
            if current == start:
                break

            r, c = current
            best_cell: Optional[tuple[int, int]] = None
            best_time = times[r, c]

            for dr, dc in self._NEIGHBOURS:
                nr, nc = r + dr, c + dc
                if not grid.is_cell_free((nr, nc)):
                    continue
                if not grid.line_of_sight(current, (nr, nc)):
                    continue
                if times[nr, nc] < best_time:
                    best_time = times[nr, nc]
                    best_cell = (nr, nc)

            if best_cell is None or best_time >= times[r, c]:
                raise NavigationPathError(
                    "Gradient descent stuck — no lower-time neighbour exists."
                )
            current = best_cell
        else:
            raise NavigationPathError("Gradient descent exceeded maximum iterations.")

        path.reverse()
        return path

    # ------------------------------------------------------------------
    # Path smoothing (shared logic with A*)
    # ------------------------------------------------------------------
    def _smooth(
        self,
        grid: OccupancyGrid,
        path: list[tuple[int, int]],
    ) -> list[tuple[int, int]]:
        """Greedy shortcut: skip waypoints when LoS is clear.

        Falls back to the raw path if a shortcut cannot be verified.
        """
        if len(path) <= 2:
            return path

        result: list[tuple[int, int]] = [path[0]]
        idx = 0
        while idx < len(path) - 1:
            candidate = len(path) - 1
            while candidate > idx and not grid.line_of_sight(path[idx], path[candidate]):
                candidate -= 1
            if candidate == idx:
                raise NavigationPathError("FMM path contains a colliding grid segment.")
            result.append(path[candidate])
            idx = candidate
        return result
