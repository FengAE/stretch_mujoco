"""A* grid planner with path smoothing.

Uses the standard A* search on an 8-connected grid, then applies a
post-processing step that removes intermediate waypoints when the
straight line between non-consecutive waypoints is collision-free.
"""

from __future__ import annotations

import heapq
import math
from typing import Optional

import numpy as np

from stretch_mujoco.navigations.base import BasePlanner, NavigationPathError, OccupancyGrid


class AStarPlanner(BasePlanner):
    """A* search on an 8-connected occupancy grid with path smoothing."""

    # 8-connected neighbourhood: (dr, dc, cost)
    _MOVES = (
        (-1, 0, 1.0),
        (1, 0, 1.0),
        (0, -1, 1.0),
        (0, 1, 1.0),
        (-1, -1, math.sqrt(2.0)),
        (-1, 1, math.sqrt(2.0)),
        (1, -1, math.sqrt(2.0)),
        (1, 1, math.sqrt(2.0)),
    )

    def __init__(self, *, smoothing: bool = True):
        """
        Parameters
        ----------
        smoothing:
            If True, apply path shortening after A* completes.
        """
        self.smoothing = smoothing

    # ------------------------------------------------------------------
    def plan(
        self,
        grid: OccupancyGrid,
        start: np.ndarray,
        goal: np.ndarray,
    ) -> list[np.ndarray]:
        """Compute an A* path on *grid* from *start* to *goal*."""
        start_xy, start_cell = self._resolve_endpoint(grid, start, "Start")
        goal_xy, goal_cell = self._resolve_endpoint(grid, goal, "Goal")

        # A* search
        cell_path = self._a_star(grid, start_cell, goal_cell)

        # Optional smoothing
        if self.smoothing:
            cell_path = self._smooth(grid, cell_path)

        return self._world_waypoints(grid, cell_path, start_xy, goal_xy)

    # ------------------------------------------------------------------
    # A* search
    # ------------------------------------------------------------------
    def _a_star(
        self,
        grid: OccupancyGrid,
        start: tuple[int, int],
        goal: tuple[int, int],
    ) -> list[tuple[int, int]]:
        frontier: list[tuple[float, tuple[int, int]]] = [(0.0, start)]
        came_from: dict[tuple[int, int], Optional[tuple[int, int]]] = {start: None}
        cost_so_far: dict[tuple[int, int], float] = {start: 0.0}

        while frontier:
            _, current = heapq.heappop(frontier)
            if current == goal:
                break

            for dr, dc, move_cost in self._MOVES:
                neighbor = (current[0] + dr, current[1] + dc)
                if not grid.is_cell_free(neighbor):
                    continue

                # Corner-cut prevention: disallow diagonal moves that clip
                # the corner of an obstacle.
                if dr != 0 and dc != 0:
                    if not grid.is_cell_free((current[0] + dr, current[1])):
                        continue
                    if not grid.is_cell_free((current[0], current[1] + dc)):
                        continue

                new_cost = cost_so_far[current] + move_cost
                if new_cost >= cost_so_far.get(neighbor, math.inf):
                    continue
                cost_so_far[neighbor] = new_cost
                heuristic = math.dist(neighbor, goal)
                heapq.heappush(frontier, (new_cost + heuristic, neighbor))
                came_from[neighbor] = current

        if goal not in came_from:
            raise NavigationPathError("A* could not find a collision-free path to the target.")

        # Reconstruct path
        path: list[tuple[int, int]] = []
        cur: Optional[tuple[int, int]] = goal
        while cur is not None:
            path.append(cur)
            cur = came_from[cur]
        path.reverse()
        return path

    # ------------------------------------------------------------------
    # Path smoothing
    # ------------------------------------------------------------------
    def _smooth(
        self,
        grid: OccupancyGrid,
        path: list[tuple[int, int]],
    ) -> list[tuple[int, int]]:
        """Greedy shortcut: skip intermediate waypoints when LoS is clear.

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
                raise NavigationPathError("A* path contains a colliding grid segment.")
            result.append(path[candidate])
            idx = candidate
        return result
