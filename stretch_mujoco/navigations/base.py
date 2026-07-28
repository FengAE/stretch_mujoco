"""Base classes for navigation planners.

Defines the abstract planner interface and the occupancy grid representation
shared by all navigation algorithms.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import math
from typing import Optional

import mujoco
import numpy as np


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class NavigationPathError(RuntimeError):
    """Raised when a navigation target cannot be reached."""


# ---------------------------------------------------------------------------
# Occupancy grid
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ObstacleFootprint:
    """Axis-aligned bounding box of an inflated obstacle in world coordinates."""

    geom_name: str
    minimum: tuple[float, float]  # (x_min, y_min)
    maximum: tuple[float, float]  # (x_max, y_max)


class OccupancyGrid:
    """2-D occupancy grid rasterised from MuJoCo collision geometry.

    The grid is row-major: ``occupancy[row, col]`` where *row* indexes the
    y-axis and *col* indexes the x-axis.  ``True`` means occupied (blocked).
    """

    def __init__(
        self,
        bounds: tuple[float, float, float, float],
        occupancy: np.ndarray,
        *,
        resolution: float,
        agent_radius: float,
        obstacles: tuple[ObstacleFootprint, ...],
    ) -> None:
        self.x_min, self.x_max, self.y_min, self.y_max = bounds
        self.occupancy = occupancy  # bool, shape (rows, cols)
        self.resolution = resolution
        self.agent_radius = agent_radius
        self.obstacles = obstacles

    # -- properties ----------------------------------------------------------

    @property
    def width(self) -> int:
        """Number of columns (x-direction)."""
        return self.occupancy.shape[1]

    @property
    def height(self) -> int:
        """Number of rows (y-direction)."""
        return self.occupancy.shape[0]

    # -- factory -------------------------------------------------------------

    @classmethod
    def from_model(
        cls,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        *,
        resolution: float = 0.08,
        agent_radius: float = 0.25,
        bounds: tuple[float, float, float, float] | None = None,
        floor_geom_name: str = "office_floor",
        minimum_obstacle_height: float = 0.08,
        maximum_obstacle_height: float = 1.80,
        require_collision: bool = True,
        exclude_prefixes: tuple[str, ...] = (),
    ) -> "OccupancyGrid":
        """Build an occupancy grid from a MuJoCo model + data pair.

        Parameters
        ----------
        resolution:
            World metres per grid cell.
        agent_radius:
            Extra inflation added around every obstacle.
        bounds:
            Explicit ``(x_min, x_max, y_min, y_max)`` walkable region.
            When *None* the bounds are inferred from the named floor box
            geom.  Required for scenes whose floor is a plane.
        floor_geom_name:
            Name of the box geom that defines the walkable floor bounds
            (ignored when *bounds* is provided).
        minimum_obstacle_height:
            Geoms whose top is below this height (in world Z) are ignored.
        maximum_obstacle_height:
            Geoms whose bottom is above this height are ignored.
        require_collision:
            When True (default), only geoms with collision enabled are
            treated as obstacles.  Set to False for visual-only scenes
            (e.g. Habitat imports) where geoms have ``contype=0``.
        exclude_prefixes:
            Geom names starting with any of these prefixes are skipped.
            Useful for excluding stage/shell geometry from Habitat scenes
            (e.g. ``exclude_prefixes=(\"habitat_stage_\",)``).
        """
        if resolution <= 0 or agent_radius < 0:
            raise ValueError("resolution must be > 0 and agent_radius >= 0")

        if bounds is not None:
            x_min, x_max, y_min, y_max = bounds
        else:
            x_min, x_max, y_min, y_max = cls._walkable_bounds(
                model, data, agent_radius, floor_geom_name
            )

        width = int(math.floor((x_max - x_min) / resolution)) + 1
        height = int(math.floor((y_max - y_min) / resolution)) + 1
        occupancy = np.zeros((height, width), dtype=bool)
        obstacles: list[ObstacleFootprint] = []

        for geom_id in range(model.ngeom):
            geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
            if geom_name == floor_geom_name:
                continue
            if exclude_prefixes and geom_name.startswith(exclude_prefixes):
                continue

            body_id = int(model.geom_bodyid[geom_id])
            # Skip mocap bodies (animated humanoids, etc.)
            if body_id >= 0 and model.body_mocapid[body_id] >= 0:
                continue
            # Skip non-colliding geoms (unless we're told to include all visual
            # geoms, e.g. for Habitat scenes that lack collision attributes).
            if require_collision and (
                model.geom_contype[geom_id] == 0 and model.geom_conaffinity[geom_id] == 0
            ):
                continue

            half_extents = cls._world_half_extents(model, data, geom_id)
            center = data.geom_xpos[geom_id]
            z_min = float(center[2] - half_extents[2])
            z_max = float(center[2] + half_extents[2])

            if z_max <= minimum_obstacle_height or z_min >= maximum_obstacle_height:
                continue

            minimum = center[:2] - half_extents[:2] - agent_radius
            maximum = center[:2] + half_extents[:2] + agent_radius
            obstacles.append(
                ObstacleFootprint(
                    geom_name,
                    (float(minimum[0]), float(minimum[1])),
                    (float(maximum[0]), float(maximum[1])),
                )
            )

            col_min = max(0, int(math.ceil((minimum[0] - x_min) / resolution)))
            col_max = min(width - 1, int(math.floor((maximum[0] - x_min) / resolution)))
            row_min = max(0, int(math.ceil((minimum[1] - y_min) / resolution)))
            row_max = min(height - 1, int(math.floor((maximum[1] - y_min) / resolution)))

            if col_min <= col_max and row_min <= row_max:
                occupancy[row_min : row_max + 1, col_min : col_max + 1] = True

        return cls(
            (x_min, x_max, y_min, y_max),
            occupancy,
            resolution=resolution,
            agent_radius=agent_radius,
            obstacles=tuple(obstacles),
        )

    # -- coordinate conversion -----------------------------------------------

    def world_to_cell(self, point: np.ndarray) -> tuple[int, int]:
        """Convert a world (x, y) coordinate to a (row, col) cell index."""
        col = int(round((float(point[0]) - self.x_min) / self.resolution))
        row = int(round((float(point[1]) - self.y_min) / self.resolution))
        return row, col

    def cell_to_world(self, cell: tuple[int, int]) -> np.ndarray:
        """Convert a (row, col) cell index back to world (x, y)."""
        row, col = cell
        return np.array([self.x_min + col * self.resolution, self.y_min + row * self.resolution])

    # -- cell queries --------------------------------------------------------

    def is_cell_free(self, cell: tuple[int, int]) -> bool:
        """Return True if *cell* is inside bounds and unoccupied."""
        row, col = cell
        return (
            0 <= row < self.occupancy.shape[0]
            and 0 <= col < self.occupancy.shape[1]
            and not self.occupancy[row, col]
        )

    def is_world_free(self, point: np.ndarray) -> bool:
        """Return True if the world point's cell is free."""
        return self.is_cell_free(self.world_to_cell(np.asarray(point, dtype=float)))

    def is_world_inside_bounds(self, point: np.ndarray) -> bool:
        """Return True when a world point lies within the walkable bounds."""
        x, y = (float(v) for v in np.asarray(point)[:2])
        return self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max

    def point_inside_obstacle(self, point: np.ndarray) -> bool:
        """Check whether a world point falls inside any inflated obstacle AABB."""
        x, y = (float(v) for v in np.asarray(point)[:2])
        eps = 1e-9
        return any(
            obs.minimum[0] + eps < x < obs.maximum[0] - eps
            and obs.minimum[1] + eps < y < obs.maximum[1] - eps
            for obs in self.obstacles
        )

    def nearest_free_cell(
        self,
        origin: tuple[int, int],
        toward: np.ndarray,
    ) -> tuple[int, int]:
        """If *origin* is occupied, return the nearest free cell (toward *toward*)."""
        if self.is_cell_free(origin):
            return origin
        max_radius = max(self.occupancy.shape)
        for radius in range(1, max_radius):
            candidates: list[tuple[int, int]] = []
            for row in range(origin[0] - radius, origin[0] + radius + 1):
                for col in range(origin[1] - radius, origin[1] + radius + 1):
                    if max(abs(row - origin[0]), abs(col - origin[1])) != radius:
                        continue
                    cell = (row, col)
                    if self.is_cell_free(cell):
                        candidates.append(cell)
            if candidates:
                return min(
                    candidates,
                    key=lambda c: float(np.linalg.norm(self.cell_to_world(c) - toward)),
                )
        raise NavigationPathError("No free navigation cell exists near the queried position")

    # -- line-of-sight -------------------------------------------------------

    def line_of_sight(
        self,
        start_cell: tuple[int, int],
        end_cell: tuple[int, int],
    ) -> bool:
        """Return True if the straight line between two cells is collision-free."""
        if any(
            not self.is_cell_free(cell) for cell in self._supercover_cells(start_cell, end_cell)
        ):
            return False
        start_world = self.cell_to_world(start_cell)
        end_world = self.cell_to_world(end_cell)
        return self.world_line_of_sight(start_world, end_world)

    def world_line_of_sight(self, start: np.ndarray, end: np.ndarray) -> bool:
        """Return True if a world-space segment stays in bounds and avoids obstacles."""
        start_world = np.asarray(start, dtype=float)[:2]
        end_world = np.asarray(end, dtype=float)[:2]
        if not self.is_world_inside_bounds(start_world) or not self.is_world_inside_bounds(
            end_world
        ):
            return False
        for obs in self.obstacles:
            if self._segment_intersects_box(
                start_world,
                end_world,
                np.array(obs.minimum),
                np.array(obs.maximum),
            ):
                return False
        return True

    @staticmethod
    def _supercover_cells(
        start: tuple[int, int], end: tuple[int, int]
    ) -> tuple[tuple[int, int], ...]:
        """Return every grid cell touched by a segment between two cell centres."""
        row, col = start
        end_row, end_col = end
        delta_row = end_row - row
        delta_col = end_col - col
        steps_row = abs(delta_row)
        steps_col = abs(delta_col)
        sign_row = 0 if delta_row == 0 else (1 if delta_row > 0 else -1)
        sign_col = 0 if delta_col == 0 else (1 if delta_col > 0 else -1)

        cells = [(row, col)]
        row_steps = 0
        col_steps = 0
        while row_steps < steps_row or col_steps < steps_col:
            row_progress = (1 + 2 * row_steps) * steps_col
            col_progress = (1 + 2 * col_steps) * steps_row
            if row_progress == col_progress:
                # At a grid corner, both side cells must be free as well as
                # the diagonal destination. This prevents corner cutting.
                if sign_row and sign_col:
                    cells.append((row + sign_row, col))
                    cells.append((row, col + sign_col))
                row += sign_row
                col += sign_col
                row_steps += 1
                col_steps += 1
            elif row_progress < col_progress:
                row += sign_row
                row_steps += 1
            else:
                col += sign_col
                col_steps += 1
            cells.append((row, col))
        return tuple(cells)

    # -- internal helpers ----------------------------------------------------

    @staticmethod
    def _walkable_bounds(
        model: mujoco.MjModel,
        data: mujoco.MjData,
        margin: float,
        floor_geom_name: str,
    ) -> tuple[float, float, float, float]:
        floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, floor_geom_name)
        if floor_id < 0:
            # Fallback: use the first box geom or a plane
            raise NavigationPathError(
                f"Navigation requires a box geom named '{floor_geom_name}' "
                f"to define walkable bounds."
            )
        if model.geom_type[floor_id] != mujoco.mjtGeom.mjGEOM_BOX:
            raise NavigationPathError(
                f"Floor geom '{floor_geom_name}' must be a box, "
                f"got type {model.geom_type[floor_id]}"
            )
        center = data.geom_xpos[floor_id]
        size = model.geom_size[floor_id]
        return (
            float(center[0] - size[0] + margin),
            float(center[0] + size[0] - margin),
            float(center[1] - size[1] + margin),
            float(center[1] + size[1] - margin),
        )

    @staticmethod
    def _world_half_extents(
        model: mujoco.MjModel,
        data: mujoco.MjData,
        geom_id: int,
    ) -> np.ndarray:
        """Conservative world-aligned half-extents of a geom."""
        geom_type = model.geom_type[geom_id]
        size = model.geom_size[geom_id]
        rotation = data.geom_xmat[geom_id].reshape(3, 3)

        if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
            return np.abs(rotation) @ size
        if geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
            return np.full(3, size[0])
        if geom_type == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
            return np.sqrt((rotation * size[np.newaxis, :]) ** 2 @ np.ones(3))
        if geom_type in {mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_CAPSULE}:
            radius = float(size[0])
            half_length = float(size[1])
            axis = np.abs(rotation[:, 2])
            if geom_type == mujoco.mjtGeom.mjGEOM_CAPSULE:
                return np.full(3, radius) + half_length * axis
            radial = radius * np.sqrt(np.maximum(0.0, 1.0 - axis**2))
            return radial + half_length * axis
        if geom_type == mujoco.mjtGeom.mjGEOM_MESH:
            # For meshes, geom_size holds the AABB half-extents.
            # Some meshes embed sentinel values (0 or 1e10) — skip those
            # by returning a zero vector so they are filtered out.
            if np.any(size > 1e6) or np.any(size < 0):
                return np.zeros(3)
            clamped = np.clip(size, 0.0, 50.0)
            return np.abs(rotation) @ clamped
        # Fallback: use the bounding radius
        return np.full(3, float(model.geom_rbound[geom_id]))

    @staticmethod
    def _segment_intersects_box(
        start: np.ndarray,
        end: np.ndarray,
        bmin: np.ndarray,
        bmax: np.ndarray,
    ) -> bool:
        """Slab-based test for intersection with the strict AABB interior."""
        # Touching an already-inflated obstacle is valid: it represents
        # exactly the requested agent clearance rather than penetration.
        eps = 1e-9
        bmin = bmin + eps
        bmax = bmax - eps
        if np.any(bmin >= bmax):
            return False
        direction = end - start
        lower = 0.0
        upper = 1.0
        for axis in range(2):
            if abs(direction[axis]) < 1e-12:
                if start[axis] < bmin[axis] or start[axis] > bmax[axis]:
                    return False
                continue
            first = (bmin[axis] - start[axis]) / direction[axis]
            second = (bmax[axis] - start[axis]) / direction[axis]
            entry, exit_ = sorted((first, second))
            lower = max(lower, entry)
            upper = min(upper, exit_)
            if lower > upper:
                return False
        return upper >= 0.0 and lower <= 1.0


# ---------------------------------------------------------------------------
# Abstract planner
# ---------------------------------------------------------------------------


class BasePlanner(ABC):
    """Abstract interface for a grid-based path planner."""

    @staticmethod
    def _resolve_endpoint(
        grid: OccupancyGrid, point: np.ndarray, label: str
    ) -> tuple[np.ndarray, tuple[int, int]]:
        """Validate a requested endpoint and resolve it to a free grid cell."""
        values = np.asarray(point, dtype=float).reshape(-1)
        if values.size < 2 or not np.all(np.isfinite(values[:2])):
            raise ValueError(f"{label} must contain two finite coordinates")
        point_xy = values[:2].copy()
        if not grid.is_world_inside_bounds(point_xy):
            raise NavigationPathError(
                f"{label} {point_xy.tolist()} lies outside the navigation bounds."
            )
        if grid.point_inside_obstacle(point_xy):
            raise NavigationPathError(
                f"{label} {point_xy.tolist()} lies inside an inflated obstacle."
            )

        cell = grid.world_to_cell(point_xy)
        if not grid.is_cell_free(cell):
            cell = grid.nearest_free_cell(cell, point_xy)
            point_xy = grid.cell_to_world(cell)
        return point_xy, cell

    @staticmethod
    def _world_waypoints(
        grid: OccupancyGrid,
        cell_path: list[tuple[int, int]],
        start: np.ndarray,
        goal: np.ndarray,
    ) -> list[np.ndarray]:
        """Convert a cell path to validated, deduplicated world waypoints."""
        waypoints = [start]
        waypoints.extend(grid.cell_to_world(cell) for cell in cell_path[1:-1])
        waypoints.append(goal)

        deduped: list[np.ndarray] = []
        for waypoint in waypoints:
            if not deduped or np.linalg.norm(waypoint - deduped[-1]) > 1e-6:
                deduped.append(waypoint)

        for first, second in zip(deduped, deduped[1:]):
            if not grid.world_line_of_sight(first, second):
                raise NavigationPathError(
                    "Resolved path contains a segment that intersects an inflated obstacle."
                )
        return deduped

    @abstractmethod
    def plan(
        self,
        grid: OccupancyGrid,
        start: np.ndarray,
        goal: np.ndarray,
    ) -> list[np.ndarray]:
        """Compute a path from *start* to *goal*.

        Parameters
        ----------
        grid:
            The occupancy grid describing free / occupied space.
        start:
            World (x, y) start position.
        goal:
            World (x, y) goal position.

        Returns
        -------
        list[np.ndarray]
            Ordered list of 2-D waypoints from start to goal (inclusive).
        """
        ...
