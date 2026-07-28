"""Local occupancy map and replaceable observation backends for FBE."""

from __future__ import annotations

import math
from typing import Iterable, Protocol

import numpy as np

from stretch_mujoco.navigations.base import OccupancyGrid


class LocalOccupancyGrid:
    """Incrementally observed 2-D occupancy map independent of its sensor.

    Cell values are ``-1`` for unknown, ``0`` for free, and ``1`` for an
    obstacle. Bounds and resolution are copied from a reference grid, but the
    reference occupancy values are never read by this class.
    """

    def __init__(self, reference_grid: OccupancyGrid):
        if not isinstance(reference_grid, OccupancyGrid):
            raise TypeError("reference_grid must be an OccupancyGrid")
        self.resolution = reference_grid.resolution
        self.agent_radius = reference_grid.agent_radius
        self.x_min, self.x_max = reference_grid.x_min, reference_grid.x_max
        self.y_min, self.y_max = reference_grid.y_min, reference_grid.y_max
        self.data = np.full(reference_grid.occupancy.shape, -1, dtype=np.int8)

    @property
    def shape(self) -> tuple[int, int]:
        return self.data.shape

    def world_to_cell(self, point: np.ndarray) -> tuple[int, int]:
        """Convert a world ``(x, y)`` coordinate to ``(row, col)``."""
        point_xy = np.asarray(point, dtype=float)
        col = int(round((float(point_xy[0]) - self.x_min) / self.resolution))
        row = int(round((float(point_xy[1]) - self.y_min) / self.resolution))
        return row, col

    def cell_to_world(self, cell: tuple[int, int]) -> np.ndarray:
        """Convert a ``(row, col)`` cell to world ``(x, y)``."""
        row, col = cell
        return np.array([self.x_min + col * self.resolution, self.y_min + row * self.resolution])

    def is_in_bounds(self, cell: tuple[int, int]) -> bool:
        row, col = cell
        return 0 <= row < self.data.shape[0] and 0 <= col < self.data.shape[1]

    def update_cells(
        self,
        *,
        free_cells: Iterable[tuple[int, int]] = (),
        obstacle_cells: Iterable[tuple[int, int]] = (),
    ) -> None:
        """Fuse observed cells into the map; obstacle observations win."""
        for cell in free_cells:
            if self.is_in_bounds(cell):
                self.data[cell] = 0
        for cell in obstacle_cells:
            if self.is_in_bounds(cell):
                self.data[cell] = 1

    def update_from_array(self, observation: np.ndarray) -> None:
        """Fuse an aligned ``{-1, 0, 1}`` occupancy observation.

        This is the adapter point for a SLAM or Nav2 occupancy grid after its
        frame, origin, resolution, and dimensions have been aligned.
        """
        values = np.asarray(observation)
        if values.shape != self.data.shape:
            raise ValueError(
                f"observation shape {values.shape} does not match local map {self.data.shape}"
            )
        if not np.all(np.isin(values, (-1, 0, 1))):
            raise ValueError("observation values must be -1, 0, or 1")
        known = values >= 0
        self.data[known] = values[known].astype(np.int8)

    def unknown_ratio(self) -> float:
        """Fraction of cells still unknown."""
        return float(np.count_nonzero(self.data == -1) / self.data.size)

    def explored_ratio(self) -> float:
        """Fraction of cells observed as free or occupied."""
        return 1.0 - self.unknown_ratio()

    def known_obstacles(self) -> np.ndarray:
        return self.data == 1

    def known_free(self) -> np.ndarray:
        return self.data == 0

    def unknown_mask(self) -> np.ndarray:
        return self.data == -1


class MapObservationSource(Protocol):
    """Sensor/map backend that can update an FBE local occupancy grid."""

    def update(
        self,
        local_map: LocalOccupancyGrid,
        robot_xy: np.ndarray,
        robot_yaw: float,
    ) -> None:
        """Fuse one observation at the supplied robot pose."""


class SimulatedLaserObservationSource:
    """Idealised 2-D laser backed by a simulation ground-truth grid."""

    def __init__(
        self,
        truth_grid: OccupancyGrid,
        *,
        num_rays: int,
        max_range_cells: int,
        fov_degrees: float,
    ) -> None:
        self.truth_grid = truth_grid
        self.num_rays = num_rays
        self.max_range_cells = max_range_cells
        self.fov_degrees = fov_degrees

    def update(
        self,
        local_map: LocalOccupancyGrid,
        robot_xy: np.ndarray,
        robot_yaw: float,
    ) -> None:
        start = local_map.world_to_cell(robot_xy)
        if not local_map.is_in_bounds(start):
            return

        local_map.update_cells(free_cells=(start,))
        half_fov = math.radians(self.fov_degrees) / 2.0
        angles = robot_yaw + np.linspace(-half_fov, half_fov, self.num_rays)
        for angle in angles:
            end = self._ray_end_cell(start, float(angle), self.max_range_cells)
            self._trace_ray(local_map, start, end)

    @staticmethod
    def _ray_end_cell(start: tuple[int, int], angle_rad: float, max_cells: int) -> tuple[int, int]:
        delta_row = math.sin(angle_rad)
        delta_col = math.cos(angle_rad)
        return (
            int(round(start[0] + delta_row * max_cells)),
            int(round(start[1] + delta_col * max_cells)),
        )

    def _trace_ray(
        self,
        local_map: LocalOccupancyGrid,
        start: tuple[int, int],
        end: tuple[int, int],
    ) -> None:
        row, col = start
        end_row, end_col = end
        delta_row = abs(end_row - row)
        delta_col = abs(end_col - col)
        sign_row = 1 if end_row > row else -1
        sign_col = 1 if end_col > col else -1
        error = delta_row - delta_col

        while True:
            if (row, col) != start:
                cell = (row, col)
                if not local_map.is_in_bounds(cell):
                    return
                if self.truth_grid.occupancy[cell]:
                    local_map.update_cells(obstacle_cells=(cell,))
                    return
                local_map.update_cells(free_cells=(cell,))

            if row == end_row and col == end_col:
                return
            doubled_error = 2 * error
            if doubled_error > -delta_col:
                error -= delta_col
                row += sign_row
            if doubled_error < delta_row:
                error += delta_row
                col += sign_col
