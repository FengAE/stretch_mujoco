"""Frontier-Based Exploration (FBE) planner.

Maintains a local occupancy grid updated by simulated laser scans, detects
frontiers (boundaries between free and unknown space), clusters them, and
navigates the robot toward the best frontier using A*.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

from stretch_mujoco.navigations._loader import _load_module
from stretch_mujoco.navigations.base import NavigationPathError, OccupancyGrid

_astar_mod = _load_module("A*", "planner")
AStarPlanner = _astar_mod.AStarPlanner
from stretch_mujoco.navigations.FBE.frontier import (
    FrontierCluster,
    best_frontier,
    cluster_frontiers,
    detect_frontier_cells,
)
from stretch_mujoco.navigations.FBE.local_map import (
    LocalOccupancyGrid,
    MapObservationSource,
    SimulatedLaserObservationSource,
)


class FBEState(Enum):
    """High-level state of the FBE state machine."""

    SCANNING = "scanning"  # updating map
    PLANNING = "planning"  # choosing next frontier
    MOVING = "moving"  # following a path
    ROTATING = "rotating"  # turning to face target
    FINISHED = "finished"  # no frontiers left


@dataclass
class FBEDiagnostics:
    """Per-step diagnostics exposed for visualisation / logging."""

    state: FBEState = FBEState.SCANNING
    explored_ratio: float = 0.0
    frontier_count: int = 0
    cluster_count: int = 0
    target_frontier: Optional[FrontierCluster] = None
    current_path: list[np.ndarray] = field(default_factory=list)
    path_progress: int = 0  # index into current_path
    finish_reason: str = ""


class FBEPlanner:
    """Frontier-Based Exploration using A* for path planning.

    Usage sketch::

        god_grid = OccupancyGrid.from_model(model, data, ...)
        fbe = FBEPlanner(god_grid, robot_xy=(0., 0.), robot_yaw=0.)

        while not fbe.is_finished():
            fbe.step(robot_xy, robot_yaw)
            if fbe.has_path():
                cmd = fbe.pop_command()
                # drive robot toward cmd ...
    """

    def __init__(
        self,
        god_grid: OccupancyGrid,
        *,
        robot_xy: tuple[float, float] = (0.0, 0.0),
        robot_yaw: float = 0.0,
        num_rays: int = 180,
        max_range_m: float = 5.0,
        fov_degrees: float = 270.0,
        min_cluster_size: int = 3,
        min_frontier_dist: float = 0.5,
        explore_threshold: float = 0.95,
        max_stagnant_scans: int = 30,
        astar: Optional[AStarPlanner] = None,
        observation_source: Optional[MapObservationSource] = None,
    ) -> None:
        """
        Parameters
        ----------
        god_grid:
            Ground-truth occupancy grid (for simulated ray-casting).
        robot_xy:
            Initial robot world position.
        robot_yaw:
            Initial robot yaw angle (radians).
        num_rays:
            Rays per scan.
        max_range_m:
            Maximum laser range in metres.
        fov_degrees:
            Sensor field-of-view in degrees.
        min_cluster_size:
            Minimum frontier cluster size (cells).
        min_frontier_dist:
            Minimum distance (m) from robot to frontier target.
        explore_threshold:
            Stop when explored ratio reaches this value.
        max_stagnant_scans:
            Finish when this many consecutive scans add no map information
            and no reachable frontier remains.
        astar:
            Optional pre-configured A* planner (created if None).
        observation_source:
            Replaceable map/sensor backend. When omitted, an idealised
            simulation laser backed by ``god_grid`` is used.
        """
        if not isinstance(god_grid, OccupancyGrid):
            raise TypeError("god_grid must be an OccupancyGrid")
        if not isinstance(num_rays, int) or isinstance(num_rays, bool) or num_rays <= 0:
            raise ValueError("num_rays must be a positive integer")
        if not np.isfinite(max_range_m) or max_range_m <= 0:
            raise ValueError("max_range_m must be finite and > 0")
        if not np.isfinite(fov_degrees) or not 0 < fov_degrees <= 360:
            raise ValueError("fov_degrees must be in (0, 360]")
        if (
            not isinstance(min_cluster_size, int)
            or isinstance(min_cluster_size, bool)
            or min_cluster_size <= 0
        ):
            raise ValueError("min_cluster_size must be a positive integer")
        if not np.isfinite(min_frontier_dist) or min_frontier_dist < 0:
            raise ValueError("min_frontier_dist must be finite and >= 0")
        if not np.isfinite(explore_threshold) or not 0 <= explore_threshold <= 1:
            raise ValueError("explore_threshold must be in [0, 1]")
        if (
            not isinstance(max_stagnant_scans, int)
            or isinstance(max_stagnant_scans, bool)
            or max_stagnant_scans <= 0
        ):
            raise ValueError("max_stagnant_scans must be a positive integer")

        initial_xy = np.asarray(robot_xy, dtype=float).reshape(-1)
        if initial_xy.size < 2 or not np.all(np.isfinite(initial_xy[:2])):
            raise ValueError("robot_xy must contain two finite coordinates")
        if not np.isfinite(robot_yaw):
            raise ValueError("robot_yaw must be finite")
        if observation_source is not None and not callable(
            getattr(observation_source, "update", None)
        ):
            raise TypeError("observation_source must provide an update() method")

        # This grid supplies map geometry for every backend. Its occupancy is
        # consulted only by the default simulated observation source below.
        self.reference_grid = god_grid
        self.local_map = LocalOccupancyGrid(god_grid)
        self.num_rays = num_rays
        self.max_range_cells = max(1, int(math.ceil(max_range_m / god_grid.resolution)))
        self.fov_degrees = fov_degrees
        self.min_cluster_size = min_cluster_size
        self.min_frontier_dist = min_frontier_dist
        self.explore_threshold = explore_threshold
        self.max_stagnant_scans = max_stagnant_scans
        self._astar = astar or AStarPlanner(smoothing=True)
        self.observation_source = observation_source or SimulatedLaserObservationSource(
            god_grid,
            num_rays=num_rays,
            max_range_cells=self.max_range_cells,
            fov_degrees=fov_degrees,
        )

        # State
        self.state = FBEState.SCANNING
        self._robot_xy = initial_xy[:2].copy()
        self._robot_yaw = float(robot_yaw)
        self._target_cluster: Optional[FrontierCluster] = None
        self._path: list[np.ndarray] = []
        self._path_idx: int = 0
        self._scan_cooldown: int = 0
        self._blocked_frontier_cells: set[tuple[int, int]] = set()
        self._stagnant_scans = 0

        # Stats
        self.diag = FBEDiagnostics()
        self._step_count = 0
        self._exploration_log: list[dict] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def is_finished(self) -> bool:
        return self.state == FBEState.FINISHED

    def has_path(self) -> bool:
        return len(self._path) > 0 and self._path_idx < len(self._path)

    def current_target(self) -> Optional[np.ndarray]:
        """Current navigation sub-goal (waypoint), or None."""
        if self.has_path():
            return self._path[self._path_idx]
        return None

    def pop_command(self) -> Optional[np.ndarray]:
        """Return the active waypoint without advancing before it is reached.

        Path progress is driven by robot pose feedback in :meth:`step`, not
        by command publication.  Keeping this method as an alias preserves
        the original public API without allowing callers to skip waypoints.
        """
        return self.current_target()

    def step(
        self,
        robot_xy: tuple[float, float] | np.ndarray,
        robot_yaw: float,
    ) -> None:
        """Run one exploration step.

        Call this at ~10-20 Hz with the robot's current pose.
        """
        self._step_count += 1
        self._robot_xy = np.asarray(robot_xy, dtype=float)
        self._robot_yaw = float(robot_yaw)

        # --- always scan ---
        self._do_scan()

        # --- state machine ---
        if self.state == FBEState.SCANNING:
            self._do_detect()

        elif self.state == FBEState.PLANNING:
            self._do_plan()

        elif self.state == FBEState.MOVING:
            self._do_move()

        elif self.state == FBEState.ROTATING:
            self._do_rotate()

        # Advancing a waypoint with a non-zero tolerance can make the segment
        # from the robot's actual pose to the next waypoint cut a corner.
        # Validate once more before exposing the next command to the caller.
        if self.state == FBEState.MOVING and not self._remaining_path_is_valid():
            self._abandon_active_path()

        # Update diagnostics
        self.diag.state = self.state
        self.diag.explored_ratio = self.local_map.explored_ratio()
        self.diag.current_path = list(self._path[self._path_idx :]) if self._path else []
        self.diag.path_progress = self._path_idx

    # ------------------------------------------------------------------
    # State handlers
    # ------------------------------------------------------------------
    def _do_scan(self) -> None:
        """Cast rays and update the local map."""
        previous = self.local_map.data.copy()
        self.observation_source.update(self.local_map, self._robot_xy, self._robot_yaw)
        if not np.array_equal(previous, self.local_map.data):
            self._stagnant_scans = 0
            # A changed map can open a route to a previously unreachable
            # frontier, so failed-frontier decisions are no longer current.
            self._blocked_frontier_cells.clear()
            if self.state == FBEState.MOVING and not self._remaining_path_is_valid():
                # A newly observed obstacle invalidated the active route.
                # Stop publishing it before the external controller moves.
                self._abandon_active_path()
        else:
            self._stagnant_scans += 1

    def _do_detect(self) -> None:
        """Detect frontiers and decide what to do next."""
        f_mask = detect_frontier_cells(self.local_map)
        clusters = cluster_frontiers(self.local_map, f_mask, self.min_cluster_size)

        self.diag.frontier_count = int(f_mask.sum())
        self.diag.cluster_count = len(clusters)

        # Check termination
        if self.local_map.explored_ratio() >= self.explore_threshold and not clusters:
            self._finish("exploration threshold reached and no frontiers remain")
            return

        # Pick the best frontier that has not already failed on this map.
        candidates = self._reachable_candidates(clusters)
        cluster = best_frontier(candidates, self._robot_xy, min_dist=self.min_frontier_dist)
        if cluster is None:
            # Try again with smaller clusters and no distance filter
            clusters2 = cluster_frontiers(self.local_map, f_mask, min_cluster_size=1)
            clusters2 = self._reachable_candidates(clusters2)
            cluster = best_frontier(clusters2, self._robot_xy, min_dist=0.0)
        if cluster is None:
            if self._stagnant_scans >= self.max_stagnant_scans:
                if self._blocked_frontier_cells:
                    reason = "no reachable frontiers and the map stopped changing"
                else:
                    reason = "no frontiers and the map stopped changing"
                self._finish(reason)
            else:
                self.state = FBEState.SCANNING
            return

        self._target_cluster = cluster
        self.diag.target_frontier = cluster
        self.state = FBEState.PLANNING

    def _do_plan(self) -> None:
        """Plan a path to the selected frontier centroid."""
        if self._target_cluster is None:
            self.state = FBEState.SCANNING
            return

        # Build a temporary occupancy grid from the local map for planning.
        # Only observed free cells are traversable.
        local_occ = self._local_to_occupancy()
        target_xy = self.local_map.cell_to_world(self._target_cluster.centroid_cell)

        try:
            path = self._astar.plan(
                local_occ,
                self._robot_xy,
                target_xy,
            )
        except NavigationPathError:
            self._blacklist_target()
            self._target_cluster = None
            self._path = []
            self.state = FBEState.SCANNING
            return

        if len(path) < 2:
            self._blacklist_target()
            self._target_cluster = None
            self._path = []
            self.state = FBEState.SCANNING
            return

        self._path = path
        self._path_idx = 1  # skip start
        self.state = FBEState.MOVING

    def _do_move(self) -> None:
        """Check whether we've reached the next waypoint or the goal."""
        if not self.has_path():
            # Reached end of path
            target_xy = (
                self.local_map.cell_to_world(self._target_cluster.centroid_cell)
                if self._target_cluster
                else self._robot_xy
            )
            dist = float(np.linalg.norm(target_xy - self._robot_xy))
            if self._target_cluster and dist < self.min_frontier_dist:
                self._path = []
                self._target_cluster = None
                self.state = FBEState.SCANNING
            else:
                self.state = FBEState.PLANNING
            return

        wp = self._path[self._path_idx]
        dist = float(np.linalg.norm(wp - self._robot_xy))
        if dist < 0.2:
            self._path_idx += 1

    def _do_rotate(self) -> None:
        """Rotate toward the target (handled by external controller)."""
        self.state = FBEState.MOVING

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _reachable_candidates(self, clusters: list[FrontierCluster]) -> list[FrontierCluster]:
        """Remove clusters already found unreachable on the current map."""
        return [
            cluster
            for cluster in clusters
            if not self._blocked_frontier_cells.intersection(cluster.cells)
        ]

    def _finish(self, reason: str) -> None:
        """Enter the terminal state with a diagnostic explanation."""
        self._path = []
        self._path_idx = 0
        self._target_cluster = None
        self.diag.target_frontier = None
        self.diag.finish_reason = reason
        self.state = FBEState.FINISHED

    def _blacklist_target(self) -> None:
        """Remember every cell in the current failed frontier cluster."""
        if self._target_cluster is not None:
            self._blocked_frontier_cells.update(self._target_cluster.cells)

    def _abandon_active_path(self) -> None:
        """Stop an unsafe route and retry exploration with another frontier."""
        self._blacklist_target()
        self._path = []
        self._path_idx = 0
        self._target_cluster = None
        self.diag.target_frontier = None
        self.state = FBEState.SCANNING

    def _remaining_path_is_valid(self) -> bool:
        """Check the unexecuted route against the latest local map."""
        if not self.has_path():
            return True
        planning_grid = self._local_to_occupancy()
        points = [self._robot_xy, *self._path[self._path_idx :]]
        for start, end in zip(points, points[1:]):
            start_cell = planning_grid.world_to_cell(start)
            end_cell = planning_grid.world_to_cell(end)
            if not planning_grid.line_of_sight(start_cell, end_cell):
                return False
        return True

    def _local_to_occupancy(self) -> OccupancyGrid:
        """Build an ``OccupancyGrid`` from the local map.

        Only known-free cells (0) are traversable. Unknown cells (-1) and
        known obstacles (1) block the path.
        """
        occ = self.local_map.data != 0
        return OccupancyGrid(
            bounds=(
                self.local_map.x_min,
                self.local_map.x_max,
                self.local_map.y_min,
                self.local_map.y_max,
            ),
            occupancy=occ,
            resolution=self.local_map.resolution,
            agent_radius=self.local_map.agent_radius,
            obstacles=(),  # no precomputed obstacle AABBs (we're using grid only)
        )
