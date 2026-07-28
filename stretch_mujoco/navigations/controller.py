"""Unified navigation controller.

Connects to a MuJoCo scene, builds an occupancy grid, and delegates path
planning to a selected algorithm (A*, FMM, or future plug-ins).
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional, Type

import mujoco
import numpy as np

from stretch_mujoco.navigations.base import (
    BasePlanner,
    NavigationPathError,
    OccupancyGrid,
)
from stretch_mujoco.navigations._loader import _load_module


class Algorithm(str, Enum):
    """Supported navigation algorithms."""

    ASTAR = "astar"
    FMM = "fmm"
    FBE = "fbe"
    VLFM = "vlfm"


_EXPLORATION_ALGORITHMS = {Algorithm.FBE, Algorithm.VLFM}


def _get_planner_registry() -> dict[Algorithm, type]:
    """Lazily build the planner registry on first access.

    This function exists so that the A* planner (whose directory name
    ``A*`` is not a valid Python identifier) is loaded via ``importlib``
    rather than a standard ``import`` statement.
    """
    _astar = _load_module("A*", "planner")
    _fmm = _load_module("FMM", "planner")
    from stretch_mujoco.navigations.FBE.planner import FBEPlanner
    from stretch_mujoco.navigations.VLFM.planner import VLFMPlanner

    return {
        Algorithm.ASTAR: _astar.AStarPlanner,
        Algorithm.FMM: _fmm.FMMPlanner,
        Algorithm.FBE: FBEPlanner,
        Algorithm.VLFM: VLFMPlanner,
    }


# Module-level cache for the registry
_PLANNER_REGISTRY: Optional[dict[Algorithm, type]] = None


def _planner_registry() -> dict[Algorithm, type]:
    global _PLANNER_REGISTRY
    if _PLANNER_REGISTRY is None:
        _PLANNER_REGISTRY = _get_planner_registry()
    return _PLANNER_REGISTRY


def register_planner(algorithm: Algorithm, planner_cls: Type[BasePlanner]) -> None:
    """Register a custom planner class for *algorithm*.

    This allows future navigation algorithms (VLFM, InstructNav, …) to be
    plugged in without modifying the controller.
    """
    _planner_registry()[algorithm] = planner_cls


class NavigationController:
    """High-level entry point for 2-D navigation on a MuJoCo scene.

    Typical usage::

        sim = StretchMujocoSimulator(scene_xml_path="office_scene.xml")
        sim.start(headless=True)

        nav = NavigationController.from_simulator(sim, algorithm=Algorithm.ASTAR)
        path = nav.plan(start=(0.0, 0.0), goal=(2.0, 1.0))

        for wp in path:
            print(wp)

    You can also construct a controller directly from an ``MjModel`` +
    ``MjData`` pair via ``NavigationController(model, data, …)``.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        *,
        algorithm: Algorithm | str = Algorithm.ASTAR,
        resolution: float = 0.08,
        agent_radius: float = 0.25,
        bounds: tuple[float, float, float, float] | None = None,
        floor_geom_name: str = "office_floor",
        minimum_obstacle_height: float = 0.08,
        maximum_obstacle_height: float = 1.80,
        require_collision: bool = True,
        exclude_prefixes: tuple[str, ...] = (),
        planner_kwargs: Optional[dict] = None,
    ) -> None:
        """
        Parameters
        ----------
        model:
            MuJoCo model (the scene + robot).
        data:
            MuJoCo data buffer, stepped at least once so geom positions
            are valid.
        algorithm:
            Which planner to use (``"astar"``, ``"fmm"``, ``"fbe"``, or
            ``"vlfm"``).
        resolution:
            Metres per grid cell.
        agent_radius:
            Extra inflation radius around obstacles (metres).
        bounds:
            Explicit ``(x_min, x_max, y_min, y_max)`` in world coordinates.
            Required when the floor is a plane (no natural bounds).  When
            *None*, bounds are inferred from the named floor box.
        floor_geom_name:
            Name of the box geom that defines walkable bounds (ignored when
            *bounds* is set).
        minimum_obstacle_height:
            Geoms whose top is below this world-Z value are ignored.
        maximum_obstacle_height:
            Geoms whose bottom is above this world-Z value are ignored.
        require_collision:
            When True, only collision geoms are treated as obstacles.
            Set False for visual-only scenes (e.g. Habitat).
        exclude_prefixes:
            Geom names with these prefixes are excluded from obstacles.
        planner_kwargs:
            Extra keyword arguments forwarded to the planner constructor
            (e.g. ``{"smoothing": False}``).
        """
        if isinstance(algorithm, str):
            algorithm = Algorithm(algorithm)

        self.algorithm = algorithm
        self.model = model
        self.data = data
        self._grid_kwargs = {
            "resolution": resolution,
            "agent_radius": agent_radius,
            "bounds": bounds,
            "floor_geom_name": floor_geom_name,
            "minimum_obstacle_height": minimum_obstacle_height,
            "maximum_obstacle_height": maximum_obstacle_height,
            "require_collision": require_collision,
            "exclude_prefixes": exclude_prefixes,
        }
        self._planner_kwargs = dict(planner_kwargs or {})

        # Build the occupancy grid once
        self.grid = OccupancyGrid.from_model(
            model,
            data,
            **self._grid_kwargs,
        )

        # Instantiate the planner
        registry = _planner_registry()
        planner_cls = registry.get(algorithm)
        if planner_cls is None:
            available = ", ".join(a.value for a in registry)
            raise ValueError(f"Unknown algorithm '{algorithm.value}'. Available: {available}")
        if algorithm in _EXPLORATION_ALGORITHMS:
            self._planner = planner_cls(self.grid, **self._planner_kwargs)
        else:
            self._planner = planner_cls(**self._planner_kwargs)

    # ------------------------------------------------------------------
    # Convenience constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_simulator(
        cls,
        simulator: "StretchMujocoSimulator",  # noqa: F821
        *,
        algorithm: Algorithm | str = Algorithm.ASTAR,
        resolution: float = 0.08,
        agent_radius: float = 0.25,
        bounds: tuple[float, float, float, float] | None = None,
        floor_geom_name: str = "office_floor",
        minimum_obstacle_height: float = 0.08,
        maximum_obstacle_height: float = 1.80,
        require_collision: bool = True,
        exclude_prefixes: tuple[str, ...] = (),
        planner_kwargs: Optional[dict] = None,
    ) -> "NavigationController":
        """Build a controller from a running ``StretchMujocoSimulator``.

        The simulator must be in headless mode (or the physics thread must
        be running) so that ``mjmodel`` and ``mjdata`` are accessible inside
        this process.  This only works when you are in the *same* process as
        the simulator (i.e., headless mode without the managed viewer
        process separation).

        For the multi-process simulator, use ``from_model_data`` instead
        with a manually loaded model.
        """
        if not simulator.is_running():
            raise RuntimeError("Simulator is not running. Call sim.start() first.")

        status = simulator.pull_status()
        if status.time == 0:
            raise RuntimeError("Simulator physics have not started yet.")

        import mujoco as _mj

        scene_path = simulator.scene_xml_path
        if scene_path is None:
            from stretch_mujoco.utils import default_scene_xml_path

            scene_path = default_scene_xml_path

        model = _mj.MjModel.from_xml_path(scene_path)
        data = _mj.MjData(model)
        _mj.mj_step(model, data)

        return cls(
            model,
            data,
            algorithm=algorithm,
            resolution=resolution,
            agent_radius=agent_radius,
            bounds=bounds,
            floor_geom_name=floor_geom_name,
            minimum_obstacle_height=minimum_obstacle_height,
            maximum_obstacle_height=maximum_obstacle_height,
            require_collision=require_collision,
            exclude_prefixes=exclude_prefixes,
            planner_kwargs=planner_kwargs,
        )

    @classmethod
    def from_scene_xml(
        cls,
        scene_xml_path: str,
        *,
        algorithm: Algorithm | str = Algorithm.ASTAR,
        resolution: float = 0.08,
        agent_radius: float = 0.25,
        bounds: tuple[float, float, float, float] | None = None,
        floor_geom_name: str = "office_floor",
        minimum_obstacle_height: float = 0.08,
        maximum_obstacle_height: float = 1.80,
        require_collision: bool = True,
        exclude_prefixes: tuple[str, ...] = (),
        planner_kwargs: Optional[dict] = None,
    ) -> "NavigationController":
        """Build a controller directly from a MuJoCo scene XML file.

        This is the simplest way to get started — just point it at your
        scene file.
        """
        import mujoco as _mj

        model = _mj.MjModel.from_xml_path(scene_xml_path)
        data = _mj.MjData(model)
        _mj.mj_step(model, data)
        return cls(
            model,
            data,
            algorithm=algorithm,
            resolution=resolution,
            agent_radius=agent_radius,
            bounds=bounds,
            floor_geom_name=floor_geom_name,
            minimum_obstacle_height=minimum_obstacle_height,
            maximum_obstacle_height=maximum_obstacle_height,
            require_collision=require_collision,
            exclude_prefixes=exclude_prefixes,
            planner_kwargs=planner_kwargs,
        )

    # ------------------------------------------------------------------
    # Planning
    # ------------------------------------------------------------------

    def plan(
        self,
        start: tuple[float, float] | np.ndarray,
        goal: tuple[float, float] | np.ndarray,
    ) -> list[np.ndarray]:
        """Compute a path from *start* to *goal*.

        Parameters
        ----------
        start:
            World (x, y) start position.
        goal:
            World (x, y) goal position.

        Returns
        -------
        list[np.ndarray]
            Ordered list of 2-D waypoints (each ``shape=(2,)``) from start
            to goal.
        """
        if self.algorithm in _EXPLORATION_ALGORITHMS:
            raise RuntimeError(
                f"{self.algorithm.value.upper()} is an exploration algorithm "
                "without a fixed goal; "
                "use step_exploration(robot_xy, robot_yaw) instead."
            )
        return self._planner.plan(
            self.grid,
            np.asarray(start, dtype=float),
            np.asarray(goal, dtype=float),
        )

    def step_exploration(
        self,
        robot_xy: tuple[float, float] | np.ndarray,
        robot_yaw: float,
    ) -> Any:
        """Advance FBE/VLFM once and return its current diagnostics."""
        if self.algorithm not in _EXPLORATION_ALGORITHMS:
            raise RuntimeError("step_exploration() requires FBE or VLFM")
        self._planner.step(np.asarray(robot_xy, dtype=float), robot_yaw)
        return self._planner.diag

    @property
    def explorer(self) -> Any:
        """Return the stateful exploration planner owned by this controller."""
        if self.algorithm not in _EXPLORATION_ALGORITHMS:
            raise RuntimeError("explorer is only available for FBE or VLFM")
        return self._planner

    def is_free(self, point: tuple[float, float] | np.ndarray) -> bool:
        """Return True if a world point lies in free space."""
        return self.grid.is_world_free(np.asarray(point, dtype=float))

    def refresh_grid(self) -> None:
        """Re-build the occupancy grid from the current MuJoCo state.

        Call this after moving objects in the scene that should affect
        navigation.
        """
        self.grid = OccupancyGrid.from_model(
            self.model,
            self.data,
            **self._grid_kwargs,
        )
        if getattr(self, "algorithm", None) in _EXPLORATION_ALGORITHMS:
            planner_cls = _planner_registry()[self.algorithm]
            self._planner = planner_cls(self.grid, **self._planner_kwargs)
