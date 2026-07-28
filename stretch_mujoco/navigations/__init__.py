"""Navigation strategies for the Stretch robot in MuJoCo scenes.

Provides a unified ``NavigationController`` that builds an occupancy grid
from a MuJoCo scene and delegates path planning to a selected algorithm.

Currently supported algorithms:

- **A*** — ``Algorithm.ASTAR``
- **FMM** (Fast Marching Method) — ``Algorithm.FMM``
- **FBE** (Frontier-Based Exploration) — ``Algorithm.FBE``
- **VLFM** (Vision-Language Frontier Map) — ``Algorithm.VLFM``

Adding a new algorithm
----------------------
1. Subclass ``BasePlanner`` and implement ``plan(grid, start, goal)``.
2. Register it via ``register_planner(Algorithm("my_algo"), MyPlanner)``
   or by adding an entry to the ``_PLANNER_REGISTRY`` in ``controller.py``.

Examples
--------
>>> from stretch_mujoco.navigations import (
...     NavigationController,
...     Algorithm,
... )
>>> nav = NavigationController.from_scene_xml("models/office_scene.xml")
>>> path = nav.plan(start=(0.0, 0.0), goal=(2.0, 1.0))
>>> for wp in path:
...     print(wp)
"""

from stretch_mujoco.navigations._loader import _load_module

from stretch_mujoco.navigations.base import (
    BasePlanner,
    NavigationPathError,
    ObstacleFootprint,
    OccupancyGrid,
)
from stretch_mujoco.navigations.controller import (
    Algorithm,
    NavigationController,
    register_planner,
)
from stretch_mujoco.navigations.FBE import FBEDiagnostics, FBEPlanner, FBEState
from stretch_mujoco.navigations.VLFM import (
    LanguageValueMap,
    VLFMDiagnostics,
    VLFMPlanner,
)

# Load planners from subdirectories (A* has a special character in its
# name so we use importlib for both for consistency).
_astar = _load_module("A*", "planner")
_fmm = _load_module("FMM", "planner")

AStarPlanner = _astar.AStarPlanner
FMMPlanner = _fmm.FMMPlanner

__all__ = [
    # Controller
    "Algorithm",
    "NavigationController",
    "register_planner",
    # Base
    "BasePlanner",
    "NavigationPathError",
    "ObstacleFootprint",
    "OccupancyGrid",
    # Planners
    "AStarPlanner",
    "FMMPlanner",
    "FBEPlanner",
    "FBEDiagnostics",
    "FBEState",
    "VLFMPlanner",
    "VLFMDiagnostics",
    "LanguageValueMap",
]
