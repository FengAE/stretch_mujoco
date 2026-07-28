"""A* navigation planner.

The ``A*`` directory name contains a special character that is not a valid
Python identifier, so this package uses ``importlib`` to expose its contents.
"""

from stretch_mujoco.navigations._loader import _load_module

_module = _load_module("A*", "planner")
AStarPlanner = _module.AStarPlanner

__all__ = ["AStarPlanner"]
