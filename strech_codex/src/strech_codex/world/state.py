"""Shared mutable world state for the Codex agent.

Holds the active robot simulator, navigation controller, and scene path
so that all MCP tools can operate on the same instance.

Pattern mirrors ``robot_project/src/robot_project/robot_tools.py``.
"""

from __future__ import annotations

from typing import Any


class _WorldState:
    """Module-level singleton holding the active simulator and navigation controller."""

    def __init__(self) -> None:
        self._sim: Any = None
        self._nav: Any = None
        self._scene_path: str | None = None
        self._robot_type: str | None = None
        self._explorer: Any = None  # FBEPlanner / VLFMPlanner for active exploration

    # -- simulator -----------------------------------------------------------

    @property
    def sim(self) -> Any:
        if self._sim is None:
            raise RuntimeError("No active robot simulator — call robot_init first")
        return self._sim

    @sim.setter
    def sim(self, value: Any) -> None:
        self._sim = value

    @property
    def has_sim(self) -> bool:
        return self._sim is not None

    # -- navigation controller -----------------------------------------------

    @property
    def nav(self) -> Any:
        if self._nav is None:
            raise RuntimeError("No navigation grid — call nav_build_grid first")
        return self._nav

    @nav.setter
    def nav(self, value: Any) -> None:
        self._nav = value

    @property
    def has_nav(self) -> bool:
        return self._nav is not None

    # -- explorer (FBE / VLFM) -----------------------------------------------

    @property
    def explorer(self) -> Any:
        if self._explorer is None:
            raise RuntimeError("No active explorer — call nav_fbe_init or nav_vlfm_init first")
        return self._explorer

    @explorer.setter
    def explorer(self, value: Any) -> None:
        self._explorer = value

    @property
    def has_explorer(self) -> bool:
        return self._explorer is not None

    # -- metadata ------------------------------------------------------------

    @property
    def scene_path(self) -> str | None:
        return self._scene_path

    @scene_path.setter
    def scene_path(self, value: str | None) -> None:
        self._scene_path = value

    @property
    def robot_type(self) -> str | None:
        return self._robot_type

    @robot_type.setter
    def robot_type(self, value: str | None) -> None:
        self._robot_type = value

    # -- reset ---------------------------------------------------------------

    def reset(self) -> dict[str, Any]:
        """Stop the simulator (if running) and clear all state."""
        if self._sim is not None:
            try:
                if self._sim.is_running():
                    self._sim.stop()
            except Exception:
                pass
        self._sim = None
        self._nav = None
        self._explorer = None
        self._scene_path = None
        self._robot_type = None
        return {"success": True, "message": "World state reset"}


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_world = _WorldState()


def get_sim() -> Any:
    """Return the active robot simulator."""
    return _world.sim


def set_sim(sim: Any) -> None:
    """Replace the active robot simulator."""
    _world.sim = sim


def has_sim() -> bool:
    return _world.has_sim


def get_nav() -> Any:
    """Return the active navigation controller."""
    return _world.nav


def set_nav(nav: Any) -> None:
    """Replace the active navigation controller."""
    _world.nav = nav


def has_nav() -> bool:
    return _world.has_nav


def get_explorer() -> Any:
    """Return the active explorer (FBEPlanner or VLFMPlanner)."""
    return _world.explorer


def set_explorer(explorer: Any) -> None:
    _world.explorer = explorer


def has_explorer() -> bool:
    return _world.has_explorer


def get_scene_path() -> str | None:
    return _world.scene_path


def set_scene_path(path: str | None) -> None:
    _world.scene_path = path


def get_robot_type() -> str | None:
    return _world.robot_type


def set_robot_type(rt: str | None) -> None:
    _world.robot_type = rt


def reset_world() -> dict[str, Any]:
    """Reset all world state."""
    return _world.reset()
