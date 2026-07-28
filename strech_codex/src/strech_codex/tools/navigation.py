"""Navigation MCP tool implementations.

Wraps the ``stretch_mujoco.navigations`` package as callable tool functions.
Each function takes keyword arguments and returns a ``{"success": bool, ...}`` dict.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from stretch_mujoco.navigations import (
    AStarPlanner,
    Algorithm,
    FBEPlanner,
    FMMPlanner,
    NavigationController,
    NavigationPathError,
    OccupancyGrid,
)

from strech_codex.world.state import (
    get_explorer,
    get_nav,
    get_robot_type,
    get_scene_path,
    get_sim,
    has_explorer,
    has_nav,
    has_sim,
    set_explorer,
    set_nav,
    set_scene_path,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


import stretch_mujoco as _stretch_mujoco  # noqa: E402

_SMUJOCOROOT = Path(_stretch_mujoco.__file__).resolve().parent

_DEFAULT_SCENE = str(_SMUJOCOROOT / "models" / "office_scene.xml")

_SCENE_SEARCH_DIRS: list[Path] = [
    _SMUJOCOROOT / "models",
    _SMUJOCOROOT / "models" / "assets" / "office_scenes",
]


def resolve_scene(name_or_path: str) -> str:
    """Resolve a scene name or partial path to a full XML path.

    1. If it's already a valid file path, return as-is.
    2. Try appending ``.xml`` and looking in the search dirs.
    3. Try the exact basename in the search dirs.
    """
    if Path(name_or_path).is_file():
        return name_or_path
    for base in _SCENE_SEARCH_DIRS:
        # "office_02_cross_axis" → "office_02_cross_axis.xml"
        candidate = base / f"{name_or_path}.xml"
        if candidate.is_file():
            return str(candidate)
        # "office_02_cross_axis.xml" → full path
        candidate = base / name_or_path
        if candidate.is_file():
            return str(candidate)
    return name_or_path  # let MuJoCo give the final error


def _load_model_data(scene_xml: str | None = None):
    """Load a MuJoCo model + data pair.

    Resolution order: explicit *scene_xml* → world state scene path →
    bundled default office scene.  *scene_xml* may be a basename such as
    ``"office_02_cross_axis"`` which is resolved against known scene directories.
    """
    path: str | None = None
    if scene_xml:
        path = resolve_scene(scene_xml)
    path = path or get_scene_path() or _DEFAULT_SCENE
    if path is None or not Path(path).is_file():
        raise RuntimeError(
            f"Scene XML not found — tried: {path or scene_xml}. "
            f"Use nav_list_scenes to see available scenes, "
            f"or pass a full path to scene_xml."
        )
    model = mujoco.MjModel.from_xml_path(path)
    _exclude_robot_from_grid(model)
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data)
    return model, data


def _exclude_robot_from_grid(model: mujoco.MjModel) -> None:
    """Disable collision on the ``base_link`` subtree in the navigation copy."""
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    if base_id < 0:
        return
    robot_bodies = {base_id}
    for body_id in range(1, model.nbody):
        parent = body_id
        while parent > 0:
            if parent == base_id:
                robot_bodies.add(body_id)
                break
            parent = int(model.body_parentid[parent])
    for geom_id in range(model.ngeom):
        if int(model.geom_bodyid[geom_id]) in robot_bodies:
            model.geom_contype[geom_id] = 0
            model.geom_conaffinity[geom_id] = 0


def _waypoints_to_list(path: list[np.ndarray]) -> list[list[float]]:
    """Convert a list of np.ndarray waypoints to a JSON-friendly list."""
    return [pt.tolist() for pt in path]


def _path_distance(path: list[np.ndarray]) -> float:
    """Compute total Euclidean distance along a path."""
    if len(path) < 2:
        return 0.0
    total = 0.0
    for i in range(len(path) - 1):
        total += float(np.linalg.norm(path[i + 1] - path[i]))
    return total


def _default_bounds(model: mujoco.MjModel) -> tuple[float, float, float, float]:
    """Provide finite bounds for robot scenes that use an infinite plane floor."""
    extent = max(3.0, float(model.stat.extent))
    center_x, center_y = (float(value) for value in model.stat.center[:2])
    return (center_x - extent, center_x + extent, center_y - extent, center_y + extent)


def _wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


# ---------------------------------------------------------------------------
# Grid building
# ---------------------------------------------------------------------------


def nav_build_grid(
    scene_xml: str | None = None,
    bounds: list[float] | None = None,
    resolution: float = 0.08,
    agent_radius: float = 0.25,
    floor_geom_name: str = "office_floor",
    minimum_obstacle_height: float = 0.08,
    maximum_obstacle_height: float = 1.80,
    require_collision: bool = True,
    exclude_prefixes: list[str] | None = None,
) -> dict[str, Any]:
    """Build (or rebuild) the occupancy grid from a MuJoCo scene.

    This must be called before any navigation tool.
    """
    try:
        model, data = _load_model_data(scene_xml)
        if scene_xml:
            set_scene_path(scene_xml)

        bounds_tuple: tuple[float, float, float, float] | None = None
        if bounds is not None:
            if len(bounds) != 4:
                raise ValueError("bounds must contain [x_min, x_max, y_min, y_max]")
            bounds_tuple = tuple(float(value) for value in bounds)
        elif mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, floor_geom_name) < 0:
            bounds_tuple = _default_bounds(model)
        elif model.geom_type[model.geom(floor_geom_name).id] != mujoco.mjtGeom.mjGEOM_BOX:
            bounds_tuple = _default_bounds(model)

        nav = NavigationController(
            model,
            data,
            algorithm=Algorithm.ASTAR,
            resolution=resolution,
            agent_radius=agent_radius,
            bounds=bounds_tuple,
            floor_geom_name=floor_geom_name,
            minimum_obstacle_height=minimum_obstacle_height,
            maximum_obstacle_height=maximum_obstacle_height,
            require_collision=require_collision,
            exclude_prefixes=tuple(exclude_prefixes or ()),
        )
        set_nav(nav)

        occ = nav.grid.occupancy
        free = int((~occ).sum())
        return {
            "success": True,
            "message": f"Grid built: {occ.shape[0]}x{occ.shape[1]}, {free}/{occ.size} free ({100*free/occ.size:.1f}%)",
            "width": occ.shape[1],
            "height": occ.shape[0],
            "resolution": resolution,
            "agent_radius": agent_radius,
            "bounds": [nav.grid.x_min, nav.grid.x_max, nav.grid.y_min, nav.grid.y_max],
            "free_cells": free,
            "total_cells": int(occ.size),
        }
    except Exception as exc:
        return {"success": False, "message": f"Failed to build grid: {exc}"}


def nav_get_grid_info() -> dict[str, Any]:
    """Return information about the current occupancy grid."""
    if not has_nav():
        return {"success": False, "message": "No grid — call nav_build_grid first"}
    try:
        nav = get_nav()
        occ = nav.grid.occupancy
        free = int((~occ).sum())
        return {
            "success": True,
            "width": occ.shape[1],
            "height": occ.shape[0],
            "resolution": nav.grid.resolution,
            "agent_radius": nav.grid.agent_radius,
            "bounds": [nav.grid.x_min, nav.grid.x_max, nav.grid.y_min, nav.grid.y_max],
            "free_cells": free,
            "total_cells": int(occ.size),
            "free_ratio": float(free / occ.size),
        }
    except Exception as exc:
        return {"success": False, "message": str(exc)}


def nav_is_free(x: float, y: float) -> dict[str, Any]:
    """Check whether a world point (x, y) is navigable."""
    if not has_nav():
        return {"success": False, "message": "No grid — call nav_build_grid first"}
    try:
        free = get_nav().is_free((x, y))
        return {"success": True, "x": x, "y": y, "is_free": free}
    except Exception as exc:
        return {"success": False, "message": str(exc)}


def nav_list_scenes() -> dict[str, Any]:
    """List all discoverable scene XML files.

    Returns scene names (without ``.xml``) and their full paths.
    Use the scene name directly as ``scene_xml`` in ``nav_build_grid``.
    """
    scenes: list[dict[str, str]] = []
    for base in _SCENE_SEARCH_DIRS:
        if not base.is_dir():
            continue
        for xml_path in sorted(base.glob("*.xml")):
            scenes.append(
                {
                    "name": xml_path.stem,
                    "path": str(xml_path),
                }
            )
    return {
        "success": True,
        "scenes": scenes,
        "count": len(scenes),
        "message": (
            f"{len(scenes)} scenes available. "
            "Use 'name' as scene_xml in robot_init or nav_build_grid."
        ),
    }


# ---------------------------------------------------------------------------
# A* path planning
# ---------------------------------------------------------------------------


def nav_astar(
    start_x: float,
    start_y: float,
    goal_x: float,
    goal_y: float,
    smoothing: bool = True,
) -> dict[str, Any]:
    """Plan an A* path from (start_x, start_y) to (goal_x, goal_y).

    Returns the waypoint list, path distance, and number of waypoints.
    """
    if not has_nav():
        return {"success": False, "message": "No grid — call nav_build_grid first"}
    try:
        planner = AStarPlanner(smoothing=smoothing)
        path = planner.plan(
            get_nav().grid,
            np.array([start_x, start_y]),
            np.array([goal_x, goal_y]),
        )
        waypoints = _waypoints_to_list(path)
        return {
            "success": True,
            "message": f"A* found path with {len(waypoints)} waypoints, distance {_path_distance(path):.2f}m",
            "waypoints": waypoints,
            "num_waypoints": len(waypoints),
            "distance_m": round(_path_distance(path), 3),
            "algorithm": "astar",
        }
    except NavigationPathError as exc:
        return {"success": False, "message": f"A* path failed: {exc}"}
    except Exception as exc:
        return {"success": False, "message": str(exc)}


# ---------------------------------------------------------------------------
# FMM path planning
# ---------------------------------------------------------------------------


def nav_fmm(
    start_x: float,
    start_y: float,
    goal_x: float,
    goal_y: float,
    smoothing: bool = True,
) -> dict[str, Any]:
    """Plan an FMM (Fast Marching Method) path from start to goal."""
    if not has_nav():
        return {"success": False, "message": "No grid — call nav_build_grid first"}
    try:
        planner = FMMPlanner(smoothing=smoothing)
        path = planner.plan(
            get_nav().grid,
            np.array([start_x, start_y]),
            np.array([goal_x, goal_y]),
        )
        waypoints = _waypoints_to_list(path)
        return {
            "success": True,
            "message": f"FMM found path with {len(waypoints)} waypoints, distance {_path_distance(path):.2f}m",
            "waypoints": waypoints,
            "num_waypoints": len(waypoints),
            "distance_m": round(_path_distance(path), 3),
            "algorithm": "fmm",
        }
    except NavigationPathError as exc:
        return {"success": False, "message": f"FMM path failed: {exc}"}
    except Exception as exc:
        return {"success": False, "message": str(exc)}


# ---------------------------------------------------------------------------
# Path execution
# ---------------------------------------------------------------------------


def nav_execute_path(
    waypoints: list[list[float]],
    timeout: float = 120.0,
    position_tolerance: float = 0.12,
    start_tolerance: float = 0.5,
    max_linear_speed: float = 0.4,
    max_angular_speed: float = 0.8,
    control_hz: float = 20.0,
) -> dict[str, Any]:
    """Drive the active robot along world-frame ``[x, y]`` waypoints."""
    if not has_sim():
        return {"success": False, "message": "No active simulator — call robot_init first"}
    try:
        path = np.asarray(waypoints, dtype=float)
        if path.ndim != 2 or path.shape[0] == 0 or path.shape[1] != 2:
            raise ValueError("waypoints must be a non-empty list of [x, y] points")
        if not np.isfinite(path).all():
            raise ValueError("waypoints must contain only finite numbers")
        if timeout <= 0 or position_tolerance <= 0 or start_tolerance < 0:
            raise ValueError("timeout/tolerances must be positive")
        if max_linear_speed <= 0 or max_angular_speed <= 0 or control_hz <= 0:
            raise ValueError("speeds and control_hz must be positive")

        sim = get_sim()
        robot_type = get_robot_type()
        omni = robot_type in {"google_robot", "tidybot"}
        start_pose = np.asarray(sim.get_base_pose(), dtype=float)
        start_error = float(np.linalg.norm(path[0] - start_pose[:2]))
        if start_error > start_tolerance:
            return {
                "success": False,
                "message": (
                    f"Robot pose {start_pose[:2].tolist()} is {start_error:.3f}m from "
                    f"path start {path[0].tolist()}; initialize/replan from the current pose"
                ),
                "current_pose": start_pose.tolist(),
                "path_start": path[0].tolist(),
            }

        deadline = time.monotonic() + min(float(timeout), 300.0)
        waypoint_index = 1 if len(path) > 1 and start_error <= position_tolerance else 0
        while waypoint_index < len(path):
            if time.monotonic() >= deadline:
                pose = np.asarray(sim.get_base_pose(), dtype=float)
                return {
                    "success": False,
                    "message": f"Path execution timed out at waypoint {waypoint_index}",
                    "waypoints_completed": waypoint_index,
                    "final_pose": pose.tolist(),
                }
            if not sim.is_running():
                raise RuntimeError("Simulator stopped during path execution")

            x, y, yaw = (float(value) for value in sim.get_base_pose())
            dx, dy = path[waypoint_index] - np.array([x, y])
            distance = float(np.hypot(dx, dy))
            waypoint_tolerance = (
                position_tolerance
                if waypoint_index == len(path) - 1
                else max(position_tolerance, 0.3)
            )
            if distance <= waypoint_tolerance:
                waypoint_index += 1
                continue

            speed = min(max_linear_speed, distance)
            if omni:
                forward = speed * (math.cos(yaw) * dx + math.sin(yaw) * dy) / distance
                lateral = speed * (-math.sin(yaw) * dx + math.cos(yaw) * dy) / distance
                sim.set_base_velocity(forward, 0.0, lateral)
            else:
                heading_error = _wrap_angle(math.atan2(dy, dx) - yaw)
                omega = float(np.clip(2.0 * heading_error, -max_angular_speed, max_angular_speed))
                forward = min(max_linear_speed, max(0.05, distance)) * max(
                    0.0, math.cos(heading_error)
                )
                if abs(heading_error) > 0.7:
                    forward = 0.0
                sim.set_base_velocity(forward, omega, 0.0)
            time.sleep(1.0 / control_hz)

        final_pose = [float(value) for value in sim.get_base_pose()]
        return {
            "success": True,
            "message": f"Executed {len(path)} waypoints with {robot_type}",
            "robot_type": robot_type,
            "drive_type": "omnidirectional" if omni else "differential",
            "waypoints_completed": len(path),
            "final_pose": final_pose,
            "goal_error_m": round(float(np.linalg.norm(path[-1] - final_pose[:2])), 4),
        }
    except Exception as exc:
        return {"success": False, "message": f"Path execution failed: {exc}"}
    finally:
        try:
            get_sim().set_base_velocity(0.0, 0.0, 0.0)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# FBE (Frontier-Based Exploration)
# ---------------------------------------------------------------------------


def nav_fbe_init(
    scene_xml: str | None = None,
    bounds: list[float] | None = None,
    resolution: float = 0.08,
    agent_radius: float = 0.25,
    num_rays: int = 180,
    max_range_m: float = 5.0,
    fov_degrees: float = 270.0,
    min_cluster_size: int = 3,
    explore_threshold: float = 0.85,
    start_x: float = 0.0,
    start_y: float = 0.0,
    start_yaw: float = 0.0,
) -> dict[str, Any]:
    """Initialize FBE (Frontier-Based Exploration).

    Builds a god grid from the scene and creates an FBEPlanner.
    Call ``nav_fbe_step`` repeatedly to explore.
    """
    try:
        model, data = _load_model_data(scene_xml)

        bounds_tuple: tuple[float, float, float, float] | None = None
        if bounds is not None and len(bounds) == 4:
            bounds_tuple = (float(bounds[0]), float(bounds[1]), float(bounds[2]), float(bounds[3]))

        god_grid = OccupancyGrid.from_model(
            model,
            data,
            resolution=resolution,
            agent_radius=agent_radius,
            bounds=bounds_tuple,
        )

        # Also set up a NavigationController for general use
        nav = NavigationController(
            model,
            data,
            algorithm=Algorithm.FBE,
            resolution=resolution,
            agent_radius=agent_radius,
            bounds=bounds_tuple,
        )
        set_nav(nav)

        fbe = FBEPlanner(
            god_grid,
            robot_xy=(start_x, start_y),
            robot_yaw=start_yaw,
            num_rays=num_rays,
            max_range_m=max_range_m,
            fov_degrees=fov_degrees,
            min_cluster_size=min_cluster_size,
            explore_threshold=explore_threshold,
        )
        set_explorer(fbe)

        occ = god_grid.occupancy
        free = int((~occ).sum())
        return {
            "success": True,
            "message": f"FBE initialized. Grid {occ.shape[0]}x{occ.shape[1]}, {free}/{occ.size} free",
            "state": fbe.state.value,
            "explored_ratio": fbe.local_map.explored_ratio(),
            "grid_bounds": [god_grid.x_min, god_grid.x_max, god_grid.y_min, god_grid.y_max],
        }
    except Exception as exc:
        return {"success": False, "message": f"FBE init failed: {exc}"}


def nav_fbe_step(
    robot_x: float,
    robot_y: float,
    robot_yaw: float,
) -> dict[str, Any]:
    """Advance FBE exploration by one step.

    Call this repeatedly (at ~10-20 Hz) with the robot's current pose.
    Returns the current state, frontier info, and the next movement command
    (target waypoint) if available.
    """
    if not has_explorer():
        return {"success": False, "message": "No active explorer — call nav_fbe_init first"}
    try:
        fbe = get_explorer()
        fbe.step((robot_x, robot_y), robot_yaw)

        result: dict[str, Any] = {
            "success": True,
            "state": fbe.state.value,
            "explored_ratio": round(fbe.local_map.explored_ratio(), 4),
            "frontier_count": fbe.diag.frontier_count,
            "cluster_count": fbe.diag.cluster_count,
        }

        if fbe.has_path():
            cmd = fbe.pop_command()
            if cmd is not None:
                result["command"] = cmd.tolist()
                result["has_path"] = True
        else:
            result["has_path"] = False

        if fbe.is_finished():
            result["message"] = f"Exploration finished: {fbe.diag.finish_reason}"

        return result
    except Exception as exc:
        return {"success": False, "message": str(exc)}


def nav_fbe_path() -> dict[str, Any]:
    """Return the current FBE path waypoints."""
    if not has_explorer():
        return {"success": False, "message": "No active explorer — call nav_fbe_init first"}
    try:
        fbe = get_explorer()
        path = fbe.diag.current_path
        return {
            "success": True,
            "waypoints": _waypoints_to_list(path),
            "num_waypoints": len(path),
            "state": fbe.state.value,
        }
    except Exception as exc:
        return {"success": False, "message": str(exc)}


# ---------------------------------------------------------------------------
# VLFM (Vision-Language Frontier Maps)
# ---------------------------------------------------------------------------


def nav_vlfm_init(
    instruction: str,
    scene_xml: str | None = None,
    bounds: list[float] | None = None,
    resolution: float = 0.08,
    agent_radius: float = 0.25,
    num_rays: int = 180,
    max_range_m: float = 5.0,
    fov_degrees: float = 270.0,
    min_cluster_size: int = 3,
    explore_threshold: float = 0.85,
    vlm_type: str = "clip",
    device: str = "cpu",
    start_x: float = 0.0,
    start_y: float = 0.0,
    start_yaw: float = 0.0,
    allow_geometric_fallback: bool = True,
) -> dict[str, Any]:
    """Initialize VLFM (Vision-Language Frontier Maps) exploration.

    Requires a VLM backend (CLIP, SigLIP, BLIP2, or OpenAI).
    The ``instruction`` is a natural-language description of what to find.
    """
    try:
        from stretch_mujoco.navigations.VLFM import VLFMPlanner
        from stretch_mujoco.navigations.VLFM.vlm_client import VLMClient

        # --- Build VLM client ---
        vlm_type = vlm_type.lower()
        if vlm_type == "clip":
            from stretch_mujoco.navigations.VLFM import CLIPVLMClient

            vlm_client: VLMClient = CLIPVLMClient(device=device)
        elif vlm_type == "siglip":
            from stretch_mujoco.navigations.VLFM import SigLIPVLMClient

            vlm_client = SigLIPVLMClient(device=device)
        elif vlm_type == "openai":
            import os as _os

            from stretch_mujoco.navigations.VLFM import OpenAIVLMClient

            api_key = _os.environ.get("OPENAI_API_KEY", "")
            vlm_client = OpenAIVLMClient(model="gpt-4o", api_key=api_key)
        elif vlm_type == "blip2":
            from stretch_mujoco.navigations.VLFM import BLIP2ITMVLMClient

            vlm_client = BLIP2ITMVLMClient(base_url="http://127.0.0.1:12182")
        else:
            return {"success": False, "message": f"Unknown VLM type: {vlm_type}"}

        vlm_client.validate_environment()
        vlm_client.prepare()

        # --- Build god grid ---
        model, data = _load_model_data(scene_xml)

        bounds_tuple: tuple[float, float, float, float] | None = None
        if bounds is not None and len(bounds) == 4:
            bounds_tuple = (float(bounds[0]), float(bounds[1]), float(bounds[2]), float(bounds[3]))

        god_grid = OccupancyGrid.from_model(
            model,
            data,
            resolution=resolution,
            agent_radius=agent_radius,
            bounds=bounds_tuple,
        )

        nav = NavigationController(
            model,
            data,
            algorithm=Algorithm.VLFM,
            resolution=resolution,
            agent_radius=agent_radius,
            bounds=bounds_tuple,
        )
        set_nav(nav)

        vlfm = VLFMPlanner(
            god_grid,
            vlm_client,
            instruction=instruction,
            robot_xy=(start_x, start_y),
            robot_yaw=start_yaw,
            num_rays=num_rays,
            max_range_m=max_range_m,
            fov_degrees=fov_degrees,
            min_cluster_size=min_cluster_size,
            explore_threshold=explore_threshold,
            allow_geometric_fallback=allow_geometric_fallback,
        )
        set_explorer(vlfm)

        return {
            "success": True,
            "message": f"VLFM initialized with instruction: '{instruction}'",
            "vlm_type": vlm_type,
            "state": vlfm.state.value,
            "explored_ratio": vlfm.local_map.explored_ratio(),
            "grid_bounds": [god_grid.x_min, god_grid.x_max, god_grid.y_min, god_grid.y_max],
        }
    except Exception as exc:
        return {"success": False, "message": f"VLFM init failed: {exc}"}


def nav_vlfm_step(
    robot_x: float,
    robot_y: float,
    robot_yaw: float,
) -> dict[str, Any]:
    """Advance VLFM exploration by one step.

    Returns state, VLM score, frontier info, and next movement command.
    """
    if not has_explorer():
        return {"success": False, "message": "No active explorer — call nav_vlfm_init first"}
    try:
        vlfm = get_explorer()
        vlfm.step((robot_x, robot_y), robot_yaw)

        from stretch_mujoco.navigations.VLFM.planner import VLFMDiagnostics

        diag = vlfm.diag
        result: dict[str, Any] = {
            "success": True,
            "state": vlfm.state.value,
            "explored_ratio": round(vlfm.local_map.explored_ratio(), 4),
            "frontier_count": diag.frontier_count,
            "cluster_count": diag.cluster_count,
            "vlm_query_count": diag.vlm_query_count,
            "last_vlm_score": round(diag.last_vlm_score, 4),
        }

        if hasattr(diag, "mode"):
            result["mode"] = diag.mode

        if vlfm.has_path():
            cmd = vlfm.pop_command()
            if cmd is not None:
                result["command"] = cmd.tolist()
                result["has_path"] = True
        else:
            result["has_path"] = False

        if vlfm.is_finished():
            result["message"] = f"Exploration finished: {diag.finish_reason}"

        return result
    except Exception as exc:
        return {"success": False, "message": str(exc)}


def nav_vlfm_inject_observation(
    rgb_image_path: str = "",
    depth_image_path: str = "",
    camera_x: float = 0.0,
    camera_y: float = 0.0,
    camera_yaw: float = 0.0,
    hfov_rad: float = 1.2,
    min_depth: float = 0.1,
    max_depth: float = 5.0,
) -> dict[str, Any]:
    """Inject an RGB-D observation into the VLFM planner for VLM scoring.

    The RGB image is read from *rgb_image_path* and scored against the
    VLFM's instruction. The resulting score is projected onto the spatial
    value map via the depth image.
    """
    if not has_explorer():
        return {"success": False, "message": "No active explorer — call nav_vlfm_init first"}
    try:
        vlfm = get_explorer()

        rgb_bytes = b""
        if rgb_image_path:
            rgb_bytes = Path(rgb_image_path).read_bytes()

        depth_img: np.ndarray | None = None
        if depth_image_path:
            # Assume a .npy file for depth
            depth_img = np.load(depth_image_path)

        if rgb_bytes and depth_img is not None:
            score = vlfm.inject_observation(
                rgb_bytes,
                depth_img,
                (camera_x, camera_y),
                camera_yaw,
                hfov_rad,
                min_depth=min_depth,
                max_depth=max_depth,
            )
            return {
                "success": True,
                "message": f"Observation injected, VLM score: {score:.4f}",
                "vlm_score": round(float(score), 4),
                "vlm_query_count": vlfm.diag.vlm_query_count,
            }
        else:
            return {
                "success": False,
                "message": "Both rgb_image_path and depth_image_path are required",
            }
    except Exception as exc:
        return {"success": False, "message": str(exc)}


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------

NAV_TOOL_FUNCTIONS: dict[str, Any] = {
    # Grid
    "nav_build_grid": nav_build_grid,
    "nav_get_grid_info": nav_get_grid_info,
    "nav_is_free": nav_is_free,
    "nav_list_scenes": nav_list_scenes,
    # A*
    "nav_astar": nav_astar,
    # FMM
    "nav_fmm": nav_fmm,
    "nav_execute_path": nav_execute_path,
    # FBE
    "nav_fbe_init": nav_fbe_init,
    "nav_fbe_step": nav_fbe_step,
    "nav_fbe_path": nav_fbe_path,
    # VLFM
    "nav_vlfm_init": nav_vlfm_init,
    "nav_vlfm_step": nav_vlfm_step,
    "nav_vlfm_inject_observation": nav_vlfm_inject_observation,
}
