"""Scene / world-query MCP tool implementations."""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np

from stretch_mujoco.utils import default_scene_xml_path

from strech_codex.world.state import get_scene_path, get_sim, has_sim

# ---------------------------------------------------------------------------
# Object queries
# ---------------------------------------------------------------------------


def scene_list_objects() -> dict[str, Any]:
    """List movable objects (free-joint bodies) in the current scene."""
    if not has_sim():
        return {"success": False, "message": "No active simulator — call robot_init first"}
    try:
        sim = get_sim()
        # Access the underlying MjModel to enumerate free-joint bodies
        if hasattr(sim, "mjmodel") and sim.mjmodel is not None:
            model = sim.mjmodel
        elif hasattr(sim, "_internal"):
            internal = sim._internal
            if hasattr(internal, "mjmodel"):
                model = internal.mjmodel
            elif hasattr(internal, "_server"):
                model = getattr(internal._server, "mjmodel", None)
            else:
                model = None
        else:
            model = None

        if model is None:
            scene_path = get_scene_path() or getattr(sim, "_scene_xml_path", None)
            model = mujoco.MjModel.from_xml_path(scene_path or default_scene_xml_path)

        free_bodies = []
        for body_id in range(model.nbody):
            jnt_id = model.body_jntadr[body_id]
            if jnt_id >= 0 and model.jnt_type[jnt_id] == 0:  # mjJNT_FREE = 0
                body_name = (
                    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
                    or f"body_{body_id}"
                )
                free_bodies.append(body_name)

        return {"success": True, "objects": free_bodies, "count": len(free_bodies)}
    except Exception as exc:
        return {"success": False, "message": str(exc)}


def scene_get_object_pose(object_name: str) -> dict[str, Any]:
    """Get the world pose (4×4 matrix) of a named body."""
    if not has_sim():
        return {"success": False, "message": "No active simulator — call robot_init first"}
    try:
        sim = get_sim()
        try:
            pose = sim.get_link_pose(object_name)
        except Exception:
            scene_path = get_scene_path() or getattr(sim, "_scene_xml_path", None)
            model = mujoco.MjModel.from_xml_path(scene_path or default_scene_xml_path)
            data = mujoco.MjData(model)
            mujoco.mj_forward(model, data)
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, object_name)
            if body_id < 0:
                raise ValueError(f"Body '{object_name}' not found in scene")
            # ponytail: Stretch does not expose live scene-body poses; upgrade the
            # unified simulator API if callers need poses after physics motion.
            pose = np.eye(4)
            pose[:3, :3] = data.xmat[body_id].reshape(3, 3)
            pose[:3, 3] = data.xpos[body_id]

        return {
            "success": True,
            "object": object_name,
            "pose": pose.tolist(),
        }
    except Exception as exc:
        return {"success": False, "message": str(exc)}


def scene_add_world_frame(
    x: float = 0.0,
    y: float = 0.0,
    z: float = 0.0,
) -> dict[str, Any]:
    """Add a coordinate-frame marker at (x, y, z) in the viewer."""
    if not has_sim():
        return {"success": False, "message": "No active simulator — call robot_init first"}
    try:
        get_sim().add_world_frame((x, y, z))
        return {"success": True, "message": f"World frame added at ({x}, {y}, {z})"}
    except Exception as exc:
        return {"success": False, "message": str(exc)}


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------

SCENE_TOOL_FUNCTIONS: dict[str, Any] = {
    "scene_list_objects": scene_list_objects,
    "scene_get_object_pose": scene_get_object_pose,
    "scene_add_world_frame": scene_add_world_frame,
}
