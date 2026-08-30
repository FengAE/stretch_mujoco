"""Robot control MCP tool implementations.

Wraps the ``stretch_mujoco.robots`` unified interface as callable tool
functions.  Supports Stretch 3, Google Robot, and Stanford TidyBot.
"""

from __future__ import annotations

import atexit
import os
from typing import Any

import numpy as np

from stretch_mujoco.robots import RobotType, create_simulator

from strech_codex.tools.navigation import resolve_scene
from strech_codex.video_recorder import RobotVideoRecorder
from strech_codex.world.state import (
    get_scene_path,
    get_sim,
    has_sim,
    set_robot_type,
    set_scene_path,
    set_sim,
)


_video_recorder: RobotVideoRecorder | None = None


def _stop_video_recorder() -> dict[str, Any] | None:
    global _video_recorder
    if _video_recorder is None:
        return None
    result = _video_recorder.stop()
    _video_recorder = None
    return result


atexit.register(_stop_video_recorder)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_actuator(sim: Any, name: str) -> Any:
    """Resolve an actuator by name string or enum value."""
    actuators = sim.Actuators
    # Try enum member name
    try:
        return actuators[name]
    except KeyError:
        pass
    # Try by value
    for member in actuators.all():
        if member.value == name:
            return member
    # Try by joint name
    try:
        return actuators.get_actuator_by_joint_names_in_mjcf(name)
    except Exception:
        pass
    available = [m.name for m in actuators.all()]
    raise ValueError(f"Unknown actuator '{name}'. Available: {available}")


def _format_pose(mat: np.ndarray) -> list[list[float]]:
    """Convert a 4x4 transform to a JSON-friendly list of lists."""
    return mat.tolist()


def _resolve_cameras(cameras: Any, names: list[str] | None) -> list[Any]:
    """Resolve MCP camera names to the robot-specific camera enum."""
    if not names:
        return []
    aliases = {"all": cameras.all, "all_rgb": cameras.rgb, "all_depth": cameras.depth}
    result: list[Any] = []
    for name in names:
        if name in aliases:
            matches = aliases[name]()
        else:
            matches = [camera for camera in cameras.all() if camera.name == name]
            if not matches:
                matches = [camera for camera in cameras.all() if camera.camera_name_in_mjcf == name]
        if not matches:
            available = [camera.name for camera in cameras.all()]
            raise ValueError(f"Unknown camera '{name}'. Available: {available}")
        result.extend(camera for camera in matches if camera not in result)
    return result


def _status_to_dict(status: Any) -> dict[str, Any]:
    """Pull key fields from a RobotStatus dataclass."""
    try:
        return status.to_dict()
    except AttributeError:
        return {"time": getattr(status, "time", 0.0), "fps": getattr(status, "fps", 0.0)}


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


def robot_init(
    robot_type: str,
    scene_xml: str | None = None,
    headless: bool = True,
    cameras_to_use: list[str] | None = None,
    camera_hz: float = 30.0,
    start_translation: list[float] | None = None,
    start_rotation_quat: list[float] | None = None,
) -> dict[str, Any]:
    """Create and start a robot simulator.

    Parameters
    ----------
    robot_type:
        One of ``"stretch3"``, ``"google_robot"``, ``"tidybot"``.
    scene_xml:
        Path to the MuJoCo scene XML.
    headless:
        If True, run without the interactive viewer.
    cameras_to_use:
        List of camera names to enable.  Pass ``["all_rgb"]`` for all RGB cameras.
    """
    global _video_recorder
    sim: Any = None
    try:
        if robot_type not in {"stretch3", "google_robot", "tidybot"}:
            return {
                "success": False,
                "message": f"Unknown robot_type '{robot_type}'. Choose: stretch3, google_robot, tidybot",
            }

        if has_sim():
            return {
                "success": False,
                "message": "A simulator is already active — call robot_stop first",
            }

        rt = RobotType(robot_type)
        scene_xml = scene_xml or get_scene_path()
        if scene_xml:
            scene_xml = resolve_scene(scene_xml)

        camera_types = {
            RobotType.STRETCH3: "stretch_mujoco.enums.stretch_cameras.StretchCameras",
            RobotType.GOOGLE_ROBOT: "stretch_mujoco.robots.google_robot.cameras.GoogleRobotCameras",
            RobotType.TIDYBOT: "stretch_mujoco.robots.tidybot.cameras.TidyBotCameras",
        }
        module_name, class_name = camera_types[rt].rsplit(".", 1)
        camera_enum = getattr(__import__(module_name, fromlist=[class_name]), class_name)
        cameras = _resolve_cameras(camera_enum, cameras_to_use)
        video_path = os.environ.get("STRECH_CODEX_VIDEO_PATH")
        evidence_enabled = bool(video_path or os.environ.get("STRECH_CODEX_EPISODE_DIR"))
        if evidence_enabled:
            cameras = camera_enum.all()
            camera_hz = min(float(camera_hz), 10.0)

        sim = create_simulator(
            rt,
            scene_xml_path=scene_xml,
            camera_hz=camera_hz,
            cameras_to_use=cameras,
            start_translation=start_translation,
            start_rotation_quat=start_rotation_quat,
        )
        sim.start(headless=headless)
        if callable(is_running := getattr(sim, "is_running", None)) and not is_running():
            raise RuntimeError("Simulator process exited during startup")
        set_sim(sim)
        set_robot_type(robot_type)
        resolved_scene = scene_xml or getattr(sim, "_scene_xml_path", None)
        if resolved_scene is None and rt == RobotType.STRETCH3:
            from stretch_mujoco.utils import default_scene_xml_path

            resolved_scene = default_scene_xml_path
        if resolved_scene:
            set_scene_path(str(resolved_scene))
        if video_path and cameras:
            _video_recorder = RobotVideoRecorder(
                sim, cameras, video_path, fps=min(float(camera_hz), 15.0)
            )
            _video_recorder.start()

        # Collect available actuators
        actuators = [m.name for m in sim.Actuators.all()]
        base_actuators = [m.name for m in sim.Actuators.all() if m.is_base_actuator()]

        return {
            "success": True,
            "message": f"Robot '{robot_type}' started (headless={headless})",
            "robot_type": robot_type,
            "actuators": actuators,
            "base_actuators": base_actuators,
            "num_actuators": len(actuators),
            "video_path": video_path,
            "camera_hz": camera_hz,
            "scene_xml": str(resolved_scene) if resolved_scene else None,
        }
    except Exception as exc:
        _stop_video_recorder()
        if sim is not None:
            try:
                sim.stop()
            except Exception:
                pass
        return {"success": False, "message": f"robot_init failed: {exc}"}


def robot_stop() -> dict[str, Any]:
    """Stop the active robot simulator."""
    if not has_sim():
        return {"success": False, "message": "No active simulator"}
    try:
        video = _stop_video_recorder()
        get_sim().stop()
        set_sim(None)
        set_robot_type(None)
        return {"success": True, "message": "Simulator stopped", "video": video}
    except Exception as exc:
        return {"success": False, "message": str(exc)}


# ---------------------------------------------------------------------------
# Joint control
# ---------------------------------------------------------------------------


def robot_move_to(actuator_name: str, position: float) -> dict[str, Any]:
    """Command an absolute joint position target."""
    if not has_sim():
        return {"success": False, "message": "No active simulator — call robot_init first"}
    try:
        sim = get_sim()
        actuator = _resolve_actuator(sim, actuator_name)
        sim.move_to(actuator, float(position))
        return {
            "success": True,
            "message": f"move_to {actuator_name} -> {position}",
            "actuator": actuator_name,
            "target": float(position),
        }
    except Exception as exc:
        return {"success": False, "message": str(exc)}


def robot_move_by(actuator_name: str, delta: float) -> dict[str, Any]:
    """Command a relative joint position increment."""
    if not has_sim():
        return {"success": False, "message": "No active simulator — call robot_init first"}
    try:
        sim = get_sim()
        actuator = _resolve_actuator(sim, actuator_name)
        sim.move_by(actuator, float(delta))
        return {
            "success": True,
            "message": f"move_by {actuator_name} += {delta}",
            "actuator": actuator_name,
            "delta": float(delta),
        }
    except Exception as exc:
        return {"success": False, "message": str(exc)}


def robot_set_base_velocity(
    v_linear: float = 0.0,
    omega: float = 0.0,
    v_lateral: float = 0.0,
) -> dict[str, Any]:
    """Set the mobile base velocity.

    - *v_linear*: forward speed (m/s)
    - *omega*: angular speed (rad/s)
    - *v_lateral*: sideways speed (m/s) — only for omni-directional bases
    """
    if not has_sim():
        return {"success": False, "message": "No active simulator — call robot_init first"}
    try:
        get_sim().set_base_velocity(v_linear, omega, v_lateral)
        return {
            "success": True,
            "message": f"Base velocity set: v={v_linear}, ω={omega}, v_lat={v_lateral}",
        }
    except Exception as exc:
        return {"success": False, "message": str(exc)}


def robot_home() -> dict[str, Any]:
    """Move the robot to its home keyframe (if supported)."""
    if not has_sim():
        return {"success": False, "message": "No active simulator — call robot_init first"}
    try:
        get_sim().home()
        return {"success": True, "message": "Robot moved to home position"}
    except NotImplementedError:
        return {"success": False, "message": "home() not supported for this robot type"}
    except Exception as exc:
        return {"success": False, "message": str(exc)}


def robot_stow() -> dict[str, Any]:
    """Move the robot to its stow/retract keyframe (if supported)."""
    if not has_sim():
        return {"success": False, "message": "No active simulator — call robot_init first"}
    try:
        get_sim().stow()
        return {"success": True, "message": "Robot moved to stow position"}
    except NotImplementedError:
        return {"success": False, "message": "stow() not supported for this robot type"}
    except Exception as exc:
        return {"success": False, "message": str(exc)}


def robot_wait_until_at_setpoint(
    actuator_name: str,
    timeout: float = 5.0,
    position_tolerance: float = 0.05,
) -> dict[str, Any]:
    """Block until the given actuator reaches its commanded target."""
    if not has_sim():
        return {"success": False, "message": "No active simulator — call robot_init first"}
    try:
        sim = get_sim()
        actuator = _resolve_actuator(sim, actuator_name)
        reached = sim.wait_until_at_setpoint(
            actuator, timeout=timeout, position_tolerance=position_tolerance
        )
        return {
            "success": reached,
            "message": f"Setpoint {'reached' if reached else 'not reached (timeout)'} for {actuator_name}",
            "actuator": actuator_name,
            "reached": reached,
        }
    except Exception as exc:
        return {"success": False, "message": str(exc)}


# ---------------------------------------------------------------------------
# Status queries
# ---------------------------------------------------------------------------


def robot_get_status() -> dict[str, Any]:
    """Pull the current robot joint-state snapshot."""
    if not has_sim():
        return {"success": False, "message": "No active simulator — call robot_init first"}
    try:
        status = get_sim().pull_status()
        return {
            "success": True,
            "status": _status_to_dict(status),
        }
    except Exception as exc:
        return {"success": False, "message": str(exc)}


def robot_get_base_pose() -> dict[str, Any]:
    """Return the robot's base pose as (x, y, theta)."""
    if not has_sim():
        return {"success": False, "message": "No active simulator — call robot_init first"}
    try:
        x, y, theta = get_sim().get_base_pose()
        return {
            "success": True,
            "x": float(x),
            "y": float(y),
            "theta": float(theta),
        }
    except Exception as exc:
        return {"success": False, "message": str(exc)}


def robot_get_ee_pose() -> dict[str, Any]:
    """Return the 4×4 end-effector pose in world coordinates."""
    if not has_sim():
        return {"success": False, "message": "No active simulator — call robot_init first"}
    try:
        mat = get_sim().get_ee_pose()
        return {
            "success": True,
            "pose": _format_pose(mat),
        }
    except Exception as exc:
        return {"success": False, "message": str(exc)}


def robot_get_camera_data(camera_name: str = "") -> dict[str, Any]:
    """Pull camera data.

    When *camera_name* is provided, returns that specific camera's image
    metadata (shape, dtype).  Otherwise returns available camera names.
    """
    if not has_sim():
        return {"success": False, "message": "No active simulator — call robot_init first"}
    try:
        sim = get_sim()
        snapshot = sim.pull_camera_data()

        if camera_name:
            # Look up the camera enum
            for cam in sim.Cameras.all():
                if cam.name == camera_name or cam.camera_name_in_mjcf == camera_name:
                    data = snapshot.get_camera_data(cam, auto_correct_rgb=True)
                    return {
                        "success": True,
                        "camera": camera_name,
                        "shape": list(data.shape),
                        "dtype": str(data.dtype),
                    }
            return {"success": False, "message": f"Camera '{camera_name}' not found"}

        # List available cameras
        rgb_names = [c.name for c in sim.Cameras.rgb()]
        depth_names = [c.name for c in sim.Cameras.depth()]
        return {
            "success": True,
            "rgb_cameras": rgb_names,
            "depth_cameras": depth_names,
        }
    except Exception as exc:
        return {"success": False, "message": str(exc)}


# ---------------------------------------------------------------------------
# Grasping
# ---------------------------------------------------------------------------


def robot_attach_object(object_id: str) -> dict[str, Any]:
    """Attach a named object body to the gripper."""
    if not has_sim():
        return {"success": False, "message": "No active simulator — call robot_init first"}
    try:
        get_sim().attach_object_to_gripper(object_id)
        return {"success": True, "message": f"Attached '{object_id}' to gripper"}
    except Exception as exc:
        return {"success": False, "message": str(exc)}


def robot_release_object() -> dict[str, Any]:
    """Release the currently grasped object."""
    if not has_sim():
        return {"success": False, "message": "No active simulator — call robot_init first"}
    try:
        get_sim().release_grasped_object()
        return {"success": True, "message": "Released grasped object"}
    except Exception as exc:
        return {"success": False, "message": str(exc)}


# ---------------------------------------------------------------------------
# Introspection
# ---------------------------------------------------------------------------


def robot_list_actuators() -> dict[str, Any]:
    """List all actuators for the current robot, with metadata."""
    if not has_sim():
        return {"success": False, "message": "No active simulator — call robot_init first"}
    try:
        sim = get_sim()
        result = []
        for m in sim.Actuators.all():
            result.append(
                {
                    "name": m.name,
                    "value": m.value,
                    "type": m.actuator_type().value,
                    "is_base": m.is_base_actuator(),
                }
            )
        return {"success": True, "actuators": result}
    except Exception as exc:
        return {"success": False, "message": str(exc)}


def robot_list_cameras() -> dict[str, Any]:
    """List all cameras for the current robot."""
    if not has_sim():
        return {"success": False, "message": "No active simulator — call robot_init first"}
    try:
        sim = get_sim()
        result = []
        for c in sim.Cameras.all():
            result.append(
                {
                    "name": c.name,
                    "mjcf_name": c.camera_name_in_mjcf,
                    "is_rgb": c.is_rgb,
                    "is_depth": c.is_depth,
                }
            )
        return {"success": True, "cameras": result}
    except Exception as exc:
        return {"success": False, "message": str(exc)}


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------

ROBOT_TOOL_FUNCTIONS: dict[str, Any] = {
    "robot_init": robot_init,
    "robot_stop": robot_stop,
    "robot_move_to": robot_move_to,
    "robot_move_by": robot_move_by,
    "robot_set_base_velocity": robot_set_base_velocity,
    "robot_home": robot_home,
    "robot_stow": robot_stow,
    "robot_wait_until_at_setpoint": robot_wait_until_at_setpoint,
    "robot_get_status": robot_get_status,
    "robot_get_base_pose": robot_get_base_pose,
    "robot_get_ee_pose": robot_get_ee_pose,
    "robot_get_camera_data": robot_get_camera_data,
    "robot_attach_object": robot_attach_object,
    "robot_release_object": robot_release_object,
    "robot_list_actuators": robot_list_actuators,
    "robot_list_cameras": robot_list_cameras,
}
