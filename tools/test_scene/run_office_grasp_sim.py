#!/usr/bin/env python3
"""Batch scripted-geometry grasp test over successful office navigation end states.

For every navigation report in ``--nav-report-dir`` matching
``<scene_id>_*_grasp_site.json`` with ``"reached": true``, this script:

  1. Loads the scene XML once, verifies gravity/dynamics on the target object,
     and boosts contact friction (object + gripper fingertips) the same way
     ``examples/collect_random_object_grasp_episodes.py`` does for reliable
     scripted grasps.
  2. Starts the simulator at the navigation report's ``final_pose``.
  3. Raises/retracts the arm during the pivot, re-aims the head, and captures a
     fresh D435i RGB-D + D405 RGB observation.
  4. Sends RGB-D to the existing GraspGen service, transforms/ranks candidates
     with ``StretchGraspIK``, and picks the best reachable grasp.
  5. Drives to a pregrasp standoff, then re-observes with the D405 wrist
     camera and re-solves the grasp target from that close-range look before
     committing to the final approach (the head-camera estimate above is
     taken from across the room and is not accurate enough on its own).
  6. Executes pregrasp -> align -> close -> contact check -> in-place lift,
     then holds the lifted pose while judging success from physical contact
     (friction + gravity, not a kinematic attach) and final lift height.

Example::

  MUJOCO_GL=egl python tools/test_scene/run_office_grasp_sim.py \
      --scene 1 --output-dir output/office_grasp_sim
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")
import cv2
import imageio.v2 as imageio
import mujoco
import numpy as np

from stretch_mujoco import StretchMujocoSimulator
from stretch_mujoco.enums.actuators import Actuators
from stretch_mujoco.enums.stretch_cameras import StretchCameras
from stretch_mujoco.grasp_task import GRASP_FRICTION
from stretch_mujoco.graspgen import (
    RGBDGraspClient,
    StretchGraspIK,
    d435i_rotated_camera_intrinsics,
    rotated_d435i_optical_pose,
)
from stretch_mujoco.graspgen.calibration import world_to_base_pose

try:
    from test_office_navigation import _is_descendant_of, _scene_paths
except ImportError:
    from tools.test_scene.test_office_navigation import _is_descendant_of, _scene_paths


NAV_LIFT_CLEARANCE = 1.1
# Minimum base-to-object distance to hold before the in-place grasp-stance
# pivot: comfortably past the arm's ~0.5m reach standoff, so the retracted
# wrist bracket's sweep radius during the rotation clears the object.
SAFE_PIVOT_DISTANCE_M = 0.65
# How far short of the head-camera-estimated grasp point to stop the arm
# before taking the close-range D405 wrist-camera look: far enough that the
# object is still comfortably past GraspGen's min_depth and the gripper isn't
# already brushing it, close enough for a materially better depth estimate
# than the head camera got from across the room.
WRIST_REFINE_STANDOFF_M = 0.18
MOTION_SPEED = 1.0
GRIPPER_OPEN = 0.56
GRIPPER_CLOSE = -0.15
MIN_LIFT_M = 0.15
CAMERA_HZ = 10.0
# Sample rate for the continuous background debug recording (see
# ``_DebugRecorder.start_continuous``), independent of the sparse
# named-stage frames captured at each scripted checkpoint below --
# a handful of keyframes can't show a gradual slip mid-lift, but a
# steady real-time recording of the whole attempt can.
CONTINUOUS_DEBUG_HZ = 5.0
# Every office-scene object lies flat on its own surface, so a side-on
# grasp never needs the wrist tilted or rolled to match some object-specific
# orientation -- forcing both flat (rather than whatever GraspGen/IK solved)
# avoids the fingertips dipping toward the desk while approaching, and any
# resulting position drift is corrected by ``_align_grasp_center``'s live
# ee_pose feedback, which runs after the wrist is set.
FLAT_WRIST_PITCH = 0.0
FLAT_WRIST_ROLL = 0.0
# After ``_align_grasp_center`` centers on the grasp point, push the gripper
# this much farther in along the grasp approach direction before closing.
# This is a real world-space forward correction (not a blind arm-joint bump),
# so the fingers close with a controlled amount of remaining depth margin.
POST_ALIGN_PUSH_M = 0.05
# Approximate wrist_yaw-axis-to-fingertip lever arm (metres), matching the
# scale ``_align_grasp_center`` already uses to turn a lateral position error
# into a wrist_yaw increment (``error_x / 0.26``). Reused below to convert a
# fixed lateral distance into the equivalent wrist_yaw offset.
WRIST_YAW_LEVER_ARM_M = 0.26
# Fixed pregrasp bias corrections, applied while holding the current
# heading/orientation via ``_shift_ee_world`` (fixed, IK-solved world-space
# displacements -- see also POST_ALIGN_PUSH_M above): PREGRASP_LATERAL_SHIFT_M
# nudges the gripper sideways (base-local +X, the only lateral DOF once the
# base is in its grasp stance) right before the "05_pregrasp" stance is
# measured/captured; PUSH_IN_DOWNWARD_SHIFT_M drops it a fixed amount further
# in world -Z right before the "06b_push_in" capture. Flip the sign of either
# if debug frames show the correction going the wrong way.
PREGRASP_LATERAL_SHIFT_M = 0.015
PUSH_IN_DOWNWARD_SHIFT_M = 0.03
# Reject a GraspGen candidate whose solved world-frame grasp point is farther
# than this from the object's actual position (from ``pull_grasp_metrics``).
# GraspGen has no way to mask out the robot's own body from the RGB-D image,
# so a stray/self-occluded detection can otherwise "solve" cleanly (low IK
# error, decent confidence) while pointing at the robot's own arm.
MAX_GRASP_POINT_OBJECT_DISTANCE_M = 0.2


def _default_prompt(asset: dict, object_id: str) -> str:
    """Turn generated ids such as ``001_bottle_022`` into a useful prompt."""
    name = str(asset.get("asset_id") or asset.get("name") or object_id)
    if name.endswith("_" + object_id.rsplit("_", 1)[-1]):
        name = name.rsplit("_", 1)[0]
    if name[:4].isdigit() and name[3] == "_":
        name = name[4:]
    return name.replace("-", " ").replace("_", " ").strip()


def _project_target(sim: StretchMujocoSimulator, target_world: np.ndarray,
                    camera_k: np.ndarray | None = None) -> tuple[float, float, float]:
    """Return target pixel (u, v) and camera-frame depth after image rotation."""
    pose = rotated_d435i_optical_pose(sim.get_link_pose("camera_color_optical_frame"))
    point = (np.linalg.inv(pose) @ np.r_[np.asarray(target_world, dtype=float), 1.0])[:3]
    if camera_k is None:
        camera_k = d435i_rotated_camera_intrinsics()
        fx, fy, cx, cy = camera_k[0, 0], camera_k[1, 1], camera_k[0, 2], camera_k[1, 2]
    else:
        fx, fy, cx, cy = camera_k[0, 0], camera_k[1, 1], camera_k[0, 2], camera_k[1, 2]
    if point[2] <= 1e-5:
        return float("nan"), float("nan"), float(point[2])
    return float(fx * point[0] / point[2] + cx), float(fy * point[1] / point[2] + cy), float(point[2])


def _debug_image(rgb: np.ndarray, depth: np.ndarray, *, u: float, v: float,
                 stage: str, detail: str = "") -> np.ndarray:
    rgb = np.asarray(rgb).copy()
    depth = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0.05)
    depth_vis = np.zeros((*depth.shape, 3), dtype=np.uint8)
    if valid.any():
        scaled = np.clip((depth - 0.05) / 2.95 * 255.0, 0, 255).astype(np.uint8)
        depth_vis = cv2.applyColorMap(255 - scaled, cv2.COLORMAP_TURBO)
        depth_vis[~valid] = 0
    canvas = np.concatenate([rgb, depth_vis], axis=1)
    if np.isfinite(u) and np.isfinite(v):
        cv2.drawMarker(canvas, (round(u), round(v)), (0, 255, 0), cv2.MARKER_CROSS, 18, 2)
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 30), (20, 20, 20), -1)
    cv2.putText(canvas, f"{stage}  {detail[:100]}", (8, 21), cv2.FONT_HERSHEY_SIMPLEX,
                0.48, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def _compose_debug_frame(sim: StretchMujocoSimulator, target: np.ndarray, stage: str,
                          camera_k: np.ndarray | None, detail: str) -> tuple[np.ndarray, dict[str, float]]:
    """Build one head(RGB+depth) + wrist(RGB) debug panel. Shared by the
    named-stage capture and the continuous background recorder below so
    both produce frames in the same layout."""
    u, v, z = _project_target(sim, target, camera_k)
    cameras = sim.pull_camera_data()
    rgb = cameras.get_camera_data(StretchCameras.cam_d435i_rgb, auto_rotate=True,
                                  auto_correct_rgb=False)
    depth = cameras.get_camera_data(StretchCameras.cam_d435i_depth, auto_rotate=True)
    frame = _debug_image(rgb, depth, u=u, v=v, stage=stage,
                         detail=f"target=({u:.0f},{v:.0f}) z={z:.2f} {detail}")
    # The head camera above is aimed once, before the grasp-stance pivot,
    # and never re-aimed -- past that point it points wherever the pivot
    # left it, not at the gripper. The wrist camera is rigidly mounted at
    # the gripper, so it stays relevant through pregrasp/align/close/lift
    # regardless of base yaw; append it so those stages stay legible.
    wrist_rgb = np.asarray(
        cameras.get_camera_data(StretchCameras.cam_d405_rgb, auto_rotate=True, auto_correct_rgb=False)
    )
    wrist_width = max(1, round(wrist_rgb.shape[1] * frame.shape[0] / wrist_rgb.shape[0]))
    wrist_panel = cv2.resize(wrist_rgb, (wrist_width, frame.shape[0]))
    frame = np.concatenate([frame, wrist_panel], axis=1)
    return frame, {"u": u, "v": v, "camera_z": z}


class _DebugRecorder:
    def __init__(self, directory: Path, enabled: bool) -> None:
        self.directory, self.enabled, self.frames = directory, enabled, []
        self.target = np.zeros(3)
        self.stage = "start"
        self._continuous_frames: list[np.ndarray] = []
        self._continuous_thread: threading.Thread | None = None
        self._continuous_stop = threading.Event()
        self._continuous_hz = CONTINUOUS_DEBUG_HZ
        if enabled:
            directory.mkdir(parents=True, exist_ok=True)

    def capture(self, sim: StretchMujocoSimulator, target: np.ndarray, stage: str,
                camera_k: np.ndarray | None = None, detail: str = "") -> dict[str, float]:
        self.target, self.stage = np.asarray(target, dtype=float), stage
        if not self.enabled:
            u, v, z = _project_target(sim, target, camera_k)
            return {"u": u, "v": v, "camera_z": z}
        frame, projection = _compose_debug_frame(sim, target, stage, camera_k, detail)
        self.frames.append(frame)
        cv2.imwrite(str(self.directory / f"{len(self.frames):02d}_{stage.lower()}.png"),
                    cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        return projection

    def start_continuous(self, sim: StretchMujocoSimulator, hz: float = CONTINUOUS_DEBUG_HZ) -> None:
        """Record the whole attempt at a steady real-time rate in a background
        thread, independent of the sparse named-stage captures -- a handful
        of keyframes can hide a gradual slip (e.g. during the friction-held
        retreat/tilt/lift), while this shows every intermediate frame."""
        if not self.enabled or self._continuous_thread is not None:
            return
        self._continuous_hz = hz
        self._continuous_stop.clear()
        start_time = time.monotonic()

        def _loop() -> None:
            period = 1.0 / hz
            while not self._continuous_stop.is_set():
                loop_started = time.monotonic()
                try:
                    if sim.is_running():
                        frame, _ = _compose_debug_frame(
                            sim, self.target, self.stage, None,
                            f"t={time.monotonic() - start_time:.1f}s",
                        )
                        self._continuous_frames.append(frame)
                except Exception:
                    pass
                elapsed = time.monotonic() - loop_started
                self._continuous_stop.wait(max(0.0, period - elapsed))

        self._continuous_thread = threading.Thread(target=_loop, daemon=True)
        self._continuous_thread.start()

    def stop_continuous(self) -> None:
        if self._continuous_thread is None:
            return
        self._continuous_stop.set()
        self._continuous_thread.join(timeout=5.0)
        self._continuous_thread = None

    def close(self, name: str) -> None:
        self.stop_continuous()
        if self.enabled and self.frames:
            imageio.mimsave(self.directory / f"{name}.gif", self.frames, duration=0.7)
        if self.enabled and self._continuous_frames:
            imageio.mimsave(
                self.directory / f"{name}_full.gif",
                self._continuous_frames,
                duration=1.0 / self._continuous_hz,
            )


def _json_compatible(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    return value


def _find_asset(manifest: dict, object_id: str) -> dict:
    for asset in manifest.get("assets", []):
        if asset.get("object_id") == object_id:
            return asset
    raise ValueError(f"object_id {object_id!r} not found in manifest assets")


def _boost_friction(model: mujoco.MjModel, body_names: list[str], friction: float) -> int:
    """Raise sliding friction on every geom in the named bodies' subtrees to
    at least ``friction`` -- takes ``max(existing, friction)`` rather than
    overwriting, so a geom that already ships with a *higher* default never
    gets weakened.

    This matters concretely for the fingertip pads: each of
    ``rubber_tip_left``/``right`` carries two overlapping collision geoms,
    one at ``priority=0`` and one at ``priority=1`` with a materially higher
    default sliding friction (mu=2.0 vs. the object's own ~0.5). MuJoCo's
    contact-parameter combination rule uses the higher-*priority* geom's
    friction *exclusively* when two contacting geoms have different
    priority (verified directly against ``data.contact[i].friction`` --
    the object geom's own friction is not blended in at all for those
    contacts). A blind overwrite to ``friction`` therefore doesn't "boost"
    that pad geom, it cuts its contact friction roughly in half (2.0 ->
    0.9), which is enough on its own to make an apparently well-closed grip
    slip once the arm moves. Mirrors
    ``collect_random_object_grasp_episodes.py::build_episode_model``.
    """
    changed = 0
    for name in body_names:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id < 0:
            continue
        for geom_id in range(model.ngeom):
            if _is_descendant_of(model, int(model.geom_bodyid[geom_id]), body_id):
                model.geom_friction[geom_id, 0] = max(model.geom_friction[geom_id, 0], friction)
                changed += 1
    return changed


def _prepare_model(xml_path: Path, object_id: str, grasp_site: str) -> dict:
    """Load the scene once, verify physics, patch friction, and compute the
    static grasp-site offset from the object's body origin."""
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, object_id)
    if body_id < 0:
        raise ValueError(f"object body not found in XML: {object_id}")
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, grasp_site)
    if site_id < 0:
        raise ValueError(f"grasp site not found in XML: {grasp_site}")

    jnt_adr = int(model.body_jntadr[body_id])
    jnt_num = int(model.body_jntnum[body_id])
    has_freejoint = jnt_num == 1 and jnt_adr >= 0 and int(model.jnt_type[jnt_adr]) == int(
        mujoco.mjtJoint.mjJNT_FREE
    )
    gravity_z = float(model.opt.gravity[2])
    subtree_mass = float(model.body_subtreemass[body_id])
    grasp_offset = (data.site(site_id).xpos - data.xpos[body_id]).copy()

    friction_geoms_patched = _boost_friction(
        model, [object_id, "rubber_tip_left", "rubber_tip_right"], GRASP_FRICTION
    )

    return {
        "model": model,
        "has_freejoint": has_freejoint,
        "gravity_z": gravity_z,
        "subtree_mass_kg": subtree_mass,
        "grasp_offset": grasp_offset,
        "friction_geoms_patched": friction_geoms_patched,
    }


def _wait(
    sim: StretchMujocoSimulator, actuator: Actuators, timeout: float = 20.0, tolerance: float = 0.02
) -> bool:
    return sim.wait_until_at_setpoint(actuator, timeout=timeout, position_tolerance=tolerance)


def _rotate_to_yaw(
    sim: StretchMujocoSimulator, target_yaw: float, timeout: float = 12.0, tolerance: float = 0.03
) -> bool:
    """Rotate the base in place to face ``target_yaw``, mirroring the final
    in-place alignment controller in ``run_office_navigation_sim.py``."""
    started = time.monotonic()
    previous_w = 0.0
    while time.monotonic() - started < timeout:
        _, _, yaw = sim.get_base_pose()
        yaw_error = float(np.arctan2(np.sin(target_yaw - yaw), np.cos(target_yaw - yaw)))
        if abs(yaw_error) <= tolerance and abs(previous_w) <= 0.05:
            sim.set_base_velocity(0.0, 0.0)
            return True
        desired_w = float(np.clip(2.2 * yaw_error, -0.8, 0.8))
        previous_w += float(np.clip(desired_w - previous_w, -0.04, 0.04))
        sim.set_base_velocity(0.0, previous_w)
        time.sleep(1.0 / 30.0)
    sim.set_base_velocity(0.0, 0.0)
    return False


def _aim_head_at(sim: StretchMujocoSimulator, target_world: np.ndarray) -> None:
    """Point the RGB-D head at a world point using image-space feedback.

    Must be called before pivoting into the grasp stance: see the comment
    above the ``_aim_head_at``/GraspGen call site in ``run_object`` for why
    aiming (and photographing) after the pivot self-occludes the object.
    """
    target_world = np.asarray(target_world, dtype=float)
    for _ in range(4):
        u, v, z = _project_target(sim, target_world)
        status = sim.pull_status()
        # Estimate each joint's image-space Jacobian. This avoids assuming the
        # optical frame and actuator sign conventions are identical.
        if np.isfinite(u) and z > 0:
            pan0 = float(status.head_pan.pos)
            step = 0.06
            sim.move_to(Actuators.head_pan, float(np.clip(pan0 + step, -4.04, 1.73)))
            _wait(sim, Actuators.head_pan, timeout=4.0)
            u1, _, _ = _project_target(sim, target_world)
            derivative = (u1 - u) / step if np.isfinite(u1) else 0.0
            sim.move_to(Actuators.head_pan, float(np.clip(
                pan0 - (u - 120.0) / derivative if abs(derivative) > 1e-3 else pan0,
                -4.04, 1.73)))
            _wait(sim, Actuators.head_pan, timeout=6.0)

        u, v, z = _project_target(sim, target_world)
        status = sim.pull_status()
        if np.isfinite(v) and z > 0:
            tilt0 = float(status.head_tilt.pos)
            step = 0.06
            sim.move_to(Actuators.head_tilt, float(np.clip(tilt0 + step, -1.53, 0.79)))
            _wait(sim, Actuators.head_tilt, timeout=4.0)
            _, v1, _ = _project_target(sim, target_world)
            derivative = (v1 - v) / step if np.isfinite(v1) else 0.0
            sim.move_to(Actuators.head_tilt, float(np.clip(
                tilt0 - (v - 212.0) / derivative if abs(derivative) > 1e-3 else tilt0,
                -1.53, 0.79)))
            _wait(sim, Actuators.head_tilt, timeout=6.0)
        if np.isfinite(u) and np.isfinite(v) and abs(u - 120) < 18 and abs(v - 212) < 18:
            break


def _move_base_forward(
    sim: StretchMujocoSimulator, dx: float, speed: float = 0.08, timeout: float = 8.0
) -> None:
    """Drive the base straight along its current heading by ``dx`` metres,
    realizing ``StretchGraspIK``'s ``base_x`` component of a solve.

    Adapted from ``examples/collect_random_grasp_episodes.py::_move_base_slow``,
    which tracks progress via raw world-X on the assumption the base heading
    is always 0; here the base can be at any heading, so progress is tracked
    as Euclidean distance traveled instead."""
    if abs(dx) < 0.005:
        return
    start_x, start_y, _ = sim.get_base_pose()
    sim.set_base_velocity(float(np.copysign(speed, dx)), 0.0)
    deadline = time.monotonic() + timeout
    traveled = 0.0
    while traveled < abs(dx) - 0.005 and time.monotonic() < deadline:
        time.sleep(0.02)
        x, y, _ = sim.get_base_pose()
        traveled = float(np.hypot(x - start_x, y - start_y))
    sim.set_base_velocity(0.0, 0.0)
    time.sleep(0.2)


def _align_grasp_center(
    sim: StretchMujocoSimulator, target_world: np.ndarray, iterations: int = 3
) -> None:
    """Closed-loop residual correction, adapted from
    ``collect_random_object_grasp_episodes.py::_align_grasp_center`` for a base
    that is not axis-aligned with world: the position error is computed in
    world frame (from live grasp metrics) and must be rotated into the base's
    local axes before being applied as arm/wrist_yaw/lift increments."""
    for _ in range(iterations):
        error_world = np.asarray(target_world, dtype=float) - sim.get_ee_pose()[:3, 3]
        base_yaw = float(sim.get_base_pose()[2])
        cosine, sine = np.cos(-base_yaw), np.sin(-base_yaw)
        error_x = cosine * error_world[0] - sine * error_world[1]
        error_y = sine * error_world[0] + cosine * error_world[1]
        error_z = error_world[2]
        status = sim.pull_status()
        sim.move_to(Actuators.lift, float(np.clip(status.lift.pos + error_z, 0.0, 1.1)))
        sim.move_to(Actuators.arm, float(np.clip(status.arm.pos - error_y, 0.0, 0.52)))
        sim.move_to(
            Actuators.wrist_yaw,
            float(
                np.clip(status.wrist_yaw.pos + np.clip(error_x / 0.26, -0.04, 0.04), -1.39, 4.42)
            ),
        )
        _wait(sim, Actuators.lift, timeout=4.0)
        _wait(sim, Actuators.arm, timeout=4.0)
        _wait(sim, Actuators.wrist_yaw, timeout=4.0)
        time.sleep(0.2)


def _align_lateral_height(
    sim: StretchMujocoSimulator, target_world: np.ndarray, iterations: int = 3
) -> None:
    """Left-right (wrist_yaw) + height (lift) closed-loop correction only --
    unlike ``_align_grasp_center``, the arm (forward/depth) axis is left
    alone here on purpose.

    Meant to run right after a wrist-refine solve, *before* the arm extends
    the rest of the way to the final grasp target. The refine solve is a
    single open-loop IK estimate that's never checked against live
    kinematics; if it's laterally off (say the object sits further left than
    estimated), driving the arm straight to that target risks a finger
    clipping/knocking the object before ``_align_grasp_center`` -- which only
    runs *after* the arm is already fully extended -- ever gets a chance to
    correct it. Lateral and height error are rotations/translations
    independent of how far the arm has extended (see ``run_object``'s
    approach-axis alignment), so it's safe to converge these two axes first,
    while the arm is still short at the standoff reach and can't yet touch
    anything.
    """
    for _ in range(iterations):
        error_world = np.asarray(target_world, dtype=float) - sim.get_ee_pose()[:3, 3]
        base_yaw = float(sim.get_base_pose()[2])
        cosine, sine = np.cos(-base_yaw), np.sin(-base_yaw)
        error_x = cosine * error_world[0] - sine * error_world[1]
        error_z = error_world[2]
        status = sim.pull_status()
        sim.move_to(Actuators.lift, float(np.clip(status.lift.pos + error_z, 0.0, 1.1)))
        sim.move_to(
            Actuators.wrist_yaw,
            float(np.clip(
                status.wrist_yaw.pos + np.clip(error_x / WRIST_YAW_LEVER_ARM_M, -0.04, 0.04),
                -1.39, 4.42,
            )),
        )
        _wait(sim, Actuators.lift, timeout=4.0)
        _wait(sim, Actuators.wrist_yaw, timeout=4.0)
        time.sleep(0.2)


def _infer_world_grasp_poses(
    client: RGBDGraspClient,
    rgb: np.ndarray,
    depth: np.ndarray,
    camera_k: np.ndarray,
    camera_pose_world: np.ndarray,
    prompt: str,
    approach_camera: np.ndarray,
    *,
    max_approach_angle_deg: float = 45.0,
    min_depth: float = 0.05,
    max_depth: float = 3.0,
) -> tuple[np.ndarray | None, np.ndarray | None, dict]:
    """Run one GraspGen inference and lift its camera-frame poses into world
    frame using ``camera_pose_world``, the camera's world pose at capture
    time. Returns ``(None, None, response)`` if GraspGen found nothing --
    the world-frame poses are otherwise independent of the camera that
    produced them, so this is shared by both the head-camera and wrist-camera
    passes in ``run_object``."""
    response = client.infer(
        rgb,
        depth,
        camera_k,
        prompt,
        candidate_top_k=200,
        desired_approach_direction=tuple(approach_camera),
        max_approach_angle_deg=max_approach_angle_deg,
        approach_allow_opposite=True,
        min_depth=min_depth,
        max_depth=max_depth,
    )
    if response.get("status") != "ok" or not response.get("num_grasps"):
        return None, None, response
    camera_tcp_poses = np.asarray(response["grasp_tcp_poses"], dtype=float)
    world_tcp_poses = np.asarray([camera_pose_world @ pose for pose in camera_tcp_poses])
    confidences = np.asarray(response["confidences"], dtype=float)
    return world_tcp_poses, confidences, response


def _select_reachable_candidate(
    grasp_ik: StretchGraspIK,
    world_tcp_poses: np.ndarray,
    confidences: np.ndarray,
    base_pose: tuple[float, float, float],
    sanity_point: np.ndarray,
    *,
    max_sanity_distance: float = MAX_GRASP_POINT_OBJECT_DISTANCE_M,
    max_wrist_yaw: float = 0.7,
):
    """Rank IK-reachable candidates -- including each pose's 180-degree
    approach-axis symmetry twin, since a parallel-jaw gripper can realize the
    same grasp with the wrist rolled by pi -- and return the best one whose
    solved world point is still near ``sanity_point``. That sanity check
    guards against GraspGen "solving" a stray/self-occluded detection of the
    robot's own arm cleanly. Returns ``(None, best_off_target_distance_m)``
    if no candidate is both reachable and on-target."""
    base_from_world = world_to_base_pose(base_pose)
    base_tcp_poses = np.asarray([base_from_world @ pose for pose in world_tcp_poses])
    symmetry = np.diag([-1.0, -1.0, 1.0, 1.0])
    candidates = grasp_ik.rank_candidates(
        np.asarray([variant for pose in base_tcp_poses for variant in (pose, pose @ symmetry)]),
        np.repeat(confidences, 2),
    )
    candidates = [c for c in candidates if abs(c.joints[3]) <= max_wrist_yaw]
    if not candidates:
        return None, None
    on_target = [
        c for c in candidates
        if np.linalg.norm(world_tcp_poses[c.index // 2][:3, 3] - sanity_point)
        <= max_sanity_distance
    ]
    if not on_target:
        best_off_target = float(min(
            np.linalg.norm(world_tcp_poses[c.index // 2][:3, 3] - sanity_point)
            for c in candidates
        ))
        return None, best_off_target
    return on_target[0], None


def _solve_standoff_pose(
    grasp_ik: StretchGraspIK,
    grasp_pose_world: np.ndarray,
    base_pose: tuple[float, float, float],
    final_joints: np.ndarray,
    standoff_m: float,
) -> np.ndarray | None:
    """IK-solve a pregrasp standoff pose: the same orientation as
    ``grasp_pose_world`` but translated ``standoff_m`` back along its own
    approach axis (local Z -- the same axis the gripper's 180-degree
    approach-axis symmetry rotates about, see ``run_object``'s ``symmetry``
    use). GraspGen's own ``approach_allow_opposite`` option means a
    candidate's Z can point either wrist-to-object or object-to-wrist, so
    rather than assume a sign, solve both directions and keep whichever one
    actually *reduces* arm reach versus ``final_joints`` -- that is, by
    construction, the one that backs the wrist away from the object,
    regardless of which way Z happens to point for this candidate. Returns
    ``None`` if neither direction yields a valid, reach-reducing solve.
    """
    approach_axis = grasp_pose_world[:3, 2]
    norm = float(np.linalg.norm(approach_axis))
    if norm < 1e-6:
        return None
    approach_axis = approach_axis / norm
    base_from_world = world_to_base_pose(base_pose)
    best_joints: np.ndarray | None = None
    for sign in (1.0, -1.0):
        pose = grasp_pose_world.copy()
        pose[:3, 3] = grasp_pose_world[:3, 3] + sign * approach_axis * standoff_m
        candidate_joints, position_error, _ = grasp_ik.solve(
            base_from_world @ pose, initial=final_joints
        )
        if position_error > 0.02 or candidate_joints[2] >= final_joints[2] - 1e-3:
            continue
        if best_joints is None or candidate_joints[2] < best_joints[2]:
            best_joints = candidate_joints
    return best_joints


def _shift_ee_world(
    sim: StretchMujocoSimulator, grasp_ik: StretchGraspIK, delta_world: np.ndarray,
) -> bool:
    """Move the end effector by a fixed world-space Cartesian displacement
    ``delta_world`` (metres), solved via IK from wherever it currently is.

    This is what makes the displacement "absolute": a fixed amount applied
    on top of the current pose regardless of how it was reached, rather than
    a bare joint-space bump (e.g. ``lift.pos -= shift``) that only equals the
    intended Cartesian delta when the mast happens to be exactly world-Z and
    no other joint has moved since -- and unlike a joint bump, this is also
    validated for IK reachability the same way the forward push is, instead
    of being applied blindly.

    Seeds the solve with the robot's *actual current* configuration (base_x
    is 0 by definition here, since ``target_base`` is already expressed
    relative to the current base pose) rather than ``StretchGraspIK.solve``'s
    generic ``DEFAULT_INITIAL`` rest pose. Without that seed, the nonlinear
    solve is free to land on a distant, unrelated joint combination that
    also happens to reach ``target_base`` -- e.g. trading arm reach for some
    nonzero ``base_x`` -- which this function used to silently discard,
    producing a "successful" (low position-error) solve whose *executed*
    arm/lift/wrist_yaw motion didn't actually land anywhere near the
    intended small nudge (confirmed directly: repeated small pushes/shifts
    at an arm position already close to its 0.52 reach limit produced no
    visible motion, or moved the wrong way, because the solver's real
    answer relied on a ``base_x`` this function never drove the base by).
    """
    delta_world = np.asarray(delta_world, dtype=float)
    if not np.any(delta_world):
        return False
    ee_pose = np.asarray(sim.get_ee_pose(), dtype=float)
    desired_ee = ee_pose.copy()
    desired_ee[:3, 3] += delta_world
    base_pose = sim.get_base_pose()
    target_base = world_to_base_pose(base_pose) @ desired_ee
    status = sim.pull_status()
    current_joints = np.array([
        0.0,
        float(status.lift.pos),
        float(status.arm.pos),
        float(status.wrist_yaw.pos),
        float(status.wrist_pitch.pos),
        float(status.wrist_roll.pos),
    ])
    joints, position_error, _ = grasp_ik.solve(target_base, initial=current_joints)
    if position_error > 0.025:
        return False

    if abs(float(joints[0])) > 0.005:
        _move_base_forward(sim, float(joints[0]))
    sim.move_to(Actuators.arm, float(np.clip(joints[2], 0.0, 0.52)))
    sim.move_to(Actuators.lift, float(np.clip(joints[1], 0.0, 1.1)))
    sim.move_to(Actuators.wrist_yaw, float(np.clip(joints[3], -1.39, 4.42)))
    _wait(sim, Actuators.arm, timeout=6.0, tolerance=0.01)
    _wait(sim, Actuators.lift, timeout=6.0, tolerance=0.01)
    _wait(sim, Actuators.wrist_yaw, timeout=6.0, tolerance=0.01)
    time.sleep(0.2)
    return True


def _push_in_along_approach(
    sim: StretchMujocoSimulator,
    grasp_ik: StretchGraspIK,
    push_m: float,
) -> float:
    """Push the gripper forward by a fixed ``push_m`` along this codebase's
    fixed "reach into the object" direction -- base-local -Y in world space
    (the same convention as ``approach_base`` in ``run_object``/
    ``_wrist_refine_grasp``) -- from wherever ``_align_grasp_center`` already
    left it, *not* however much distance remains to the nominal grasp point.
    This step exists specifically because GraspGen/IK's depth estimate is
    frequently a slight underestimate: it's meant to deliberately overshoot
    past the estimated surface by a fixed margin so the fingers close on the
    object instead of on air, the same way the original blind joint-space
    ``arm.pos += push_m`` bump did, just expressed in Cartesian space so it
    works regardless of base heading.

    Deliberately does *not* push along the gripper's own approach axis (a
    candidate's solved ``grasp_pose_world`` Z, or the live ``ee_pose`` Z at
    the current, possibly ``FLAT_WRIST_PITCH``-forced orientation): that axis
    is not reliably "forward into the object" at all here. Confirmed directly
    -- at this URDF's ``FLAT_WRIST_PITCH = 0.0``, the gripper's Z axis is
    ``[0, 0, 1]``, i.e. *vertical*, not the flat/forward orientation the name
    suggests (that instead needs ``wrist_pitch = -pi/2``). Pushing "along
    approach" at that orientation is really a near-pure vertical push: a
    milk-box run measured it moving the gripper +2.6cm *up* for a 3cm push,
    which overwhelmed the separate ``PUSH_IN_DOWNWARD_SHIFT_M`` correction
    and left the fingers short of the object at close -- visible in debug
    GIFs as "the push does nothing" (no forward motion at all, since none of
    it was ever horizontal) plus an unexplained rise right before close. The
    base-forward direction has none of that ambiguity: it's fixed, always
    horizontal, and is exactly the axis ``_align_grasp_center`` already used
    to line the gripper up on the target.

    (An earlier version of this function instead computed how much distance
    remained to the candidate pose's own translation and capped the push to
    that -- but ``_align_grasp_center`` already converges the end effector
    onto that same point, so the "remaining" distance was consistently near
    zero and the push collapsed to ~0.004m regardless of ``push_m``, visibly
    not moving the arm at all in debug GIFs.)
    """
    if push_m <= 0:
        return 0.0
    base_yaw = float(sim.get_base_pose()[2])
    approach = np.array([np.sin(base_yaw), -np.cos(base_yaw), 0.0])
    return push_m if _shift_ee_world(sim, grasp_ik, approach * push_m) else 0.0


def _wrist_refine_grasp(
    sim: StretchMujocoSimulator,
    grasp_ik: StretchGraspIK,
    args: argparse.Namespace,
    scene_id: str,
    object_id: str,
    prompt: str,
    fallback_grasp_point: np.ndarray,
) -> tuple[np.ndarray | None, np.ndarray | None, dict[str, Any]]:
    """Take a close-range D405 wrist-camera look from the current pregrasp
    standoff stance and re-solve the grasp target from it.

    The head-camera (D435i) estimate ``fallback_grasp_point`` was taken from
    across the room before the grasp-stance pivot; by the time the arm is at
    the standoff stance the gripper's own D405 is centimeters from the
    object with a much better-conditioned depth estimate, so re-running
    GraspGen here is what actually fixes leftover lateral/depth error instead
    of just tracking the original (possibly wrong) target more precisely.

    Unlike the D435i, the D405's MuJoCo image is not rotated in software, so
    no ``rotated_d435i_optical_pose``-style correction is needed for its
    intrinsics or link pose.

    Returns ``(None, None, diag)`` if GraspGen finds nothing usable at this
    range or every candidate is unreachable/off-target -- the caller should
    fall back to its existing estimate rather than abort the attempt.
    """
    diag: dict[str, Any] = {"wrist_refine_used": False}
    cameras = sim.pull_camera_data()
    rgb = cameras.get_camera_data(
        StretchCameras.cam_d405_rgb, auto_rotate=True, auto_correct_rgb=False
    )
    depth = cameras.get_camera_data(StretchCameras.cam_d405_depth, auto_rotate=True)
    camera_k = np.asarray(cameras.cam_d405_K, dtype=np.float32)
    if rgb.shape[:2] != depth.shape or camera_k.shape != (3, 3):
        diag["wrist_refine_skip_reason"] = "camera_data_inconsistent"
        return None, None, diag

    camera_pose_world = sim.get_link_pose("gripper_camera_color_optical_frame")
    base_pose = sim.get_base_pose()
    base_from_world = world_to_base_pose(base_pose)
    # The base is already in its final grasp-stance yaw at this point (the
    # pivot happened before pregrasp started), so -- unlike the D435i's
    # "future base pose" hypothetical -- this is just the current heading:
    # the arm's local -Y axis already faces the object.
    approach_base = np.array([0.0, -1.0, 0.0], dtype=float)
    approach_camera = (base_from_world[:3, :3] @ camera_pose_world[:3, :3]).T @ approach_base

    with RGBDGraspClient(args.graspgen_host, args.graspgen_port) as client:
        world_tcp_poses, confidences, response = _infer_world_grasp_poses(
            client, rgb, depth, camera_k, camera_pose_world, prompt, approach_camera,
            max_depth=0.8,
        )
    (args.output_dir / f"{scene_id}_{object_id}_wrist_refine_response.json").write_text(
        json.dumps(_json_compatible(response), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if world_tcp_poses is None:
        diag["wrist_refine_skip_reason"] = f"graspgen_{response.get('status', 'no_grasps')}"
        return None, None, diag

    object_now = np.asarray(
        sim.pull_grasp_metrics().get("object_position", fallback_grasp_point), dtype=float
    )
    candidate, best_off_target = _select_reachable_candidate(
        grasp_ik, world_tcp_poses, confidences, base_pose, object_now,
    )
    if candidate is None:
        diag["wrist_refine_skip_reason"] = (
            "ik_unreachable" if best_off_target is None else "candidate_off_target"
        )
        if best_off_target is not None:
            diag["wrist_refine_best_off_target_m"] = best_off_target
        return None, None, diag

    source_index = candidate.index // 2
    refined_point = world_tcp_poses[source_index][:3, 3]
    diag.update(
        wrist_refine_used=True,
        wrist_refine_position_error_m=candidate.position_error,
        wrist_refine_orientation_error_rad=candidate.orientation_error,
        wrist_refine_shift_m=float(np.linalg.norm(refined_point - fallback_grasp_point)),
    )
    return candidate.joints, refined_point, diag


def run_object(
    xml_path: Path,
    manifest: dict,
    scene_id: str,
    report_path: Path,
    args: argparse.Namespace,
) -> dict:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    goal_site = report["goal_site"]
    object_id = goal_site.removesuffix("_grasp_site")
    result: dict[str, Any] = {
        "object_id": object_id,
        "grasp_site": goal_site,
        "nav_report": str(report_path),
        "final_pose": report.get("final_pose"),
    }

    if not report.get("reached", False):
        result.update(success=False, skip_reason="navigation_not_reached")
        return result

    asset = _find_asset(manifest, object_id)

    prepared = _prepare_model(xml_path, object_id, goal_site)
    result.update(
        gravity_z=prepared["gravity_z"],
        friction_geoms_patched=prepared["friction_geoms_patched"],
        subtree_mass_kg=prepared["subtree_mass_kg"],
        manifest_mass_kg=asset.get("mass_kg"),
    )
    if not prepared["has_freejoint"]:
        result.update(success=False, skip_reason="not_dynamic")
        return result
    if prepared["gravity_z"] >= 0:
        result.update(success=False, skip_reason="gravity_disabled")
        return result

    x, y, yaw0 = report["final_pose"]
    sim = StretchMujocoSimulator(
        model=prepared["model"],
        cameras_to_use=[
            StretchCameras.cam_d435i_rgb,
            StretchCameras.cam_d435i_depth,
            StretchCameras.cam_d405_rgb,
            StretchCameras.cam_d405_depth,
        ],
        camera_hz=CAMERA_HZ,
        start_translation=[float(x), float(y), 0.0],
        start_rotation_quat=[float(np.cos(yaw0 / 2.0)), 0.0, 0.0, float(np.sin(yaw0 / 2.0))],
    )
    debug = _DebugRecorder(
        (args.output_dir / "debug") / object_id,
        args.debug,
    )
    try:
        sim.start(headless=True, use_passive_viewer=False)
        try:
            sim.get_base_pose()
        except RuntimeError as error:
            raise RuntimeError(
                "MuJoCo physics process stopped during startup; inspect the chained "
                f"exception for the MJCF/runtime cause: {error}"
            ) from error
        sim.set_robot_motion_speed(MOTION_SPEED)
        debug.start_continuous(sim)

        # Navigation keeps the lift high and the arm retracted, facing the
        # object -- this is the only stance with a clear line of sight.
        # Pivoting first (as the previous version did) to present the arm's
        # local -Y axis to the object, and *then* aiming the head and taking
        # the GraspGen photo, swings the wrist bracket (which has real length
        # even fully retracted) directly between the head camera and the
        # object at that stance, so GraspGen ends up "grasping" the robot's
        # own arm instead of the object (confirmed in a captured run: the
        # 03_observation/04_grasp_candidate crosshair landed on the gripper,
        # and the solved arm reach was ~0.17m -- basically zero). So: take
        # the photo and solve for a grasp candidate here, in world frame,
        # and only pivot into the grasp stance afterward -- the world-frame
        # poses below don't depend on which way the base is currently facing.
        sim.move_to(Actuators.arm, 0.0)
        sim.move_to(Actuators.lift, NAV_LIFT_CLEARANCE)
        _wait(sim, Actuators.arm, timeout=20.0)
        _wait(sim, Actuators.lift, timeout=20.0)
        sim.request_grasp_metrics(object_id)
        time.sleep(0.3)
        initial_metrics = sim.pull_grasp_metrics()
        initial_object_position = np.asarray(initial_metrics["object_position"], dtype=float)
        debug.capture(sim, initial_object_position, "01_nav_pose", detail="arm retracted, lift high")

        obj_x, obj_y = report["requested_goal"]
        base_x0, base_y0, _ = sim.get_base_pose()
        bearing = float(np.arctan2(obj_y - base_y0, obj_x - base_x0))
        grasp_yaw = float(np.arctan2(np.sin(bearing + np.pi / 2), np.cos(bearing + np.pi / 2)))
        result.update(bearing_to_object=bearing, grasp_stance_yaw=grasp_yaw)
        grasp_ik = StretchGraspIK(sim.urdf_model)

        _aim_head_at(sim, initial_object_position)
        debug.capture(sim, initial_object_position, "02_camera_aim")

        sim.move_to(Actuators.gripper, GRIPPER_OPEN)
        open_started = time.monotonic()
        gripper_opening = float(sim.pull_status().gripper.pos)
        while gripper_opening < 0.50 and time.monotonic() - open_started < 30.0:
            time.sleep(0.1)
            gripper_opening = float(sim.pull_status().gripper.pos)
        result["pregrasp_gripper_opening"] = gripper_opening

        cameras = sim.pull_camera_data()
        rgb = cameras.get_camera_data(
            StretchCameras.cam_d435i_rgb, auto_rotate=True, auto_correct_rgb=False
        )
        depth = cameras.get_camera_data(StretchCameras.cam_d435i_depth, auto_rotate=True)
        camera_k = np.asarray(cameras.cam_d435i_K, dtype=np.float32)
        if rgb.shape[:2] != depth.shape or camera_k.shape != (3, 3):
            raise RuntimeError("D435i RGB, depth, and intrinsics are inconsistent")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        observation_path = args.output_dir / f"{scene_id}_{object_id}_observation.npz"
        np.savez_compressed(observation_path, rgb=rgb, depth=depth, camera_K=camera_k)
        result["observation_npz"] = str(observation_path)

        # Use the existing RGB-D GraspGen service instead of the authored site
        # as the grasp target.  The photo is taken from the current
        # (nav-facing) heading, so bias the approach direction toward the
        # *future* grasp stance instead: the in-place pivot below only
        # changes yaw, not base position, so a hypothetical base pose at the
        # solved ``grasp_yaw`` gives the exact future orientation.
        raw_camera_pose = sim.get_link_pose("camera_color_optical_frame")
        world_camera_pose = rotated_d435i_optical_pose(raw_camera_pose)
        future_base_from_world = world_to_base_pose((base_x0, base_y0, grasp_yaw))
        approach_base = np.array([0.0, -1.0, 0.0], dtype=float)
        approach_camera = (future_base_from_world[:3, :3] @ world_camera_pose[:3, :3]).T @ approach_base
        prompt = _default_prompt(asset, object_id)
        projection = debug.capture(sim, initial_object_position, "03_observation",
                                   camera_k, detail=f"prompt={prompt}")
        result.update(graspgen_prompt=prompt, camera_target_projection=projection)
        with RGBDGraspClient(args.graspgen_host, args.graspgen_port) as client:
            health = client.health()
            if not health.get("ready"):
                raise RuntimeError(f"GraspGen service is not ready: {health}")
            world_tcp_poses, confidences, response = _infer_world_grasp_poses(
                client, rgb, depth, camera_k, world_camera_pose, prompt, approach_camera,
            )
        (args.output_dir / f"{scene_id}_{object_id}_graspgen_response.json").write_text(
            json.dumps(_json_compatible(response), indent=2, ensure_ascii=False), encoding="utf-8"
        )
        if world_tcp_poses is None:
            result.update(success=False, skip_reason=f"graspgen_{response.get('status', 'no_grasps')}")
            return result
        best_pose = world_tcp_poses[int(np.argmax(confidences))]

        # Both checks below must run *before* the pivot: once the base has
        # rotated into the grasp stance it can only translate perpendicular
        # to the object (for lateral centering), never closer/farther -- a
        # differential-drive base can't strafe. Right now, still facing the
        # object as navigation left it, driving forward/backward directly
        # changes the distance to it.
        #
        # 1) A pivot that starts too close risks sweeping the retracted
        #    wrist bracket -- which has real length even fully retracted --
        #    straight through the object: confirmed via the wrist camera, an
        #    object ended up wedged under the desk before the arm ever
        #    moved, purely from this in-place rotation.
        base_x0, base_y0, _ = sim.get_base_pose()
        distance_to_target = float(np.hypot(best_pose[0, 3] - base_x0, best_pose[1, 3] - base_y0))
        if distance_to_target < SAFE_PIVOT_DISTANCE_M:
            _move_base_forward(sim, distance_to_target - SAFE_PIVOT_DISTANCE_M)

        # 2) Navigation only guarantees its own path tolerance, not that the
        #    object ends up within the arm's ~0.5m reach once the pivot below
        #    turns the arm's fixed extension axis onto it. Check reach
        #    against the depth-sensed GraspGen candidate itself (the same
        #    RGB-D photo already sent above -- no privileged sim state), in
        #    the *future* post-pivot frame (a hypothetical base pose at the
        #    solved ``grasp_yaw``, matching the ``approach_camera`` bias
        #    above) since the pivot hasn't happened yet, and close any
        #    shortfall by driving straight at the object. Otherwise every IK
        #    candidate below saturates the arm short of the target and gets
        #    rejected -- the arm would only ever move up/down, never reach.
        base_x0, base_y0, _ = sim.get_base_pose()
        _, reach_shortfall_m, _ = grasp_ik.solve(
            world_to_base_pose((base_x0, base_y0, grasp_yaw)) @ best_pose
        )
        result["reach_shortfall_m"] = reach_shortfall_m
        if reach_shortfall_m > 0.02:
            _move_base_forward(sim, reach_shortfall_m + 0.05)

        # Only now pivot into the grasp stance (arm's local -Y axis facing
        # the object); everything above stays valid across this rotation.
        rotated = _rotate_to_yaw(sim, grasp_yaw)
        result.update(rotated_to_stance=rotated)

        # The photographed grasp point is a one-shot estimate from before
        # any of the retreat/approach/pivot driving above -- all of which
        # takes real physics time the object can settle/slide during. Track
        # that with the same privileged object-position signal already used
        # for the sanity check below (compared against its value at the
        # photo, not the object's current position, since a grasp point is
        # itself offset from the object's center) and re-target every
        # candidate by the same shift, rather than executing a pregrasp
        # aimed at where the object no longer is.
        object_now = np.asarray(
            sim.pull_grasp_metrics().get("object_position", initial_object_position), dtype=float
        )
        object_drift = object_now - initial_object_position
        result["object_drift_m"] = float(np.linalg.norm(object_drift))
        if np.linalg.norm(object_drift) > MAX_GRASP_POINT_OBJECT_DISTANCE_M:
            result.update(success=False, skip_reason="object_moved_too_much")
            return result
        world_tcp_poses = world_tcp_poses.copy()
        world_tcp_poses[:, :3, 3] += object_drift

        # GraspGen has no way to mask out the robot's own body from the
        # RGB-D image, so a stray/self-occluded detection can otherwise
        # "solve" cleanly (low IK error, decent confidence) while actually
        # pointing at the robot's own arm rather than the object -- the
        # drift correction above shifts every candidate uniformly, so the
        # sanity check inside ``_select_reachable_candidate`` still catches a
        # candidate that was never near the object at all.
        candidate, best_off_target = _select_reachable_candidate(
            grasp_ik, world_tcp_poses, confidences, sim.get_base_pose(), object_now,
        )
        if candidate is None:
            if best_off_target is None:
                result.update(success=False, skip_reason="ik_unreachable")
            else:
                result.update(
                    success=False,
                    skip_reason="candidate_off_target",
                    best_candidate_object_distance_m=best_off_target,
                )
            return result

        joints = candidate.joints
        source_index = candidate.index // 2
        grasp_pose_world = world_tcp_poses[source_index]
        grasp_point = grasp_pose_world[:3, 3]
        result.update(
            grasp_point=grasp_point,
            ik_position_error_m=candidate.position_error,
            ik_orientation_error_rad=candidate.orientation_error,
            graspgen_confidence=candidate.confidence,
            graspgen_candidate=source_index,
            ik_joints=joints,
        )
        # The head camera was aimed pre-pivot and never touched since --
        # fine for execution (nothing below reads it), but it now looks
        # nowhere near the grasp point, which makes every debug frame from
        # here on (pregrasp/align/close/lift) useless for actually watching
        # the attempt. Base and head don't move again after this, so one
        # re-aim covers the rest of the sequence.
        _aim_head_at(sim, grasp_point)
        debug.capture(sim, grasp_point, "04_grasp_candidate", camera_k,
                      detail=f"confidence={candidate.confidence:.3f}")

        if args.dry_run:
            result.update(success=None, skip_reason="dry_run")
            return result

        # Pregrasp: realize the IK solve's required base_x displacement, then
        # safe lift clearance, then wrist orientation, then partial then full
        # arm/lift extension. Adapted from ``_prepare_gripper`` +
        # ``_move_to_horizontal_grasp`` in collect_random_object_grasp_episodes.py
        # (base_x drive via ``_move_base_forward``, the heading-agnostic
        # counterpart of that script's ``_move_base_slow``), but sequenced to
        # extend the arm at a SAFE, ELEVATED lift height first and only
        # descend onto the grasp point afterward -- extending arm and lift
        # together at object height sweeps the open gripper straight through
        # the object (and its close office-counter neighbors), disturbing it
        # before the gripper ever gets to close. A fixed +0.15m margin over
        # the grasp height used to be enough for the tabletop task this was
        # adapted from, but is not for a taller office object (confirmed via
        # the wrist camera: a pouch-shaped object was swept off the desk
        # during this exact arm extension) -- reuse NAV_LIFT_CLEARANCE, the
        # same height navigation already relies on for a clear line of sight
        # over this desk.
        _move_base_forward(sim, float(joints[0]))
        safe_lift = NAV_LIFT_CLEARANCE
        sim.move_to(Actuators.arm, 0.0)
        _wait(sim, Actuators.arm, timeout=30.0)
        sim.move_to(Actuators.lift, safe_lift)
        _wait(sim, Actuators.lift, timeout=30.0)

        # Solve a *Cartesian* pregrasp standoff pose -- same orientation as
        # the final grasp, translated WRIST_REFINE_STANDOFF_M back along its
        # own approach axis -- rather than reaching it by simply shrinking
        # the "arm" joint while staying up at the elevated `safe_lift`. The
        # object's true height is normally well below `safe_lift`, so a
        # wrist camera held there looks down at it from a steep, wrong angle
        # and mostly misses it (a first version of this did exactly that,
        # and the "refinement" made pregrasp drift worse, not better).
        # Solving the standoff in Cartesian space keeps the camera pointed
        # at the same spot as the final approach, just further back along it.

        if PREGRASP_LATERAL_SHIFT_M != 0:
            # Fixed absolute shift, base-local +X (the only lateral DOF here)
            # rotated into world frame and applied via the same IK-solved
            # ``_shift_ee_world`` used for the forward/downward pushes below,
            # rather than the ``error_x / WRIST_YAW_LEVER_ARM_M`` lever-arm
            # approximation ``_align_grasp_center``/``_align_lateral_height``
            # use elsewhere -- all three fixed bias corrections now go
            # through the same mechanism.
            base_yaw = float(sim.get_base_pose()[2])
            delta_world = PREGRASP_LATERAL_SHIFT_M * np.array(
                [np.cos(base_yaw), np.sin(base_yaw), 0.0]
            )
            _shift_ee_world(sim, grasp_ik, delta_world)

        standoff_joints = _solve_standoff_pose(
            grasp_ik, grasp_pose_world, sim.get_base_pose(), joints, WRIST_REFINE_STANDOFF_M,
        )
        result["wrist_refine_standoff_solved"] = standoff_joints is not None
        approach_joints = standoff_joints if standoff_joints is not None else joints

        sim.move_to(Actuators.wrist_yaw, float(approach_joints[3]))
        sim.move_to(Actuators.wrist_pitch, FLAT_WRIST_PITCH)
        sim.move_to(Actuators.wrist_roll, FLAT_WRIST_ROLL)
        _wait(sim, Actuators.wrist_yaw, timeout=20.0)
        _wait(sim, Actuators.wrist_pitch, timeout=20.0)
        _wait(sim, Actuators.wrist_roll, timeout=20.0)

        # Extend to the standoff's reach while still elevated, then descend
        # straight down to its (near-final) height -- the same
        # elevate-extend-then-descend order used for the final approach
        # below, just targeting the backed-off pose first.
        sim.move_to(Actuators.arm, float(approach_joints[2]))
        _wait(sim, Actuators.arm, timeout=25.0, tolerance=0.01)
        sim.move_to(Actuators.lift, float(approach_joints[1]))
        _wait(sim, Actuators.lift, timeout=25.0, tolerance=0.01)

        refined_joints, refined_grasp_point, refine_diag = None, None, {"wrist_refine_used": False}
        if standoff_joints is not None:
            refined_joints, refined_grasp_point, refine_diag = _wrist_refine_grasp(
                sim, grasp_ik, args, scene_id, object_id, prompt, grasp_point,
            )
        else:
            refine_diag["wrist_refine_skip_reason"] = "standoff_unsolvable"
        result.update(refine_diag)
        if refined_joints is not None:
            joints, grasp_point = refined_joints, refined_grasp_point
        debug.capture(sim, grasp_point, "04b_wrist_refine", camera_k,
                      detail=f"used={refined_joints is not None}")

        # Commit to the final wrist/arm/lift target -- a small correction
        # from the standoff stance either way, comparable in size to the
        # ``_align_grasp_center`` residual correction already applied below.
        sim.move_to(Actuators.wrist_yaw, float(joints[3]))
        sim.move_to(Actuators.wrist_pitch, FLAT_WRIST_PITCH)
        sim.move_to(Actuators.wrist_roll, FLAT_WRIST_ROLL)
        _wait(sim, Actuators.wrist_yaw, timeout=20.0)
        _wait(sim, Actuators.wrist_pitch, timeout=20.0)
        _wait(sim, Actuators.wrist_roll, timeout=20.0)

        # Verify/correct left-right and height *before* driving the arm the
        # rest of the way in -- the arm is still short at the standoff reach
        # here, so it can't yet touch the object; if the wrist-refine solve's
        # wrist_yaw was off, this is the last chance to fix it before the
        # forward extension below.
        _align_lateral_height(sim, grasp_point)



        sim.move_to(Actuators.arm, float(joints[2]))
        arm_reached = _wait(sim, Actuators.arm, timeout=25.0, tolerance=0.01)
        sim.move_to(Actuators.lift, float(joints[1]))
        lift_reached = _wait(sim, Actuators.lift, timeout=25.0, tolerance=0.01)
        time.sleep(0.3)


        status = sim.pull_status()
        result.update(
            pregrasp_arm_reached=arm_reached,
            pregrasp_lift_reached=lift_reached,
            pregrasp_arm_error_m=float(joints[2]) - float(status.arm.pos),
            pregrasp_lift_error_m=float(joints[1]) - float(status.lift.pos),
            pregrasp_wrist_yaw_error=float(joints[3]) - float(status.wrist_yaw.pos),
            pregrasp_wrist_pitch_error=FLAT_WRIST_PITCH - float(status.wrist_pitch.pos),
            pregrasp_wrist_roll_error=FLAT_WRIST_ROLL - float(status.wrist_roll.pos),
            pregrasp_center_distance_m=sim.pull_grasp_metrics().get("center_distance_m"),
        )
        # ``center_distance_m`` is body-origin to body-origin, so it always
        # carries a nonzero baseline offset (the grasp_site's own height above
        # the object's origin); the gripper-to-intended-grasp-point distance
        # is the more meaningful pregrasp diagnostic.
        result["pregrasp_target_distance_m"] = float(
            np.linalg.norm(sim.get_ee_pose()[:3, 3] - grasp_point)
        )
        debug.capture(sim, grasp_point, "05_pregrasp", camera_k,
                      detail=f"dist={result['pregrasp_target_distance_m']:.3f}m")

        _align_grasp_center(sim, grasp_point)
        time.sleep(0.2)
        debug.capture(sim, grasp_point, "06_aligned", camera_k)

        if POST_ALIGN_PUSH_M > 0:
            # Deliberately overshoot the already-aligned position by a fixed
            # ``POST_ALIGN_PUSH_M`` along the gripper's *current* (post-flat-
            # wrist) approach axis, to compensate for GraspGen/IK's depth
            # estimate typically falling short of the object's real surface.
            real_push_m = _push_in_along_approach(sim, grasp_ik, POST_ALIGN_PUSH_M)
            if PUSH_IN_DOWNWARD_SHIFT_M != 0:
                # Same fixed, IK-solved world-space displacement as the
                # forward push above (see ``_shift_ee_world``) -- not a bare
                # ``lift.pos -=`` joint bump, which only equals a true -Z
                # Cartesian drop when the mast is exactly world-vertical and
                # isn't checked for IK reachability the way this is.
                _shift_ee_world(sim, grasp_ik, np.array([0.0, 0.0, -PUSH_IN_DOWNWARD_SHIFT_M]))

            debug.capture(sim, grasp_point, "06b_push_in", camera_k,
                          detail=f"push={real_push_m:.3f}m")

        sim.move_to(Actuators.gripper, GRIPPER_CLOSE)
        close_started = time.monotonic()
        gripper_position = float(sim.pull_status().gripper.pos)
        grasp_metrics = sim.pull_grasp_metrics()
        while (
            not grasp_metrics.get("bilateral_contact", False)
            and time.monotonic() - close_started < 30.0
        ):
            time.sleep(0.1)
            gripper_position = float(sim.pull_status().gripper.pos)
            grasp_metrics = sim.pull_grasp_metrics()
            if gripper_position <= 0.0 and time.monotonic() - close_started >= 2.0:
                break
        result["grasp_metrics_at_close"] = grasp_metrics
        debug.capture(sim, grasp_point, "07_close", camera_k,
                      detail=f"L={left_contacts if 'left_contacts' in locals() else 0} R={right_contacts if 'right_contacts' in locals() else 0}")

        bilateral_contact = bool(grasp_metrics.get("bilateral_contact", False))
        left_contacts = int(grasp_metrics.get("left_finger_contacts", 0))
        right_contacts = int(grasp_metrics.get("right_finger_contacts", 0))
        if not (bilateral_contact and left_contacts > 0 and right_contacts > 0):
            result.update(success=False, skip_reason="no_bilateral_contact")
            return result

        # No sim.attach_object_to_gripper() call here on purpose: that call
        # kinematically welds the object to the gripper for the rest of the
        # run -- every physics step it overwrites the object's qpos to a
        # fixed gripper-relative pose, zeroes its velocity, and disables its
        # gravity/collisions (see ``_apply_grasp_attachment`` in
        # mujoco_server.py) -- which would make the lift below succeed
        # unconditionally regardless of grip quality. This test is meant to
        # exercise the real thing instead: whether the boosted fingertip/
        # object friction (``GRASP_FRICTION``, applied in ``_prepare_model``)
        # plus the closed gripper's clamping force alone -- with gravity
        # still fully acting on the object -- can hold it through the
        # in-place lift. Check contact after the lift below so a slip is
        # attributed to the actual post-close motion instead of only
        # surfacing as a failed final lift height.
        time.sleep(0.3)

        # After closing, lift in place. Do not retract the arm or tilt/roll
        # the wrist: those motions change the grasp point in world X/Y and
        # can pull the object out of the fingers before the lift is applied.
        current_lift = float(sim.pull_status().lift.pos)
        sim.move_to(Actuators.lift, float(np.clip(current_lift + 0.20, 0.0, 1.1)))
        _wait(sim, Actuators.lift, timeout=30.0)
        time.sleep(0.5)  # hold the lifted pose while physics settles

        final_metrics = sim.pull_grasp_metrics()
        lift_delta = float(final_metrics["object_position"][2] - initial_object_position[2])
        final_bilateral = bool(final_metrics.get("bilateral_contact", False))
        result.update(
            initial_object_position=initial_object_position,
            final_grasp_metrics=final_metrics,
            lift_delta_m=lift_delta,
            bilateral_contact=final_bilateral,
            success=final_bilateral and lift_delta >= MIN_LIFT_M,
        )
        debug.capture(sim, np.asarray(final_metrics["object_position"], dtype=float), "08_lift", camera_k,
                      detail=f"lift={lift_delta:.3f}m")
        if not result["success"]:
            result["skip_reason"] = "insufficient_lift_or_lost_contact"
        return result
    finally:
        debug.close(f"{scene_id}_{object_id}_process")
        if sim.is_running():
            sim.stop()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", default="1", help="1..10 or scene id")
    parser.add_argument("--nav-report-dir", type=Path, default=Path("output/office_nav_sim"))
    parser.add_argument("--object-id", default=None, help="comma-separated object ids to restrict to")
    parser.add_argument("--output-dir", type=Path, default=Path("output/office_grasp_sim"))
    parser.add_argument("--graspgen-host", default=os.getenv("GRASPGEN_HOST", "10.29.150.95"))
    parser.add_argument("--graspgen-port", type=int, default=5557)
    parser.add_argument("--debug", action="store_true",
                        help="Save per-stage RGB-D debug PNGs and a process GIF")
    parser.add_argument("--list", action="store_true", help="List discovered reports and exit")
    parser.add_argument("--dry-run", action="store_true", help="IK planning only, no motion")
    args = parser.parse_args()

    xml_path, manifest_path = _scene_paths(args.scene)[0]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    scene_id = manifest["scene_id"]

    report_paths = sorted(args.nav_report_dir.glob(f"{scene_id}_*_grasp_site.json"))
    if args.object_id:
        wanted = {value.strip() for value in args.object_id.split(",") if value.strip()}
        report_paths = [
            p for p in report_paths if p.name.removeprefix(f"{scene_id}_").removesuffix(
                "_grasp_site.json"
            ) in wanted
        ]

    if args.list:
        for path in report_paths:
            report = json.loads(path.read_text(encoding="utf-8"))
            print(f"{report['goal_site']}: reached={report.get('reached')} path={path}")
        return 0

    if not report_paths:
        print(f"No navigation reports found for {scene_id} in {args.nav_report_dir}")
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for report_path in report_paths:
        result = run_object(xml_path, manifest, scene_id, report_path, args)
        results.append(result)
        object_json = args.output_dir / f"{scene_id}_{result['object_id']}_grasp.json"
        object_json.write_text(
            json.dumps(_json_compatible(result), indent=2, ensure_ascii=False), encoding="utf-8"
        )
        status = "SKIP" if result.get("skip_reason") and result.get("success") is None else (
            "PASS" if result.get("success") else "FAIL"
        )
        print(
            f"{status} {result['object_id']}: success={result.get('success')} "
            f"reason={result.get('skip_reason')} lift_delta_m={result.get('lift_delta_m')} "
            f"report={object_json}"
        )

    summary_path = args.output_dir / f"{scene_id}_grasp_summary.json"
    summary = {
        "scene_id": scene_id,
        "results": [
            {
                "object_id": r["object_id"],
                "success": r.get("success"),
                "skip_reason": r.get("skip_reason"),
                "lift_delta_m": r.get("lift_delta_m"),
            }
            for r in results
        ],
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Summary: {summary_path}")

    attempted = [r for r in results if r.get("success") is not None]
    failed = [r for r in attempted if not r.get("success")]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
