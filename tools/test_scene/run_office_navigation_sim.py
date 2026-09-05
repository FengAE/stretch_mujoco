#!/usr/bin/env python3
"""Execute an office navigation path in the real MuJoCo Stretch simulator.

Planning uses a copy of the scene with the robot collision geoms disabled;
execution loads the original XML, so wheel/base/furniture contacts remain
physical.  A side-by-side GIF contains the fixed simulator overview camera
and the robot D435i camera.  The run also records lidar proximity warnings,
pose tracking, and path metrics in JSON.

Example::

  MUJOCO_GL=egl conda run -n habitat_310 python tools/run_office_navigation_sim.py \
      --scene 1 --goal zone_snack_center --output-dir output/office_nav_sim
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
import tempfile

os.environ.setdefault("MUJOCO_GL", "egl")
import imageio.v2 as imageio
import mujoco
import numpy as np

FINAL_HOLD_SECONDS = 3.0
FINAL_ALIGN_MAX_W = 0.8
FINAL_ALIGN_ACCEL = 0.04
FINAL_ALIGN_FRAME_HZ = 15.0

from stretch_mujoco.enums.stretch_cameras import StretchCameras
from stretch_mujoco.enums.stretch_sensors import StretchSensors
from stretch_mujoco.navigations import Algorithm, NavigationController, NavigationPathError
from stretch_mujoco.robots.stretch3 import Stretch3RobotSimulator

try:
    from test_office_navigation import (  # noqa: E402
        _disable_robot_collision, _free_near, _scene_paths,
    )
except ImportError:
    from tools.test_scene.test_office_navigation import (  # noqa: E402
        _disable_robot_collision, _free_near, _scene_paths,
    )


def _parse_goal(model: mujoco.MjModel, data: mujoco.MjData, goal: str) -> np.ndarray:
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, goal)
    if site_id < 0:
        raise ValueError(f"goal site not found: {goal}")
    return data.site(goal).xpos[:2].copy()


def _goal_names(model: mujoco.MjModel) -> list[str]:
    """Return useful navigation target site names in the loaded scene."""
    names = []
    for site_id in range(model.nsite):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, site_id) or ""
        if name.startswith("zone_") and name.endswith("_center"):
            names.append(name)
        elif name.endswith("_grasp_site"):
            names.append(name)
    return names


def _side_by_side(overview: np.ndarray, d435: np.ndarray, *, label: str) -> np.ndarray:
    # Camera snapshots are RGB when auto_correct_rgb=False. Resize the D435i
    # view to the overview height while retaining its aspect ratio.
    overview = np.asarray(overview)
    d435 = np.asarray(d435)
    if d435.ndim == 2:
        d435 = np.repeat(d435[..., None], 3, axis=2)
    if overview.ndim == 2:
        overview = np.repeat(overview[..., None], 3, axis=2)
    target_h = overview.shape[0]
    if d435.shape[0] != target_h:
        import cv2
        d435 = cv2.resize(d435, (round(d435.shape[1] * target_h / d435.shape[0]), target_h))
    canvas = np.concatenate([overview, d435], axis=1).astype(np.uint8, copy=False)
    # Add a small status banner without changing the camera data itself.
    import cv2
    cv2.rectangle(canvas, (0, 0), (min(canvas.shape[1], 620), 30), (20, 20, 20), -1)
    cv2.putText(canvas, label[:90], (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def _distance_to_path(point: np.ndarray, path: list[np.ndarray]) -> float:
    return min(float(np.linalg.norm(point - p)) for p in path)


def _front_lidar_minimum(values: np.ndarray) -> float | None:
    """Return the nearest range in the forward ±35° lidar sector.

    The minimum over the complete 360° scan is not a valid collision guard:
    the robot may start close to a desk behind or beside it while driving
    forward safely.  Stretch's replicated lidar rays are ordered around the
    base, with index zero pointing forward.
    """
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.size == 0:
        return None
    half = max(1, round(values.size * 35.0 / 360.0))
    sector = np.concatenate([values[:half], values[-half:]])
    finite = sector[np.isfinite(sector) & (sector > 0)]
    return float(finite.min()) if finite.size else None


def _heading_error(position: np.ndarray, yaw: float, target: np.ndarray) -> float:
    """Signed yaw error needed to face a target point."""
    delta = np.asarray(target, dtype=float) - np.asarray(position, dtype=float)
    desired = float(np.arctan2(delta[1], delta[0]))
    return float(np.arctan2(np.sin(desired - yaw), np.cos(desired - yaw)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", default="1", help="1..10 or scene id")
    parser.add_argument("--goal", default="zone_snack_center", help="MuJoCo site name")
    parser.add_argument("--list-goals", action="store_true",
                        help="List valid zone/grasp site goals for the selected scene and exit")
    parser.add_argument("--algorithm", choices=("astar", "fmm"), default="astar")
    parser.add_argument("--resolution", type=float, default=0.10)
    parser.add_argument("--agent-radius", type=float, default=0.42,
                        help="Inflation for the physical base envelope")
    parser.add_argument("--control-hz", type=float, default=30.0)
    parser.add_argument("--max-seconds", type=float, default=120.0)
    parser.add_argument("--goal-tolerance", type=float, default=0.15)
    parser.add_argument("--final-yaw-tolerance", type=float, default=0.15,
                        help="Heading tolerance at a grasp site, in radians")
    parser.add_argument("--waypoint-tolerance", type=float, default=0.16)
    parser.add_argument("--max-v", type=float, default=0.6)
    parser.add_argument("--max-w", type=float, default=1.25)
    parser.add_argument("--collision-lidar-threshold", type=float, default=0.08,
                        help="Stop if minimum base lidar range drops below this value")
    parser.add_argument("--frame-hz", type=float, default=4.0)
    parser.add_argument("--gif-speed", type=float, default=4.0,
                        help="GIF playback speed multiplier")
    parser.add_argument("--stuck-seconds", type=float, default=20.0,
                        help="Stop when progress toward a waypoint stalls")
    parser.add_argument("--warmup-seconds", type=float, default=2.0)
    parser.add_argument("--output-dir", type=Path, default=Path("output/office_nav_sim"))
    parser.add_argument("--headless", action="store_true", default=True)
    args = parser.parse_args()
    if args.control_hz <= 0 or args.frame_hz <= 0:
        parser.error("control/frame rates must be positive")

    xml_path, manifest_path = _scene_paths(args.scene)[0]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    scene_id = manifest["scene_id"]

    # Planning model: disable only robot collision geoms.
    plan_model = mujoco.MjModel.from_xml_path(str(xml_path))
    plan_data = mujoco.MjData(plan_model)
    mujoco.mj_forward(plan_model, plan_data)
    _disable_robot_collision(plan_model)
    mujoco.mj_forward(plan_model, plan_data)
    if args.list_goals:
        print(f"{scene_id} goals:")
        print("\n".join(f"  {name}" for name in _goal_names(plan_model)))
        return 0
    start = np.asarray(manifest["robot"]["initial_pose"][:2], dtype=float)
    requested_goal = _parse_goal(plan_model, plan_data, args.goal)
    nav = NavigationController(plan_model, plan_data, algorithm=Algorithm(args.algorithm),
                                resolution=args.resolution, agent_radius=args.agent_radius,
                                require_collision=True, planner_kwargs={"smoothing": False})
    if not nav.is_free(start):
        raise RuntimeError(f"robot start is not free in planning grid: {start.tolist()}")
    # Zone targets are selected by the static test helper using manifest bounds
    # and maximum clearance; only grasp sites use a local fallback here.
    if args.goal.startswith("zone_"):
        from test_office_navigation import _best_zone_point
        zone_type = args.goal.removeprefix("zone_").removesuffix("_center")
        zone = next((z for z in manifest.get("zones", []) if z.get("type") == zone_type), None)
        goal = _best_zone_point(nav, zone["bounds"], requested_goal) if zone else _free_near(nav, requested_goal)
    else:
        goal = _free_near(nav, requested_goal)
    align_to_object = args.goal.endswith("_grasp_site")
    success_tolerance = max(args.goal_tolerance, 0.25) if align_to_object else args.goal_tolerance
    if goal is None:
        raise RuntimeError(f"could not find free floor approach near {args.goal}")
    try:
        path = nav.plan(start, goal)
    except NavigationPathError as error:
        raise RuntimeError(f"navigation plan failed: {error}") from error

    args.output_dir.mkdir(parents=True, exist_ok=True)
    gif_path = args.output_dir / f"{scene_id}_{args.goal}.gif"
    json_path = args.output_dir / f"{scene_id}_{args.goal}.json"
    cameras = [StretchCameras.cam_d435i_rgb, StretchCameras.office_overview_rgb]
    # Generated offices call the fixed cameras ``overview``/``top`` while the
    # simulator API exposes the canonical ``office_overview`` name.  Add a
    # harmless alias in a temporary wrapper so the regular camera pipeline can
    # render it without modifying generated assets.
    wrapper_dir = Path(tempfile.mkdtemp(prefix="office_nav_scene_"))
    wrapper = wrapper_dir / xml_path.name
    wrapper.write_text(
        f'<mujoco><include file="{xml_path.resolve()}"/><worldbody>'
        '<camera name="office_overview" mode="targetbody" target="scene_center" '
        'pos="0 0 24" fovy="45"/></worldbody></mujoco>', encoding="utf-8"
    )
    # Ensure the wrapper itself compiles before starting the multi-process sim.
    mujoco.MjModel.from_xml_path(str(wrapper))
    yaw0 = float(manifest["robot"]["initial_pose"][2])
    sim = Stretch3RobotSimulator(scene_xml_path=str(wrapper), cameras_to_use=cameras,
                                  camera_hz=max(1.0, args.frame_hz,
                                                FINAL_ALIGN_FRAME_HZ if align_to_object else 0.0),
                                  start_translation=[float(start[0]), float(start[1]), 0.0],
                                  start_rotation_quat=[float(np.cos(yaw0 / 2.0)), 0.0, 0.0,
                                                       float(np.sin(yaw0 / 2.0))])
    frames: list[np.ndarray] = []
    trace: list[dict] = []
    collisions: list[dict] = []
    wp_index = 1
    reached = False
    stopped_for_collision = False
    close_lidar_ticks = 0
    sim.start(headless=True, use_passive_viewer=False)
    # Force propagation of an exception raised by the background physics
    # process.  Otherwise ``is_running()`` can become false and the loop would
    # silently report an empty run.
    try:
        sim.get_base_pose()
    except RuntimeError as error:
        sim.stop()
        raise RuntimeError(
            "MuJoCo physics process stopped during startup; inspect the chained "
            f"exception for the MJCF/runtime cause: {error}"
        ) from error
    sim.set_base_velocity(0.0, 0.0, 0.0)
    time.sleep(max(0.0, args.warmup_seconds))
    started = time.perf_counter()
    next_frame = started
    previous_v = previous_w = 0.0
    best_distance = float("inf")
    last_progress = started
    aligning_final = False
    try:
        while sim.is_running() and time.perf_counter() - started < args.max_seconds:
            now = time.perf_counter()
            x, y, yaw = sim.get_base_pose()
            position = np.array([x, y], dtype=float)
            final_waypoint = wp_index >= len(path) - 1
            target = np.asarray(path[-1] if final_waypoint else path[wp_index], dtype=float)
            error = target - position
            distance = float(np.linalg.norm(error))
            if final_waypoint and distance <= success_tolerance:
                if not align_to_object:
                    reached = True
                    sim.set_base_velocity(0.0, 0.0, 0.0)
                    break
                # Finish a grasp-site run facing the target object.
                if not aligning_final:
                    previous_v = previous_w = 0.0
                    aligning_final = True
                to_object = requested_goal - position
                if np.linalg.norm(to_object) < 1e-6:
                    to_object = requested_goal - goal
                yaw_error = _heading_error(position, yaw, position + to_object)
                if abs(yaw_error) <= args.final_yaw_tolerance and abs(previous_w) <= FINAL_ALIGN_ACCEL:
                    reached = True
                    sim.set_base_velocity(0.0, 0.0, 0.0)
                    trace.append({"time_s": now - started, "pose": [x, y, yaw],
                                  "target": target.tolist(), "waypoint": wp_index,
                                  "distance_to_target_m": distance,
                                  "distance_to_path_m": _distance_to_path(position, path),
                                  "v": 0.0, "omega": 0.0,
                                  "min_lidar_m": None,
                                  "final_yaw_error_rad": abs(yaw_error)})
                    break
                v = 0.0
                desired_w = float(np.clip(2.2 * yaw_error, -FINAL_ALIGN_MAX_W, FINAL_ALIGN_MAX_W))
                w = previous_w + float(np.clip(
                    desired_w - previous_w, -FINAL_ALIGN_ACCEL, FINAL_ALIGN_ACCEL
                ))
                previous_v, previous_w = float(v), float(w)
                sim.set_base_velocity(v, w, 0.0)
                if now >= next_frame:
                    try:
                        snapshot = sim.pull_camera_data()
                        overview = snapshot.get_camera_data(
                            StretchCameras.office_overview_rgb, auto_correct_rgb=False
                        )
                        d435 = snapshot.get_camera_data(
                            StretchCameras.cam_d435i_rgb,
                            auto_rotate=True,
                            auto_correct_rgb=False,
                        )
                        frames.append(_side_by_side(overview, d435, label=f"{scene_id}  final-align"))
                        next_frame = now + 1.0 / FINAL_ALIGN_FRAME_HZ
                    except (ValueError, RuntimeError):
                        pass
                trace.append({"time_s": now - started, "pose": [x, y, yaw],
                              "target": target.tolist(), "waypoint": wp_index,
                              "distance_to_target_m": distance,
                              "distance_to_path_m": _distance_to_path(position, path),
                              "v": v, "omega": w, "min_lidar_m": None,
                              "final_yaw_error_rad": abs(yaw_error)})
                time.sleep(1.0 / args.control_hz)
                continue
            aligning_final = False
            # Do not call an in-place rotation "stuck"; only evaluate progress
            # while the base is actually translating toward its waypoint.
            if distance < best_distance - 0.05:
                best_distance, last_progress = distance, now
            elif previous_v > 0.03 and now - last_progress > args.stuck_seconds:
                collisions.append({"time_s": now - started, "event": "stuck",
                                   "pose": [x, y, yaw], "waypoint": wp_index,
                                   "distance_to_target_m": distance})
                stopped_for_collision = True
                sim.set_base_velocity(0.0, 0.0, 0.0)
                break
            if distance <= args.waypoint_tolerance and wp_index < len(path) - 1:
                wp_index += 1
                best_distance = float("inf")
                last_progress = now
                continue
            desired = float(np.arctan2(error[1], error[0]))
            yaw_error = float(np.arctan2(np.sin(desired - yaw), np.cos(desired - yaw)))
            # Slow down while turning to avoid wheel slip and corner impacts.
            # Rotate in place for large heading errors.  Driving an arc while
            # initially facing away from the first waypoint makes the physical
            # base overshoot narrow grid corridors and hit chair legs.
            if abs(yaw_error) > 0.55:
                v = 0.0
            else:
                turn_scale = max(0.45, 1.0 - abs(yaw_error) / np.pi)
                v = float(np.clip(1.4 * distance, 0.0, args.max_v) * turn_scale)
            w = float(np.clip(2.2 * yaw_error, -args.max_w, args.max_w))
            v = previous_v + np.clip(v - previous_v, -0.08, 0.08)
            w = previous_w + np.clip(w - previous_w, -0.16, 0.16)
            previous_v, previous_w = float(v), float(w)
            sim.set_base_velocity(v, w, 0.0)

            min_lidar = None
            try:
                lidar = sim.pull_sensor_data().get_data(StretchSensors.base_lidar)
                min_lidar = _front_lidar_minimum(np.asarray(lidar))
            except (ValueError, AttributeError):
                pass
            if v > 0.03 and min_lidar is not None and min_lidar < args.collision_lidar_threshold:
                close_lidar_ticks += 1
            else:
                close_lidar_ticks = 0
            # Require persistence; a single stale/zero lidar sample during
            # camera startup must not abort an otherwise valid run.
            if close_lidar_ticks >= 3 and now - started > 1.0:
                event = {"time_s": now - started, "pose": [x, y, yaw], "min_lidar_m": min_lidar,
                         "waypoint": wp_index}
                collisions.append(event)
                stopped_for_collision = True
                sim.set_base_velocity(0.0, 0.0, 0.0)
                break

            trace.append({"time_s": now - started, "pose": [x, y, yaw],
                          "target": target.tolist(), "waypoint": wp_index,
                          "distance_to_target_m": distance, "distance_to_path_m": _distance_to_path(position, path),
                          "v": v, "omega": w, "min_lidar_m": min_lidar})
            if now >= next_frame:
                try:
                    snapshot = sim.pull_camera_data()
                    overview = snapshot.get_camera_data(StretchCameras.office_overview_rgb,
                                                         auto_correct_rgb=False)
                    d435 = snapshot.get_camera_data(StretchCameras.cam_d435i_rgb,
                                                    auto_rotate=True, auto_correct_rgb=False)
                    frames.append(_side_by_side(overview, d435,
                                                label=f"{scene_id}  {args.goal}  wp {wp_index}/{len(path)-1}  d={distance:.2f}m"))
                    next_frame = now + 1.0 / args.frame_hz
                except ValueError:
                    pass
            time.sleep(1.0 / args.control_hz)
    finally:
        if sim.is_running():
            sim.set_base_velocity(0.0, 0.0, 0.0)
            if reached:
                # Keep the robot at the grasp pose and record those frames;
                # sleeping alone would not extend the rendered GIF.
                hold_frames = max(1, round(
                    FINAL_HOLD_SECONDS * args.frame_hz * max(args.gif_speed, 0.1)
                ))
                for _ in range(hold_frames):
                    frame = frames[-1] if frames else None
                    try:
                        snapshot = sim.pull_camera_data()
                        overview = snapshot.get_camera_data(
                            StretchCameras.office_overview_rgb, auto_correct_rgb=False
                        )
                        d435 = snapshot.get_camera_data(
                            StretchCameras.cam_d435i_rgb,
                            auto_rotate=True,
                            auto_correct_rgb=False,
                        )
                        frame = _side_by_side(overview, d435, label=f"{scene_id}  final")
                    except (ValueError, RuntimeError):
                        pass
                    if frame is not None:
                        frames.append(frame)
                    time.sleep(FINAL_HOLD_SECONDS / hold_frames)
            sim.stop()

    if not trace and not reached and not stopped_for_collision:
        # A background MuJoCo process can terminate before the first control
        # tick (for example because an MJCF runtime error was raised).  Keep
        # this diagnostic in the JSON instead of silently producing an empty
        # video/report.
        collisions.append({"time_s": 0.0, "event": "simulator_stopped_before_first_tick"})

    if not stopped_for_collision and trace:
        reached = bool(np.linalg.norm(np.asarray(trace[-1]["pose"][:2]) - goal) <= success_tolerance)
    if frames:
        imageio.mimsave(gif_path, frames,
                        duration=1.0 / (args.frame_hz * max(args.gif_speed, 0.1)), loop=0)
    final_pose = trace[-1]["pose"] if trace else [float(start[0]), float(start[1]), float(manifest["robot"]["initial_pose"][2])]
    final_error = float(np.linalg.norm(np.asarray(final_pose[:2]) - goal))
    final_yaw_error = (
        abs(_heading_error(np.asarray(final_pose[:2]), final_pose[2], requested_goal))
        if args.goal.endswith("_grasp_site") and trace else None
    )
    report = {
        "scene_id": scene_id, "xml": str(xml_path), "goal_site": args.goal,
        "requested_goal": requested_goal.tolist(), "approach_goal": goal.tolist(),
        "start": start.tolist(), "algorithm": args.algorithm,
        "resolution_m": args.resolution, "agent_radius_m": args.agent_radius,
        "path": [p.tolist() for p in path], "path_length_m": float(sum(np.linalg.norm(path[i+1]-path[i]) for i in range(len(path)-1))),
        "reached": reached, "stopped_for_collision": stopped_for_collision,
        "final_pose": final_pose, "final_error_m": final_error,
        "final_yaw_error_rad": final_yaw_error,
        "collision_events": collisions, "trace_samples": len(trace),
        "gif": str(gif_path) if frames else None,
    }
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("scene_id", "goal_site", "reached", "stopped_for_collision", "final_error_m", "gif")}, ensure_ascii=False))
    return 0 if reached and not stopped_for_collision else 1


if __name__ == "__main__":
    raise SystemExit(main())
