#!/usr/bin/env python3
"""NavDP demo that drives the real Stretch3 MuJoCo simulator.

Unlike ``demo.py`` (which integrates ``(v, w)`` onto a 2-D point), this script
commands the actual simulated Stretch base via ``set_base_velocity`` and reads
its pose from ``get_base_pose``.  Observations still come from the aligned
virtual ``ImageCapture`` camera (real-robot parity is achieved by swapping the
observation source for the D435i ROS2 topics — ``NavDPPlanner`` does not care).

Requires a running ``navdp_server.py`` (see ``--server-url``).

Usage
-----
  .venv/bin/python stretch_mujoco/navigations/NavDP/demo_simulator.py --headless \
      --server-url http://127.0.0.1:8888 --goal "3.0,2.0" --steps 240
"""

from __future__ import annotations

import time
from pathlib import Path

import click
import numpy as np

from stretch_mujoco.navigations import Algorithm, NavigationController
from stretch_mujoco.navigations.NavDP import NavDPClient
from stretch_mujoco.navigations.VLFM.image_capture import ImageCapture
from stretch_mujoco.robots.stretch3 import Stretch3RobotSimulator


def _default_scene() -> str:
    import stretch_mujoco

    return str(Path(stretch_mujoco.__file__).resolve().parent / "models" / "office_scene.xml")


def _find_free_start(nav: NavigationController, seed: int = 42) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    xs = np.linspace(nav.grid.x_min + 1.0, nav.grid.x_max - 1.0, 12)
    ys = np.linspace(nav.grid.y_min + 1.0, nav.grid.y_max - 1.0, 12)
    free = [(float(x), float(y)) for x in xs for y in ys if nav.is_free((x, y))]
    if not free:
        raise RuntimeError("No free start position found in the scene grid")
    rng.shuffle(free)
    return free[0]


@click.command()
@click.option("--scene", default=None, show_default=True, help="MuJoCo scene XML path.")
@click.option("--server-url", default="http://127.0.0.1:8888", show_default=True)
@click.option(
    "--intrinsic",
    default="304.24,304.07,212,120",
    show_default=True,
    help="Camera K (fx,fy,cx,cy) for the NavDP server (D435i-like default).",
)
@click.option("--start", default=None, help="World start 'x,y' (default: auto free point).")
@click.option("--goal", default="3.0,2.0", show_default=True, help="World goal 'x,y'.")
@click.option("--camera-height", default=1.0, show_default=True)
@click.option("--pitch", default=-0.15, show_default=True)
@click.option("--fovy", default=70.0, show_default=True)
@click.option("--max-v", default=0.4, show_default=True)
@click.option("--max-w", default=0.6, show_default=True)
@click.option("--goal-tol", default=0.3, show_default=True)
@click.option(
    "--stop-threshold",
    default=-1.5,
    show_default=True,
    help="NavDP stops when its best trajectory value falls below this.",
)
@click.option(
    "--plan-interval",
    default=8,
    show_default=True,
    help="Minimum control ticks between replans (starvation safety net).",
)
@click.option("--control-hz", default=20.0, show_default=True)
@click.option("--steps", default=240, show_default=True)
@click.option("--headless", is_flag=True)
def main(
    scene,
    server_url,
    intrinsic,
    start,
    goal,
    camera_height,
    pitch,
    fovy,
    max_v,
    max_w,
    goal_tol,
    stop_threshold,
    plan_interval,
    control_hz,
    steps,
    headless,
):
    """NavDP navigation driving the real Stretch3 simulator base."""
    scene_path = scene or _default_scene()
    fx, fy, cx, cy = (float(v) for v in intrinsic.split(","))
    intrinsic_k = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
    goal_xy = np.array([float(v) for v in goal.split(",")], dtype=float)[:2]

    # --- Planner + grid (separate static model/data, used for ImageCapture) ---
    client = NavDPClient(base_url=server_url)
    nav = NavigationController.from_scene_xml(
        scene_path,
        algorithm=Algorithm.NAVDP,
        planner_kwargs={
            "client": client,
            "intrinsic": intrinsic_k,
            "stop_threshold": float(stop_threshold),
            "plan_interval": int(plan_interval),
        },
    )
    nav.navdp.tracker.max_v = float(max_v)
    nav.navdp.tracker.max_w = float(max_w)
    nav.navdp.tracker.goal_tol = float(goal_tol)

    start_xy = np.array([float(v) for v in start.split(",")] if start else _find_free_start(nav))

    capture = ImageCapture(nav.model, nav.data, width=424, height=240)

    # --- Simulator ---
    sim = Stretch3RobotSimulator(
        scene_xml_path=scene_path,
        cameras_to_use=[],
        start_translation=[float(start_xy[0]), float(start_xy[1]), 0.0],
        start_rotation_quat=[1.0, 0.0, 0.0, 0.0],
    )
    sim.start(headless=headless, use_passive_viewer=True)

    print(f"\n{'='*60}")
    print("  NavDP Simulator Navigation")
    print(f"{'='*60}")
    print(f"Scene: {scene_path}")
    print(f"Start: ({start_xy[0]:.2f}, {start_xy[1]:.2f})")
    print(f"Goal:  ({goal_xy[0]:.2f}, {goal_xy[1]:.2f})")
    print(f"Control: {control_hz} Hz | steps={steps} | max_v={max_v} max_w={max_w}\n")

    reached = False
    dt = 1.0 / float(control_hz)
    t0 = time.perf_counter()
    try:
        for step in range(steps):
            if not sim.is_running():
                raise RuntimeError("Simulator stopped during navigation")
            x, y, yaw = (float(v) for v in sim.get_base_pose())
            rgb_bytes, depth_m = capture.capture_at_robot(
                (x, y), yaw, camera_height=camera_height, pitch=pitch, fovy=fovy
            )
            v, w = nav.step_navdp((x, y), yaw, rgb_bytes, depth_m, goal_xy)
            sim.set_base_velocity(v, w, 0.0)

            if nav.navdp.tracker.reached((x, y), goal_xy):
                reached = True
                print(f"[{step:3d}] REACHED goal at ({x:.2f}, {y:.2f})")
                break

            if step % 20 == 0 or nav.navdp.diag.replan_count == 1:
                err = nav.navdp.diag.last_error or "-"
                print(
                    f"[{step:3d}] pos=({x:.2f}, {y:.2f}) yaw={yaw:+.2f} "
                    f"v={v:+.2f} w={w:+.2f} replan={nav.navdp.diag.replan_count} err={err}"
                )
            time.sleep(dt)
    finally:
        try:
            sim.set_base_velocity(0.0, 0.0, 0.0)
        except Exception:
            pass
        sim.stop()
        capture.close()

    elapsed = time.perf_counter() - t0
    final_pose = sim.get_base_pose() if sim.is_running() else (None, None, None)
    if final_pose[0] is None:
        final_pose = (x, y, yaw)
    final_err = float(np.linalg.norm(np.array(final_pose[:2]) - goal_xy))
    print(
        f"\nDone in {elapsed:.1f}s | replans={nav.navdp.diag.replan_count} | "
        f"final goal error={final_err:.2f}m | reached={reached}"
    )


if __name__ == "__main__":
    main()
