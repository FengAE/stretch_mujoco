#!/usr/bin/env python3
"""InternVLA-N1 dual-system demo driving the real Stretch3 MuJoCo simulator.

Commands the actual simulated Stretch base via ``set_base_velocity`` and reads
its pose from ``get_base_pose``, while feeding virtual aligned RGB-D views
(``ImageCapture``) to the running ``http_internvla_server.py`` (port 5801).

Run **without** ``--headless`` to see the MuJoCo 3D scene viewer with the
Stretch robot driving; add ``--headless`` for a pure terminal run.

Requires a running InternVLA-N1 server (see ``--server-url``).

Usage
-----
  .venv/bin/python stretch_mujoco/navigations/InternVLAS2/demo_simulator.py \
      --server-url http://127.0.0.1:5801 \
      --instruction "Go to the computer monitor"
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import click
import numpy as np

from stretch_mujoco.navigations import Algorithm, NavigationController
from stretch_mujoco.navigations.InternVLAS2 import InternVLAClient
from stretch_mujoco.navigations.VLFM.image_capture import ImageCapture
from stretch_mujoco.robots.stretch3 import Stretch3RobotSimulator


def _wrap(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


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
@click.option("--instruction", default="", help="Natural-language navigation goal.")
@click.option("--scene", default=None, show_default=True, help="MuJoCo scene XML path.")
@click.option("--server-url", default="http://127.0.0.1:5801", show_default=True)
@click.option("--start", default=None, help="World start 'x,y' (default: auto free point).")
@click.option("--camera-height", default=1.0, show_default=True)
@click.option("--pitch", default=-0.15, show_default=True)
@click.option("--fovy", default=70.0, show_default=True)
@click.option("--max-v", default=0.4, show_default=True)
@click.option("--max-w", default=0.6, show_default=True)
@click.option("--control-hz", default=20.0, show_default=True)
@click.option("--steps", default=240, show_default=True)
@click.option("--headless", is_flag=True, help="Run without the MuJoCo scene viewer.")
def main(
    instruction,
    scene,
    server_url,
    start,
    camera_height,
    pitch,
    fovy,
    max_v,
    max_w,
    control_hz,
    steps,
    headless,
):
    """InternVLA-N1 dual-system navigation driving the Stretch3 simulator base."""
    scene_path = scene or _default_scene()

    # --- Planner + grid (separate static model/data, used for ImageCapture) ---
    client = InternVLAClient(base_url=server_url)
    nav = NavigationController.from_scene_xml(
        scene_path,
        algorithm=Algorithm.INTERVLAS2,
        planner_kwargs={"client": client, "instruction": instruction},
    )
    nav.internvla.tracker.max_v = float(max_v)
    nav.internvla.tracker.max_w = float(max_w)

    start_xy = np.array([float(v) for v in start.split(",")] if start else _find_free_start(nav))
    capture = ImageCapture(nav.model, nav.data, width=424, height=240)

    # --- Simulator (passive viewer shows the 3D scene when not headless) ---
    sim = Stretch3RobotSimulator(
        scene_xml_path=scene_path,
        cameras_to_use=[],
        start_translation=[float(start_xy[0]), float(start_xy[1]), 0.0],
        start_rotation_quat=[1.0, 0.0, 0.0, 0.0],
    )
    sim.start(headless=headless, use_passive_viewer=True)

    print(f"\n{'='*60}")
    print("  InternVLA-N1 Dual-System Navigation (Simulator)")
    print(f"{'='*60}")
    print(f"Instruction: {instruction or '(server default)'}")
    print(f"Server: {server_url}")
    print(f"Scene: {scene_path}")
    print(f"Start: ({start_xy[0]:.2f}, {start_xy[1]:.2f})")
    print(f"Control: {control_hz} Hz | steps={steps} | max_v={max_v} max_w={max_w}\n")

    reached = False
    stuck_steps = 0
    last_pose = (float(start_xy[0]), float(start_xy[1]), 0.0)
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
            v, w = nav.step_internvla((x, y), yaw, rgb_bytes, depth_m, instruction=instruction or None)
            sim.set_base_velocity(v, w, 0.0)

            if nav.internvla.diag.stop_requested:
                reached = True
                print(f"[{step:3d}] S2 says STOP — reached at ({x:.2f}, {y:.2f})")
                break

            # Stuck only when neither position nor heading changes.
            moved_pos = math.hypot(x - last_pose[0], y - last_pose[1])
            moved_yaw = abs(_wrap(yaw - last_pose[2]))
            last_pose = (x, y, yaw)
            if moved_pos < 0.02 and moved_yaw < 0.02:
                stuck_steps += 1
            else:
                stuck_steps = 0
            if stuck_steps >= 30:
                print(f"[{step:3d}] STUCK — no progress at ({x:.2f}, {y:.2f})")
                break

            if step % 20 == 0:
                err = nav.internvla.diag.last_error or "-"
                print(
                    f"[{step:3d}] pos=({x:.2f}, {y:.2f}) yaw={yaw:+.2f} "
                    f"v={v:+.2f} w={w:+.2f} type={nav.internvla.diag.last_response_type} "
                    f"act={nav.internvla.diag.last_discrete_action} "
                    f"replan={nav.internvla.diag.replan_count} err={err}"
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
    print(
        f"\nDone in {elapsed:.1f}s | replans={nav.internvla.diag.replan_count} | "
        f"reached={reached}"
    )


if __name__ == "__main__":
    main()
