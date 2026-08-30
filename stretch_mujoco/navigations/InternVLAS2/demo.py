#!/usr/bin/env python3
"""InternVLA-N1 dual-system demo (point kinematics).

Feeds the running ``http_internvla_server.py`` (port 5801) with virtual RGB-D
views from a MuJoCo scene and executes the returned trajectory / discrete
actions on a 2-D point robot.  The dual system understands a natural-language
instruction (S2) and executes navigation (S1).

Requires a running InternVLA-N1 server (see ``--server-url``).

Usage
-----
  .venv/bin/python stretch_mujoco/navigations/InternVLAS2/demo.py --headless \
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


def _wrap(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def _default_scene() -> str:
    import stretch_mujoco

    return str(Path(stretch_mujoco.__file__).resolve().parent / "models" / "office_scene.xml")


def _find_free_start(nav: NavigationController, seed: int = 42) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    xs = np.linspace(nav.grid.x_min + 0.8, nav.grid.x_max - 0.8, 12)
    ys = np.linspace(nav.grid.y_min + 0.8, nav.grid.y_max - 0.8, 12)
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
@click.option("--render-width", default=424, show_default=True)
@click.option("--render-height", default=240, show_default=True)
@click.option("--camera-height", default=1.0, show_default=True)
@click.option("--pitch", default=-0.15, show_default=True)
@click.option("--fovy", default=70.0, show_default=True)
@click.option("--max-v", default=0.4, show_default=True)
@click.option("--max-w", default=0.6, show_default=True)
@click.option("--goal-tol", default=0.3, show_default=True)
@click.option("--dt", default=0.1, show_default=True)
@click.option("--steps", default=150, show_default=True)
@click.option("--headless", is_flag=True)
@click.option("--save/--no-save", default=True, help="Save result plot.")
def main(
    instruction,
    scene,
    server_url,
    start,
    render_width,
    render_height,
    camera_height,
    pitch,
    fovy,
    max_v,
    max_w,
    goal_tol,
    dt,
    steps,
    headless,
    save,
):
    """InternVLA-N1 dual-system point navigation demo."""
    if headless:
        import matplotlib

        matplotlib.use("Agg")
    scene_path = scene or _default_scene()
    width, height = int(render_width), int(render_height)

    client = InternVLAClient(base_url=server_url)
    nav = NavigationController.from_scene_xml(
        scene_path,
        algorithm=Algorithm.INTERVLAS2,
        planner_kwargs={"client": client, "instruction": instruction},
    )
    nav.internvla.tracker.max_v = float(max_v)
    nav.internvla.tracker.max_w = float(max_w)
    nav.internvla.tracker.goal_tol = float(goal_tol)

    robot = np.array([float(v) for v in start.split(",")] if start else _find_free_start(nav))
    yaw = 0.0
    capture = ImageCapture(nav.model, nav.data, width=width, height=height)

    print(f"\n{'='*60}")
    print("  InternVLA-N1 Dual-System Navigation")
    print(f"{'='*60}")
    print(f"Instruction: {instruction or '(server default)'}")
    print(f"Server: {server_url}")
    print(f"Scene: {scene_path}")
    print(f"Start: ({robot[0]:.2f}, {robot[1]:.2f})  yaw={yaw:.2f}\n")

    path_history: list[np.ndarray] = [np.array([robot[0], robot[1], yaw])]
    reached = False
    stuck_steps = 0
    t0 = time.perf_counter()
    try:
        for step in range(steps):
            rgb_bytes, depth_m = capture.capture_at_robot(
                tuple(robot), yaw, camera_height=camera_height, pitch=pitch, fovy=fovy
            )
            v, w = nav.step_internvla(robot, yaw, rgb_bytes, depth_m, instruction=instruction or None)

            yaw = math.atan2(math.sin(yaw + w * dt), math.cos(yaw + w * dt))
            robot = robot + v * np.array([math.cos(yaw), math.sin(yaw)]) * dt
            path_history.append(np.array([robot[0], robot[1], yaw]))

            if nav.internvla.diag.stop_requested:
                reached = True
                print(f"[{step:3d}] S2 says STOP — reached at ({robot[0]:.2f}, {robot[1]:.2f})")
                break

            # Stuck only when neither position nor heading changes.
            if step > 0:
                last = path_history[-2]
                moved_pos = float(np.linalg.norm(robot - last[:2]))
                moved_yaw = abs(_wrap(yaw - last[2]))
                if moved_pos < 0.02 and moved_yaw < 0.02:
                    stuck_steps += 1
                else:
                    stuck_steps = 0
            if stuck_steps >= 30:
                print(f"[{step:3d}] STUCK — no progress at ({robot[0]:.2f}, {robot[1]:.2f})")
                break

            if step % 10 == 0:
                err = nav.internvla.diag.last_error or "-"
                print(
                    f"[{step:3d}] pos=({robot[0]:.2f}, {robot[1]:.2f}) yaw={yaw:+.2f} "
                    f"v={v:+.2f} w={w:+.2f} type={nav.internvla.diag.last_response_type} "
                    f"act={nav.internvla.diag.last_discrete_action} "
                    f"replan={nav.internvla.diag.replan_count} err={err}"
                )
    finally:
        capture.close()

    elapsed = time.perf_counter() - t0
    print(
        f"\nDone in {elapsed:.1f}s | replans={nav.internvla.diag.replan_count} | "
        f"reached={reached} | final pos=({robot[0]:.2f}, {robot[1]:.2f})"
    )

    if save:
        _save_plot(nav, np.array(path_history), instruction, reached)


def _save_plot(
    nav: NavigationController,
    path: np.ndarray,
    instruction: str,
    reached: bool,
) -> None:
    """Save a top-down plot of the driven path over the occupancy grid."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 8))
    ax.imshow(
        nav.grid.occupancy.astype(float),
        extent=[nav.grid.x_min, nav.grid.x_max, nav.grid.y_min, nav.grid.y_max],
        origin="lower",
        cmap="Greys",
        aspect="equal",
        interpolation="none",
    )
    ax.plot(path[:, 0], path[:, 1], "-o", color="tab:blue", markersize=2, linewidth=2, label="driven path")
    ax.plot(path[0, 0], path[0, 1], "o", color="tab:green", markersize=10, markeredgecolor="black", label="start")
    ax.set_xlim(nav.grid.x_min, nav.grid.x_max)
    ax.set_ylim(nav.grid.y_min, nav.grid.y_max)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    title = f"InternVLA-N1 — {instruction[:40]}{'...' if len(instruction) > 40 else ''}"
    ax.set_title(f"{title} (reached={reached})")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.15)
    fig.tight_layout()
    out = Path(__file__).resolve().parent / "internvla_demo.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Figure saved → {out}")


if __name__ == "__main__":
    main()
