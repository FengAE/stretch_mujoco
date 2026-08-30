#!/usr/bin/env python3
"""NavDP point-kinematics demo.

Drives a 2-D point robot through a MuJoCo scene using the NavDP closed loop:
virtual aligned camera (ImageCapture) → NavDP server (HTTP) → trajectory →
lookahead tracker → ``(v, w)`` integrated directly onto the point.

Requires a running ``navdp_server.py`` (see ``--server-url``).

Usage
-----
  .venv/bin/python stretch_mujoco/navigations/NavDP/demo.py --headless \
      --server-url http://127.0.0.1:8888 --goal "3.0,2.0" --steps 120
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np

from stretch_mujoco.navigations import Algorithm, NavigationController
from stretch_mujoco.navigations.NavDP import NavDPClient
from stretch_mujoco.navigations.VLFM.image_capture import ImageCapture


def _default_scene() -> str:
    import stretch_mujoco

    return str(Path(stretch_mujoco.__file__).resolve().parent / "models" / "office_scene.xml")


def _find_free_start(nav: NavigationController, seed: int = 42) -> tuple[float, float]:
    """Pick a world position on the floor that is free in the occupancy grid."""
    rng = np.random.default_rng(seed)
    xs = np.linspace(nav.grid.x_min + 0.8, nav.grid.x_max - 0.8, 12)
    ys = np.linspace(nav.grid.y_min + 0.8, nav.grid.y_max - 0.8, 12)
    free = [
        (float(x), float(y))
        for x in xs
        for y in ys
        if nav.is_free((x, y))
    ]
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
@click.option("--render-width", default=424, show_default=True)
@click.option("--render-height", default=240, show_default=True)
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
@click.option("--dt", default=0.1, show_default=True, help="Control-tick integration step (s).")
@click.option("--steps", default=120, show_default=True)
@click.option("--headless", is_flag=True, help="Run without the matplotlib window.")
@click.option("--save/--no-save", default=True, help="Save result plot.")
def main(
    scene,
    server_url,
    intrinsic,
    start,
    goal,
    render_width,
    render_height,
    camera_height,
    pitch,
    fovy,
    max_v,
    max_w,
    goal_tol,
    stop_threshold,
    plan_interval,
    dt,
    steps,
    headless,
    save,
):
    """NavDP point-kinematics navigation demo."""
    if headless:
        import matplotlib

        matplotlib.use("Agg")
    scene_path = scene or _default_scene()
    fx, fy, cx, cy = (float(v) for v in intrinsic.split(","))
    intrinsic_k = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
    goal_xy = np.array([float(v) for v in goal.split(",")], dtype=float)[:2]

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
    # Speed limits for the point kinematics (mirror the tracker defaults).
    nav.navdp.tracker.max_v = float(max_v)
    nav.navdp.tracker.max_w = float(max_w)
    nav.navdp.tracker.goal_tol = float(goal_tol)

    robot = np.array([float(v) for v in start.split(",")] if start else _find_free_start(nav))
    yaw = 0.0

    capture = ImageCapture(nav.model, nav.data, width=int(render_width), height=int(render_height))

    print(f"\n{'='*60}")
    print("  NavDP Point Navigation")
    print(f"{'='*60}")
    print(f"Scene: {scene_path}")
    print(f"Server: {server_url}")
    print(f"Start: ({robot[0]:.2f}, {robot[1]:.2f})  yaw={yaw:.2f}")
    print(f"Goal:  ({goal_xy[0]:.2f}, {goal_xy[1]:.2f})")
    print(f"Grid: {nav.grid.occupancy.shape} | bounds "
          f"X[{nav.grid.x_min:.1f},{nav.grid.x_max:.1f}] "
          f"Y[{nav.grid.y_min:.1f},{nav.grid.y_max:.1f}]")
    print(f"Steps: {steps} @ {dt}s  | max_v={max_v} max_w={max_w} goal_tol={goal_tol}\n")

    path_history: list[np.ndarray] = [robot.copy()]
    reached = False
    t0 = time.perf_counter()

    try:
        for step in range(steps):
            rgb_bytes, depth_m = capture.capture_at_robot(
                tuple(robot), yaw, camera_height=camera_height, pitch=pitch, fovy=fovy
            )
            v, w = nav.step_navdp(robot, yaw, rgb_bytes, depth_m, goal_xy)

            yaw = math.atan2(math.sin(yaw + w * dt), math.cos(yaw + w * dt))
            robot = robot + v * np.array([math.cos(yaw), math.sin(yaw)]) * dt
            path_history.append(robot.copy())

            if nav.navdp.tracker.reached(robot, goal_xy):
                reached = True
                print(f"[{step:3d}] REACHED goal at ({robot[0]:.2f}, {robot[1]:.2f})")
                break

            if step % 10 == 0 or nav.navdp.diag.replan_count == 1:
                err = nav.navdp.diag.last_error or "-"
                print(
                    f"[{step:3d}] pos=({robot[0]:.2f}, {robot[1]:.2f}) "
                    f"yaw={yaw:+.2f} v={v:+.2f} w={w:+.2f} "
                    f"replan={nav.navdp.diag.replan_count} err={err}"
                )
    finally:
        capture.close()

    elapsed = time.perf_counter() - t0
    final_err = float(np.linalg.norm(robot - goal_xy))
    print(
        f"\nDone in {elapsed:.1f}s | replans={nav.navdp.diag.replan_count} | "
        f"final goal error={final_err:.2f}m | reached={reached}"
    )

    if save:
        _save_plot(nav, np.array(path_history), robot, goal_xy, scene_path, reached)
    if not headless:
        plt.show()


def _save_plot(
    nav: NavigationController,
    path: np.ndarray,
    final_pose: np.ndarray,
    goal_xy: np.ndarray,
    scene_path: str,
    reached: bool,
) -> None:
    """Save a top-down plot of the driven path over the occupancy grid."""
    occ = nav.grid.occupancy
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.imshow(
        occ.astype(float),
        extent=[nav.grid.x_min, nav.grid.x_max, nav.grid.y_min, nav.grid.y_max],
        origin="lower",
        cmap="Greys",
        aspect="equal",
        interpolation="none",
    )
    ax.plot(path[:, 0], path[:, 1], "-o", color="tab:blue", markersize=2, linewidth=2, label="driven path")
    ax.plot(goal_xy[0], goal_xy[1], "X", color="tab:red", markersize=14, markeredgecolor="black", label="goal")
    ax.plot(path[0, 0], path[0, 1], "o", color="tab:green", markersize=10, markeredgecolor="black", label="start")
    ax.set_xlim(nav.grid.x_min, nav.grid.x_max)
    ax.set_ylim(nav.grid.y_min, nav.grid.y_max)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(f"NavDP Point Navigation — reached={reached}")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.15)
    fig.tight_layout()
    out = Path(__file__).resolve().parent / "navdp_demo.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Figure saved → {out}")


if __name__ == "__main__":
    main()
