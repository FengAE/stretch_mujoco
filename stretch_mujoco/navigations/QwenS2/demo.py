#!/usr/bin/env python3
"""QwenS2 demo — instruction-driven NavDP navigation with an online Qwen2.5-VL.

An online Qwen2.5-VL (DashScope by default) points at a goal pixel from the
current camera view + a natural-language instruction; NavDP's pixel-goal policy
executes the trajectory in a MuJoCo scene (point kinematics).

Requires:
  - a running ``navdp_server.py`` (see ``--server-url``)
  - an OpenAI-compatible Qwen endpoint (DashScope default) + API key
    (``--api-key`` or ``DASHSCOPE_API_KEY`` / ``QWEN_API_KEY``)

Usage
-----
  .venv/bin/python stretch_mujoco/navigations/QwenS2/demo.py --headless \
      --instruction "go to the green armchair" \
      --server-url http://127.0.0.1:8890
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import click
import numpy as np

from stretch_mujoco.navigations import Algorithm, NavigationController
from stretch_mujoco.navigations.NavDP import NavDPClient
from stretch_mujoco.navigations.QwenS2 import QwenPixelGoalSelector
from stretch_mujoco.navigations.VLFM.image_capture import ImageCapture


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
@click.option("--instruction", required=True, help="Natural-language navigation goal.")
@click.option("--scene", default=None, show_default=True, help="MuJoCo scene XML path.")
@click.option("--server-url", default="http://127.0.0.1:8888", show_default=True)
@click.option(
    "--intrinsic",
    default="304.24,304.07,212,120",
    show_default=True,
    help="Camera K (fx,fy,cx,cy) for the NavDP server (D435i-like default).",
)
@click.option("--start", default=None, help="World start 'x,y' (default: auto free point).")
@click.option("--api-key", default="", help="Qwen API key (or DASHSCOPE_API_KEY/QWEN_API_KEY).")
@click.option("--model", default="qwen-vl-max", show_default=True)
@click.option(
    "--base-url",
    default="https://dashscope.aliyuncs.com/compatible-mode/v1",
    show_default=True,
    help="OpenAI-compatible Qwen endpoint.",
)
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
    help="Minimum control ticks between (re)planning cycles.",
)
@click.option("--dt", default=0.15, show_default=True)
@click.option("--steps", default=120, show_default=True)
@click.option("--headless", is_flag=True)
@click.option("--save/--no-save", default=True, help="Save result plot.")
def main(
    instruction,
    scene,
    server_url,
    intrinsic,
    start,
    api_key,
    model,
    base_url,
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
    """QwenS2 — instruction-driven NavDP navigation demo."""
    if headless:
        import matplotlib

        matplotlib.use("Agg")
    scene_path = scene or _default_scene()
    fx, fy, cx, cy = (float(v) for v in intrinsic.split(","))
    intrinsic_k = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
    width, height = int(render_width), int(render_height)

    selector = QwenPixelGoalSelector(model=model, base_url=base_url, api_key=api_key or None)
    try:
        selector.validate_environment()
    except Exception as exc:
        raise click.ClickException(f"Qwen selector unavailable: {exc}") from exc

    client = NavDPClient(base_url=server_url)
    nav = NavigationController.from_scene_xml(
        scene_path,
        algorithm=Algorithm.QWENS2,
        planner_kwargs={
            "client": client,
            "selector": selector,
            "intrinsic": intrinsic_k,
            "instruction": instruction,
            "image_size": (width, height),
            "stop_threshold": float(stop_threshold),
            "plan_interval": int(plan_interval),
        },
    )
    nav.qwens2.tracker.max_v = float(max_v)
    nav.qwens2.tracker.max_w = float(max_w)
    nav.qwens2.tracker.goal_tol = float(goal_tol)

    robot = np.array([float(v) for v in start.split(",")] if start else _find_free_start(nav))
    yaw = 0.0
    capture = ImageCapture(nav.model, nav.data, width=width, height=height)

    print(f"\n{'='*60}")
    print("  QwenS2 — Instruction-driven NavDP Navigation")
    print(f"{'='*60}")
    print(f"Instruction: {instruction}")
    print(f"Qwen: {model} @ {base_url}")
    print(f"NavDP: {server_url}")
    print(f"Scene: {scene_path}")
    print(f"Start: ({robot[0]:.2f}, {robot[1]:.2f})  yaw={yaw:.2f}\n")

    path_history: list[np.ndarray] = [robot.copy()]
    reached = False
    stuck_steps = 0
    t0 = time.perf_counter()
    try:
        for step in range(steps):
            rgb_bytes, depth_m = capture.capture_at_robot(
                tuple(robot), yaw, camera_height=camera_height, pitch=pitch, fovy=fovy
            )
            v, w = nav.step_qwens2(robot, yaw, rgb_bytes, depth_m)

            yaw = math.atan2(math.sin(yaw + w * dt), math.cos(yaw + w * dt))
            robot = robot + v * np.array([math.cos(yaw), math.sin(yaw)]) * dt
            path_history.append(robot.copy())

            if nav.qwens2.diag.stop_requested:
                reached = True
                print(f"[{step:3d}] VLM says reached — STOP at ({robot[0]:.2f}, {robot[1]:.2f})")
                break

            if step > 0 and np.linalg.norm(robot - path_history[-2]) < 0.02:
                stuck_steps += 1
            else:
                stuck_steps = 0
            if stuck_steps >= 30:
                print(f"[{step:3d}] STUCK — no progress for 30 ticks at ({robot[0]:.2f}, {robot[1]:.2f})")
                break

            if step % 10 == 0:
                err = nav.qwens2.diag.last_error or "-"
                print(
                    f"[{step:3d}] pos=({robot[0]:.2f}, {robot[1]:.2f}) yaw={yaw:+.2f} "
                    f"v={v:+.2f} w={w:+.2f} pixel={nav.qwens2.diag.last_pixel} "
                    f"replan={nav.qwens2.diag.replan_count} err={err}"
                )
    finally:
        capture.close()

    elapsed = time.perf_counter() - t0
    print(
        f"\nDone in {elapsed:.1f}s | replans={nav.qwens2.diag.replan_count} | "
        f"reached={reached} | final pos=({robot[0]:.2f}, {robot[1]:.2f})"
    )

    if save:
        _save_plot(nav, np.array(path_history), robot, instruction, reached)


def _save_plot(
    nav: NavigationController,
    path: np.ndarray,
    final_pose: np.ndarray,
    instruction: str,
    reached: bool,
) -> None:
    """Save a top-down plot of the driven path over the occupancy grid."""
    occ = nav.grid.occupancy
    fig, ax = plt_figure()
    ax.imshow(
        occ.astype(float),
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
    ax.set_title(f"QwenS2 — {instruction[:40]}{'...' if len(instruction)>40 else ''} (reached={reached})")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.15)
    fig.tight_layout()
    out = Path(__file__).resolve().parent / "qwens2_demo.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Figure saved → {out}")


def plt_figure():
    import matplotlib.pyplot as plt

    return plt.subplots(figsize=(9, 8))


if __name__ == "__main__":
    main()
