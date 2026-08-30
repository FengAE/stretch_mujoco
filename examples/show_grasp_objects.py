"""Render imported RoboTwin objects on the table in the Stretch MuJoCo sim.

Builds a scene with converted graspable objects, places them on the table,
starts the simulator, verifies grasp metrics track a named object, and saves
head/wrist/overview camera frames to an output directory for visual inspection.
This is also a template for pointing the collection pipeline at these objects.

Usage:
  env MUJOCO_GL=egl uv run python examples/show_grasp_objects.py
  env MUJOCO_GL=egl uv run python examples/show_grasp_objects.py \
      --objects 071_can,035_apple --out-dir /tmp/obj_shots
"""

from __future__ import annotations

import os
from pathlib import Path
import time

os.environ.setdefault("MUJOCO_GL", "egl")

import click
import cv2
import numpy as np

from examples.evaluate_openpi_policy import CAMERAS, _enter_manipulation_mode
from stretch_mujoco import StretchMujocoSimulator
from stretch_mujoco.enums.stretch_cameras import StretchCameras
from stretch_mujoco.graspgen.objects import (
    available_objects,
    build_grasp_scene,
    default_positions,
    place_on_table,
)
from stretch_mujoco.openpi_contract import FPS


def _save_camera_frame(
    sim: StretchMujocoSimulator, camera: StretchCameras, path: Path, label: str
) -> None:
    """Save one RGB frame from *camera* as an annotated PNG."""
    frame = sim.pull_camera_data().get_camera_data(
        camera, auto_rotate=True, auto_correct_rgb=False
    )
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (overlay.shape[1], 26), (20, 24, 28), -1)
    cv2.putText(
        overlay, label, (7, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1
    )
    cv2.imwrite(str(path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    print(f"  wrote {path}")


@click.command()
@click.option(
    "--objects",
    default=None,
    help="comma-separated object ids (default: all converted objects)",
)
@click.option(
    "--out-dir",
    type=click.Path(path_type=Path, file_okay=False),
    default=Path("evaluations/grasp_objects"),
)
@click.option("--target", default="071_can", help="object name to check grasp metrics on")
def main(objects: str | None, out_dir: Path, target: str) -> None:
    """Render the imported graspable objects and verify body-name tracking."""
    names = [name.strip() for name in objects.split(",")] if objects else available_objects()
    if not names:
        raise SystemExit("no converted objects found; run tools/convert_robotwin_objects.py")
    if target not in names:
        print(f"note: target {target!r} not in scene; using {names[0]!r}")
        target = names[0]

    model = build_grasp_scene(names)
    place_on_table(model, default_positions(names))

    sim = StretchMujocoSimulator(model=model, camera_hz=FPS, cameras_to_use=CAMERAS)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        sim.start(headless=True)
        sim.set_robot_motion_speed(3.0)
        time.sleep(1.5)
        _enter_manipulation_mode(sim)
        time.sleep(0.3)

        # Confirm a named object body is trackable by grasp metrics.
        sim.request_grasp_metrics(target)
        time.sleep(0.3)
        metrics = sim.pull_grasp_metrics()
        print(
            f"grasp metrics for {target!r}: "
            f"object_id={metrics.get('object_id')} "
            f"position={np.asarray(metrics.get('object_position')).round(3).tolist()}"
        )
        if metrics.get("object_id") != target:
            raise RuntimeError(f"grasp metrics did not track {target!r}: {metrics}")

        _save_camera_frame(
            sim, StretchCameras.cam_d435i_rgb, out_dir / "head.png", "HEAD D435i"
        )
        _save_camera_frame(
            sim, StretchCameras.cam_d405_rgb, out_dir / "wrist.png", "WRIST D405"
        )
        _save_camera_frame(
            sim,
            StretchCameras.office_overview_rgb,
            out_dir / "overview.png",
            "THIRD PERSON | objects on table",
        )
    finally:
        if sim.is_running():
            sim.stop()

    print(f"objects in scene: {', '.join(names)}")
    print(f"done -> {out_dir}")


if __name__ == "__main__":
    main()
