"""Open one of the generated simplified HSSD home scenes."""

from __future__ import annotations

import json
import time
from pathlib import Path

import click
import mujoco
import mujoco.viewer


SCENE_ROOT = Path(__file__).resolve().parents[1] / "stretch_mujoco" / "models" / "assets" / "home_scenes"


@click.command()
@click.option("--scene", type=click.IntRange(1, 10), default=1, show_default=True)
@click.option("--top-view", is_flag=True, help="Start with the top-down camera.")
@click.option("--headless", is_flag=True, help="Compile the scene without opening a window.")
def main(scene: int, top_view: bool, headless: bool) -> None:
    """Interactively inspect a generated home scene."""
    catalog = json.loads((SCENE_ROOT / "catalog.json").read_text(encoding="utf-8"))
    entries = catalog.get("scenes", [])
    if len(entries) != 10:
        raise click.ClickException(f"Expected 10 generated scenes in {SCENE_ROOT}, found {len(entries)}")
    path = SCENE_ROOT / entries[scene - 1]["mjcf"]
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    click.echo(f"Loaded {path.name}: {model.ntex} textures, {model.nmesh} meshes, {model.ngeom} geoms")
    if headless:
        return
    with mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.cam.lookat[:] = model.stat.center
        viewer.cam.distance = model.stat.extent * (1.05 if top_view else 1.25)
        viewer.cam.azimuth = 90 if top_view else 135
        viewer.cam.elevation = -89 if top_view else -32
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.01)


if __name__ == "__main__":
    main()
