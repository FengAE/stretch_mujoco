"""Display the shortlisted HSSD office assets together in a MuJoCo gallery."""

from __future__ import annotations

import time
from pathlib import Path

import click
import mujoco
import mujoco.viewer

from stretch_mujoco.office_asset_gallery import DEFAULT_CACHE, load_gallery_model, prepare_gallery


@click.command()
@click.option("--headless", is_flag=True, help="Build and validate the gallery without a window.")
@click.option(
    "--real-scale", is_flag=True, help="Preserve real-world scale instead of display scale."
)
@click.option("--per-category", type=click.IntRange(1, 3), default=3, show_default=True)
@click.option(
    "--rebuild", is_flag=True, help="Rebuild permanent OBJ and PNG office asset packages."
)
@click.option(
    "--cache-dir", type=click.Path(path_type=Path), default=DEFAULT_CACHE, show_default=True
)
@click.option(
    "--ktx-command",
    type=click.Path(path_type=Path),
    default=None,
    help="Path to the Khronos ktx executable used to decode BasisU textures.",
)
def main(
    headless: bool,
    real_scale: bool,
    per_category: int,
    rebuild: bool,
    cache_dir: Path,
    ktx_command: Path | None,
) -> None:
    """Build a textured comparison gallery without modifying office_scene.xml."""
    gallery = prepare_gallery(
        cache_dir=cache_dir,
        per_category=per_category,
        real_scale=real_scale,
        rebuild=rebuild,
        ktx_command=ktx_command,
    )
    model = load_gallery_model(gallery)
    data = mujoco.MjData(model)

    click.echo(f"Gallery: {gallery.xml_path}")
    click.echo(f"Assets: {len(gallery.assets)}")
    for index, (asset, scale) in enumerate(
        zip(gallery.assets, gallery.display_scales, strict=True), start=1
    ):
        click.echo(f"{index:02d}  {asset.category:22}  scale={scale:5.2f}  {asset.display_name}")
    if headless:
        click.echo(
            f"Compiled MuJoCo model with {model.ntex} textures, {model.nmat} materials, "
            f"{model.nmesh} meshes and {model.ngeom} geoms."
        )
        return

    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "gallery_overview")
    with mujoco.viewer.launch_passive(
        model, data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        viewer.cam.fixedcamid = camera_id
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.01)


if __name__ == "__main__":
    main()
