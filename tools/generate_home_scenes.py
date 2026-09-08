"""Generate ten simplified home scenes from ten distinct HSSD scene IDs.

Unlike ``generate_office_scenes.py``, this script does not reuse one hand-made
layout with furniture permutations.  Each output scene is a conversion of a
different HSSD scene instance (the uncluttered split), with a Stretch robot
included and a manifest that records the source HSSD ID.

The converted HSSD assets are cached below the output directory for reproducible
reruns. HSSD itself is not redistributed by this repository; pass
``--hssd-root`` when it is installed somewhere else.
"""

from __future__ import annotations

import json
import math
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import click
import cv2
import mujoco
import numpy as np

from stretch_mujoco.habitat_scene_gallery import (
    DEFAULT_HSSD_ROOT,
    prepare_habitat_scene,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_ROOT = PROJECT_ROOT / "stretch_mujoco" / "models"
STRETCH_XML = MODELS_ROOT / "stretch.xml"
DEFAULT_OUTPUT = MODELS_ROOT / "assets" / "home_scenes"

# These are deliberately different HSSD scene IDs.  They cover compact homes,
# apartments, and larger multi-room interiors while keeping the uncluttered
# object counts practical for navigation and manipulation experiments.
DEFAULT_SCENE_IDS = (
    "102344193",
    "106365897_174225972",
    "103997403_171030405",
    "104348133_171513054",
    "105515307_173104317",
    "105515490_173104566",
    "102815859",
    "102344250",
    "104862513_172226580",
    "108294897_176710602",
)


def _numbers(values: tuple[float, ...] | np.ndarray) -> str:
    return " ".join(f"{float(value):.9g}" for value in values)


def _scene_name(index: int, hssd_id: str) -> str:
    return f"home_{index:02d}_{hssd_id}"


def _write_robot_include(
    output_dir: Path, scene_name: str, bounds: np.ndarray, yaw: float
) -> tuple[Path, tuple[float, float, float]]:
    """Write a robot-only include and choose a deterministic open-area start.

    HSSD scene geoms are visual-only, so this start is intentionally a simple
    geometric choice near the scene center.  It is recorded in the manifest and
    can be changed later without regenerating the converted HSSD assets.
    """
    tree = ET.parse(STRETCH_XML)
    root = tree.getroot()
    compiler = root.find("compiler")
    if compiler is None:
        raise RuntimeError("stretch.xml has no compiler element")
    compiler.set("assetdir", str((MODELS_ROOT / "assets").resolve()))
    base = root.find("./worldbody/body[@name='base_link']")
    if base is None:
        raise RuntimeError("stretch.xml has no base_link body")
    lo, hi = np.asarray(bounds[0], dtype=float), np.asarray(bounds[1], dtype=float)
    # Stay away from the outer shell while retaining a stable, reproducible
    # position for all source scenes.
    xy = lo[:2] + 0.50 * (hi[:2] - lo[:2])
    start = (float(xy[0]), float(xy[1]), float(yaw))
    base.set("pos", _numbers((start[0], start[1], 0.0)))
    base.set("quat", _numbers((math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))))
    path = output_dir / f"{scene_name}_robot.xml"
    ET.indent(root, space="  ")
    tree.write(path, encoding="unicode", xml_declaration=False)
    return path, start


def _add_robot_include(scene_xml: Path, robot_xml: Path, scene_name: str) -> None:
    tree = ET.parse(scene_xml)
    root = tree.getroot()
    root.set("model", scene_name)
    # The include is placed before the scene compiler, matching the existing
    # office generator and allowing Stretch's defaults/assets to be merged.
    root.insert(0, ET.Element("include", {"file": robot_xml.name}))
    ET.indent(root, space="  ")
    tree.write(scene_xml, encoding="unicode", xml_declaration=False)


def _render_preview(xml_path: Path, output_path: Path) -> dict[str, int]:
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    camera = "habitat_top" if mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_CAMERA, "habitat_top"
    ) >= 0 else -1
    renderer = mujoco.Renderer(model, height=480, width=640)
    renderer.update_scene(data, camera=camera)
    image = renderer.render()
    renderer.close()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    return {"textures": model.ntex, "materials": model.nmat, "meshes": model.nmesh, "geoms": model.ngeom}


def _write_readme(output_dir: Path) -> None:
    (output_dir / "README.md").write_text(
        """# Generated simplified home scenes

Ten scenes generated from ten different HSSD scene IDs. The script uses the
HSSD `scenes-uncluttered` split, converts each stage/object to MuJoCo, and adds
a Hello Robot Stretch include. Converted meshes and textures are stored in
`_hssd_cache/`.

```bash
python examples/generated_home_scene.py --scene 1
```

Regenerate with a different dataset location using:

```bash
python tools/generate_home_scenes.py --hssd-root /path/to/hssd-hab
```
""",
        encoding="utf-8",
    )


def generate(
    output: Path,
    hssd_root: Path,
    scene_ids: tuple[str, ...],
    *,
    skip_previews: bool,
    rebuild: bool,
    ktx_command: Path | None,
) -> None:
    if len(scene_ids) != 10:
        raise click.ClickException(f"Expected exactly 10 HSSD scene IDs, got {len(scene_ids)}")
    if len(set(scene_ids)) != len(scene_ids):
        raise click.ClickException("HSSD scene IDs must be unique")
    hssd_root = hssd_root.resolve()
    missing = [
        scene_id
        for scene_id in scene_ids
        if not (hssd_root / "scenes-uncluttered" / f"{scene_id}.scene_instance.json").is_file()
    ]
    if missing:
        raise click.ClickException(
            "Missing HSSD scenes-uncluttered files for: " + ", ".join(missing)
        )
    output.mkdir(parents=True, exist_ok=True)
    for old_file in output.glob("home_*.*"):
        old_file.unlink()
    cache_root = output / "_hssd_cache"
    if rebuild and cache_root.exists():
        shutil.rmtree(cache_root)

    catalog = []
    for index, hssd_id in enumerate(scene_ids, start=1):
        scene_name = _scene_name(index, hssd_id)
        click.echo(f"Generating {scene_name} from HSSD {hssd_id}")
        prepared = prepare_habitat_scene(
            scene_id=hssd_id,
            hssd_root=hssd_root,
            cache_root=cache_root,
            # This is the simplification knob: remove clutter while retaining
            # the architectural stage and primary furniture.
            uncluttered=True,
            stage_alpha=1.0,
            include_stage=True,
            include_collision=True,
            include_grasp_sites=True,
            rebuild=rebuild,
            ktx_command=ktx_command,
        )
        scene_xml = output / f"{scene_name}.xml"
        shutil.copy2(prepared.xml_path, scene_xml)
        yaw = (index - 1) % 4 * (math.pi / 2)
        robot_xml, robot_start = _write_robot_include(output, scene_name, prepared.bounds, yaw)
        _add_robot_include(scene_xml, robot_xml, scene_name)

        manifest = {
            "scene_id": scene_name,
            "title": f"Simplified Home {index:02d}",
            "source_dataset": "HSSD",
            "hssd_scene_id": hssd_id,
            "hssd_scene_split": "scenes-uncluttered",
            "hssd_root": str(hssd_root),
            "layout": "imported_hssd_home",
            "simplification": {"uncluttered": True, "stage_alpha": 1.0},
            "object_count": prepared.object_count,
            "unique_template_count": prepared.unique_template_count,
            "category_counts": prepared.category_counts,
            "bounds_mujoco": prepared.bounds.tolist(),
            "zones": [
                {
                    "type": "home",
                    "bounds": [
                        float(prepared.bounds[0, 0]),
                        float(prepared.bounds[1, 0]),
                        float(prepared.bounds[0, 1]),
                        float(prepared.bounds[1, 1]),
                    ],
                }
            ],
            "assets": prepared.object_records,
            "robot": {
                "model": "Hello Robot Stretch",
                "initial_pose": list(robot_start),
                "body": "base_link",
            },
            "mjcf": scene_xml.name,
            "robot_include": robot_xml.name,
            "preview": f"{scene_name}.png",
        }
        manifest_path = output / f"{scene_name}.json"
        if not skip_previews:
            manifest["model"] = _render_preview(scene_xml, output / manifest["preview"])
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        catalog.append(
            {
                "scene_id": scene_name,
                "hssd_scene_id": hssd_id,
                "title": manifest["title"],
                "mjcf": scene_xml.name,
                "manifest": manifest_path.name,
                "preview": manifest["preview"],
                "object_count": prepared.object_count,
                "robot_initial_pose": list(robot_start),
            }
        )

    (output / "catalog.json").write_text(
        json.dumps(
            {
                "scene_count": 10,
                "source_dataset": "HSSD",
                "scene_split": "scenes-uncluttered",
                "scenes": catalog,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_readme(output)
    click.echo(f"Generated {len(catalog)} scenes in {output}")


@click.command()
@click.option("--output", type=click.Path(path_type=Path), default=DEFAULT_OUTPUT, show_default=True)
@click.option("--hssd-root", type=click.Path(path_type=Path), default=DEFAULT_HSSD_ROOT, show_default=True)
@click.option(
    "--scene-id",
    "scene_ids",
    multiple=True,
    help="HSSD scene ID; repeat exactly ten times. Defaults to the curated home list.",
)
@click.option("--skip-previews", is_flag=True, help="Skip EGL preview rendering.")
@click.option("--rebuild", is_flag=True, help="Discard converted HSSD cache before generating.")
@click.option("--ktx-command", type=click.Path(path_type=Path), help="Path to Khronos ktx for BasisU textures.")
def main(
    output: Path,
    hssd_root: Path,
    scene_ids: tuple[str, ...],
    skip_previews: bool,
    rebuild: bool,
    ktx_command: Path | None,
) -> None:
    """Generate ten simplified homes, each from a distinct HSSD scene ID."""
    generate(
        output,
        hssd_root,
        tuple(scene_ids) or DEFAULT_SCENE_IDS,
        skip_previews=skip_previews,
        rebuild=rebuild,
        ktx_command=ktx_command,
    )


if __name__ == "__main__":
    main()
