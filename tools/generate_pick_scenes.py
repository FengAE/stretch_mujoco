#!/usr/bin/env python3
"""Generate diverse pick scenes: varied table textures + random object layouts.

Each generated scene is a self-contained MJCF in ``models/assets/pick_scene/``
that loads the Stretch robot, a table with one RoboTwin-derived texture, and the
12 graspable objects placed at a random-but-stable layout on the table. Layouts
are accepted only if they settle with every object still on the table.

The table surface can also be switched at runtime via
``graspgen.objects.build_grasp_scene(objects, table_texture=...)``.

Usage:
  uv run python tools/generate_pick_scenes.py
  uv run python tools/generate_pick_scenes.py --textures dark,light
  uv run python tools/generate_pick_scenes.py --seed 7 --out-dir models/pick_scene
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from stretch_mujoco.graspgen.objects import (  # noqa: E402
    DOCKING_STATION_XML,
    GRASP_OBJECTS_DIR,
    MODELS_DIR,
    STRETCH_XML,
    TABLE_TEXTURES,
    available_objects,
    build_grasp_scene,
    place_on_table,
    randomize_positions,
)

DEFAULT_OUT_DIR = MODELS_DIR / "assets" / "pick_scene"
DEFAULT_TEXTURES = [
    name for name in sorted(TABLE_TEXTURES) if name != "wood"
]

_SCENE_TEMPLATE = """<mujoco model="{model}">
  <include file="_stretch_fixed.xml"/>
  <include file="_docking_fixed.xml"/>

  <statistic center="0 0 .75" extent="1.2" meansize="0.05"/>

  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="-120" elevation="-20" offwidth="960" offheight="540"/>
  </visual>

  <asset>
    <material name="floor" rgba=".1 .1 .1 1" reflectance="0.1"/>
    <texture type="2d" name="table_tex" file="{table_tex}"/>
    <material name="table_mat" texture="table_tex" texrepeat="1 1"/>
{object_assets}
  </asset>

  <worldbody>
    <light pos="0 0 1.5" dir="0 0 -1" directional="true"/>
    <camera name="office_overview" mode="targetbody" target="table"
            pos="1.35 0.25 1.45" fovy="55"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="floor"/>
    <body name="table" pos="0 -1 .24">
      <geom type="box" size=".6 .5 .24" mass="1" material="table_mat"/>
    </body>
{object_bodies}
  </worldbody>
</mujoco>
"""


def _object_asset_and_body(
    object_name: str,
    pose: tuple[tuple[float, float, float], tuple[float, float, float, float]],
) -> tuple[str, str]:
    """Extract the <asset> and <body> XML from an object MJCF for inlining.

    Rewrites mesh/texture file paths to be relative to the pick_scene dir and
    sets the free-body initial position + quaternion to *pose*.
    """
    root = ET.parse(GRASP_OBJECTS_DIR / object_name / f"{object_name}.xml").getroot()
    # Paths are relative to the model-wide assetdir (models/assets), which is set
    # by the included _stretch_fixed.xml compiler.
    prefix = f"grasp_objects/{object_name}"

    asset_elem = root.find("asset")
    for elem in asset_elem.iter():
        if "file" in elem.attrib:
            elem.attrib["file"] = f"{prefix}/{Path(elem.attrib['file']).name}"

    (x, y, z), (w, qx, qy, qz) = pose
    body = root.find("worldbody/body")
    body.set("pos", f"{x:.4f} {y:.4f} {z:.4f}")
    body.set("quat", f"{w:.6f} {qx:.6f} {qy:.6f} {qz:.6f}")

    assets = "".join(
        "    " + ET.tostring(child, encoding="unicode").strip() + "\n"
        for child in asset_elem
    )
    body_xml = ET.tostring(body, encoding="unicode").strip()
    return assets.rstrip("\n"), body_xml


def _settled_layout(
    object_names: list[str], rng: np.random.Generator
) -> dict[str, tuple[tuple[float, float, float], tuple[float, float, float, float]]]:
    """Sample a random layout, settle it, and return each object's REST pose.

    The initial drop has all objects land at once, so round ones can bump and
    roll. Baking the POST-settle rest pose (position + quaternion) into the scene
    means a freshly loaded scene starts fully at rest — nothing drops, nothing
    rolls, regardless of friction. Layouts where anything falls off the table
    during the settle are rejected and re-sampled.
    """
    for _ in range(30):
        positions = randomize_positions(object_names, rng)
        model = build_grasp_scene(object_names)
        place_on_table(model, positions)
        data = mujoco.MjData(model)
        for _ in range(2500):  # 5 s settle
            mujoco.mj_step(model, data)
        if all(
            _on_table(data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)])
            for name in object_names
        ):
            poses: dict[
                str, tuple[tuple[float, float, float], tuple[float, float, float, float]]
            ] = {}
            for name in object_names:
                body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
                qpos = int(model.jnt_qposadr[int(model.body_jntadr[body_id])])
                poses[name] = (
                    tuple(float(v) for v in data.xpos[body_id]),
                    tuple(float(v) for v in data.qpos[qpos + 3 : qpos + 7]),
                )
            return poses
    raise RuntimeError("could not find a stable random layout")


def _on_table(xyz) -> bool:
    x, y, z = xyz
    return z > 0.4 and abs(x) < 0.62 and -1.52 < y < -0.48


def _write_robot_fixes(out_dir: Path) -> None:
    """Write assetdir-corrected copies of the robot XMLs into *out_dir*.

    MuJoCo resolves an included file's ``assetdir`` relative to the *including*
    file's directory, so including ``models/stretch.xml`` from a nested folder
    breaks its ``assets/`` paths. These copies retarget ``assetdir`` at
    ``models/assets`` and are included as siblings by every pick scene.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for source, target in (
        (STRETCH_XML, out_dir / "_stretch_fixed.xml"),
        (DOCKING_STATION_XML, out_dir / "_docking_fixed.xml"),
    ):
        text = source.read_text().replace('assetdir="assets"', 'assetdir="../../assets"')
        target.write_text(text)


def generate_scene(
    object_names: list[str],
    texture: str,
    poses: dict[str, tuple[tuple[float, float, float], tuple[float, float, float, float]]],
    out_dir: Path,
) -> Path:
    """Write one self-contained pick scene XML and return its path.

    Each object body is baked at its POST-settle rest pose (position + quat), so
    loading the scene starts everything at rest.
    """
    _write_robot_fixes(out_dir)
    # Relative to the model-wide assetdir (models/assets).
    table_tex = f"table_textures/{Path(TABLE_TEXTURES[texture]).name}"

    object_assets: list[str] = []
    object_bodies: list[str] = []
    for name in object_names:
        assets, body = _object_asset_and_body(name, poses[name])
        object_assets.append(assets)
        object_bodies.append("    " + body)

    xml = _SCENE_TEMPLATE.format(
        model=f"pick_scene_{texture}",
        table_tex=table_tex,
        object_assets="\n".join(object_assets),
        object_bodies="\n".join(object_bodies),
    )
    path = out_dir / f"pick_scene_{texture}.xml"
    path.write_text(xml + "\n")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--textures",
        default=",".join(DEFAULT_TEXTURES),
        help="comma-separated table textures (default: all non-wood)",
    )
    parser.add_argument(
        "--objects",
        default=None,
        help="comma-separated object ids (default: all converted objects)",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="output directory"
    )
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for layouts")
    args = parser.parse_args()

    object_names = (
        [name.strip() for name in args.objects.split(",") if name.strip()]
        if args.objects
        else available_objects()
    )
    textures = [name.strip() for name in args.textures.split(",") if name.strip()]
    missing = [name for name in textures if name not in TABLE_TEXTURES]
    if missing:
        raise SystemExit(f"unknown textures: {missing}; choose from {sorted(TABLE_TEXTURES)}")

    rng = np.random.default_rng(args.seed)
    for i, texture in enumerate(textures, 1):
        poses = _settled_layout(object_names, rng)
        path = generate_scene(object_names, texture, poses, args.out_dir)
        layout = ", ".join(
            f"{name}=({x:.2f},{y:.2f})" for name, ((x, y, _), _) in poses.items()
        )
        print(f"[{i}/{len(textures)}] {texture}: {path.name}")
        print(f"    layout: {layout}")
    print(f"done -> {args.out_dir}")


if __name__ == "__main__":
    main()
