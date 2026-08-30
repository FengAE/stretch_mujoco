"""Build MuJoCo scenes from converted graspable objects.

Objects converted by ``tools/convert_robotwin_objects.py`` live in
``models/assets/grasp_objects/<name>/`` as ``<include>``-able MJCFs. This module
assembles a scene with the Stretch robot and table (same layout as
``models/scene.xml``) plus any subset of those objects, and places them on the
table via their free-joint initial position (qpos0), mirroring the
``_set_free_body_position`` pattern from ``examples/collect_random_grasp_episodes.py``.

Grasp validation is keyed off body names (``mj_name2id`` for ``mjOBJ_BODY``), so
any body placed here is automatically trackable via ``request_grasp_metrics``.
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile

import mujoco
import numpy as np

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
GRASP_OBJECTS_DIR = MODELS_DIR / "assets" / "grasp_objects"
TABLE_TEXTURES_DIR = MODELS_DIR / "assets" / "table_textures"
STRETCH_XML = MODELS_DIR / "stretch.xml"
DOCKING_STATION_XML = MODELS_DIR / "docking_station.xml"
WOOD_TEXTURE = MODELS_DIR / "assets" / "wood.png"

# Table textures copied from RoboTwin's background_texture set; "wood" is the
# stock scene.xml material.
TABLE_TEXTURES: dict[str, Path] = {
    "wood": WOOD_TEXTURE,
    **{path.stem: path for path in sorted(TABLE_TEXTURES_DIR.glob("*.png"))},
}

# scene.xml table: body at (0, -1, 0.24) with box half-height 0.24 -> top at z=0.48.
TABLE_TOP_Z = 0.48

_OBJECT_XML = """<mujoco model="grasp scene">
  <include file="{stretch}"/>
  <include file="{docking}"/>
  {objects}

  <statistic center="0 0 .75" extent="1.2" meansize="0.05"/>

  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="-120" elevation="-20" offwidth="960" offheight="540"/>
  </visual>

  <asset>
    <material name="floor" rgba=".1 .1 .1 1" reflectance="0.1"/>
    {table_asset}
  </asset>

  <worldbody>
    <light pos="0 0 1.5" dir="0 0 -1" directional="true"/>
    <camera name="office_overview" mode="targetbody" target="table"
            pos="1.35 0.25 1.45" fovy="55"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="floor"/>
    <body name="table" pos="0 -1 .24">
      <geom type="box" size=".6 .5 .24" mass="1" material="{table_material}"/>
    </body>
  </worldbody>
</mujoco>
"""


def available_objects() -> list[str]:
    """Return the names of all converted graspable objects."""
    return sorted(dir.name for dir in GRASP_OBJECTS_DIR.iterdir() if dir.is_dir())


def load_metadata(object_name: str) -> dict:
    """Return the placement metadata written by the converter for *object_name*."""
    path = GRASP_OBJECTS_DIR / object_name / "metadata.json"
    if not path.exists():
        raise FileNotFoundError(
            f"object {object_name!r} not converted; run "
            "tools/convert_robotwin_objects.py first"
        )
    return json.loads(path.read_text())


def table_textures() -> list[str]:
    """Return the names of available table materials (textures)."""
    return list(TABLE_TEXTURES)


def build_grasp_scene(
    object_names: list[str],
    table_texture: str = "wood",
) -> mujoco.MjModel:
    """Build an MjModel with the Stretch robot, table, and *object_names*.

    ``table_texture`` selects the table surface material (see :func:`table_textures`).
    Objects are included as free-jointed bodies at their local origin; call
    :func:`place_on_table` afterwards to set their resting positions.
    """
    missing = [name for name in object_names if not (GRASP_OBJECTS_DIR / name).is_dir()]
    if missing:
        raise FileNotFoundError(
            f"objects not converted: {missing}; run tools/convert_robotwin_objects.py"
        )
    texture_path = TABLE_TEXTURES.get(table_texture)
    if texture_path is None:
        raise ValueError(
            f"unknown table texture {table_texture!r}; choose from {table_textures()}"
        )
    includes = "\n".join(
        f'<include file="{(GRASP_OBJECTS_DIR / name / (name + ".xml")).resolve()}"/>'
        for name in object_names
    )
    table_asset = (
        f'<texture type="2d" name="table_tex" file="{texture_path.resolve()}"/>\n'
        f'    <material name="table_mat" texture="table_tex" texrepeat="1 1"/>'
    )
    table_material = "table_mat"
    xml = _OBJECT_XML.format(
        stretch=STRETCH_XML.resolve(),
        docking=DOCKING_STATION_XML.resolve(),
        objects=includes,
        table_asset=table_asset,
        table_material=table_material,
    )
    # Load via a temp file inside models/ so relative assetdir="assets" and
    # mesh paths resolve exactly like the stock scene.xml. Assets are compiled
    # into the MjModel, so the temp file is unlinked right after.
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".xml", dir=MODELS_DIR, delete=False
    ) as fh:
        fh.write(xml)
        temp_path = Path(fh.name)
    try:
        return mujoco.MjModel.from_xml_path(str(temp_path))
    finally:
        temp_path.unlink(missing_ok=True)


def set_body_position(
    model: mujoco.MjModel, body_name: str, xyz: tuple[float, float, float]
) -> None:
    """Set the free-joint initial position of a named body via qpos0."""
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        raise ValueError(f"unknown body: {body_name}")
    joint_id = int(model.body_jntadr[body_id])
    if model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_FREE:
        raise ValueError(f"body {body_name} has no free joint")
    qpos = int(model.jnt_qposadr[joint_id])
    model.qpos0[qpos : qpos + 3] = np.asarray(xyz, dtype=float)


def place_on_table(
    model: mujoco.MjModel,
    positions: dict[str, tuple[float, float]],
) -> None:
    """Place each named object so its lowest vertex rests on the table top.

    ``positions`` maps object name -> (x, y); z is derived from the object's
    converted mesh bounds (``metadata.min_z``).
    """
    for name, (x, y) in positions.items():
        meta = load_metadata(name)
        z = TABLE_TOP_Z - float(meta["min_z"])
        set_body_position(model, name, (x, y, z))


def default_positions(object_names: list[str]) -> dict[str, tuple[float, float]]:
    """Spread objects in rows across the table (table spans x~[-.6,.6], y~[-1.5,-.5]).

    30 cm spacing keeps even the wide keyboard (45 cm) from piling onto neighbors.
    """
    positions: dict[str, tuple[float, float]] = {}
    for index, name in enumerate(object_names):
        row = index // 4
        col = index % 4
        x = -0.45 + col * 0.30
        y = -0.65 - row * 0.30
        positions[name] = (x, y)
    return positions


def randomize_positions(
    object_names: list[str],
    rng: np.random.Generator | None = None,
) -> dict[str, tuple[float, float]]:
    """Sample random (x, y) table positions that keep objects from overlapping.

    Uses rejection sampling with a margin from the table edges and a separation
    that scales with each object's footprint (the wide keyboard needs more room).
    """
    if rng is None:
        rng = np.random.default_rng()
    footprints: dict[str, float] = {}
    for name in object_names:
        extents = load_metadata(name)["extents"]
        footprints[name] = max(
            extents[0][1] - extents[0][0], extents[1][1] - extents[1][0]
        )
    positions: dict[str, tuple[float, float]] = {}
    for name in object_names:
        for _ in range(300):
            x = float(rng.uniform(-0.40, 0.40))
            y = float(rng.uniform(-1.30, -0.75))
            if all(
                (x - px) ** 2 + (y - py) ** 2
                > (0.15 + footprints[name] / 2 + footprints[other] / 2) ** 2
                for other, (px, py) in positions.items()
            ):
                break
        positions[name] = (x, y)
    return positions


def randomize_and_place(
    model: mujoco.MjModel,
    object_names: list[str],
    rng: np.random.Generator | None = None,
) -> dict[str, tuple[float, float]]:
    """Randomize object positions on the table and apply them to *model*."""
    positions = randomize_positions(object_names, rng)
    place_on_table(model, positions)
    return positions
