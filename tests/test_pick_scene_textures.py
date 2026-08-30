import json
from pathlib import Path

import mujoco


ASSETS = (
    Path(__file__).resolve().parents[1] / "stretch_mujoco" / "models" / "assets"
)


def test_textured_grasp_objects_keep_their_uv_coordinates() -> None:
    model = mujoco.MjModel.from_xml_path(
        str(ASSETS / "pick_scene" / "pick_scene_dark.xml")
    )

    textured = 0
    for object_dir in (ASSETS / "grasp_objects").iterdir():
        metadata = json.loads((object_dir / "metadata.json").read_text())
        if not metadata["textured"]:
            continue
        textured += 1
        mesh_name = f"{object_dir.name.replace('-', '_')}_visual"
        mesh = model.mesh(mesh_name)
        assert model.mesh_texcoordnum[mesh.id] == model.mesh_vertnum[mesh.id]

    assert textured == 12
