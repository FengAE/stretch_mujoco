"""Inspect and query the office semantic world without starting the viewer."""

import json
from pathlib import Path

import click
import mujoco

from stretch_mujoco.semantics import InteractionRole, SemanticWorld


MODELS_PATH = Path(__file__).resolve().parents[1] / "stretch_mujoco" / "models"
OFFICE_SCENE_PATH = MODELS_PATH / "office_scene.xml"
OFFICE_SEMANTICS_PATH = MODELS_PATH / "office_semantics.json"


@click.command()
@click.option("--object-id", default="document_report", show_default=True)
def main(object_id: str) -> None:
    """Print an object's semantics and the office interaction-point poses."""
    model = mujoco.MjModel.from_xml_path(str(OFFICE_SCENE_PATH))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    world = SemanticWorld.from_json(OFFICE_SEMANTICS_PATH)
    world.validate_model(model)

    semantic_object = world.object(object_id)
    relations = world.find_relations(subject=object_id)
    grasp_points = world.interaction_points_for(role=InteractionRole.DOCUMENT_GRASP)
    output = {
        "object": {
            "id": semantic_object.object_id,
            "type": semantic_object.object_type.value,
            "attributes": semantic_object.attributes,
            "relations": [
                {"relation": relation.relation.value, "object": relation.object}
                for relation in relations
            ],
        },
        "pending_requests": [
            request.object_id for request in world.pending_requests("employee_01")
        ],
        "employee_01_can_access": world.can_access("employee_01", object_id),
        "stretch_3_can_access": world.can_access("stretch_3", object_id),
        "document_grasp_points": {
            point.point_id: world.interaction_pose(point.point_id, model, data)[:3, 3].tolist()
            for point in grasp_points
        },
    }
    click.echo(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
