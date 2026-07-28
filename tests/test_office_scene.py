from pathlib import Path

import mujoco
import pytest

from stretch_mujoco.mujoco_server import MujocoServer


OFFICE_SCENE_PATH = (
    Path(__file__).resolve().parents[1] / "stretch_mujoco" / "models" / "office_scene.xml"
)
SNACKS = ("soda_can", "cereal_box", "bread_snack", "lemon")


def test_office_scene_loads_with_robot_and_furniture() -> None:
    model = mujoco.MjModel.from_xml_path(str(OFFICE_SCENE_PATH))

    for body_name in (
        "base_link",
        "snack_counter",
        "workstation_left",
        "workstation_right",
        "humanoid_preview",
    ):
        assert model.body(body_name).id >= 0

    humanoid_visual = model.geom("humanoid_preview_frame_idle_00_body")
    assert model.geom_type[humanoid_visual.id] == mujoco.mjtGeom.mjGEOM_MESH
    assert model.site("chair_left_sit").id >= 0
    assert model.site("chair_right_sit").id >= 0
    assert model.camera("office_overview").id >= 0


def test_human_navigation_sites_are_outside_furniture_footprints() -> None:
    model = mujoco.MjModel.from_xml_path(str(OFFICE_SCENE_PATH))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    meeting_gap = (
        data.body("meeting_table").xpos[1] - 0.38 - data.site("meeting_human_stand_site").xpos[1]
    )
    snack_gap = data.site("snack_human_stand_site").xpos[1] - (
        data.body("snack_counter").xpos[1] - 0.38
    )

    assert meeting_gap >= 0.35
    assert snack_gap <= -0.30


def test_all_snacks_are_free_and_graspable() -> None:
    model = mujoco.MjModel.from_xml_path(str(OFFICE_SCENE_PATH))

    for snack_name in SNACKS:
        body = model.body(snack_name)
        visual = model.geom(f"{snack_name}_visual")
        joint_id = body.jntadr[0]
        assert body.jntnum == 1
        assert model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE
        assert model.body_mass[body.id] > 0
        assert model.geom_type[visual.id] == mujoco.mjtGeom.mjGEOM_MESH


class _VisibilityProxy:
    def __init__(self, object_id: str, visible: bool) -> None:
        self.visibility = {object_id: visible}

    def get_object_visibility(self) -> dict[str, bool]:
        return self.visibility


class _GraspProxy:
    def __init__(self, object_id: str) -> None:
        self.object_id = object_id

    def get_grasped_object(self) -> str:
        return self.object_id

    def set_grasped_object(self, object_id: str) -> None:
        self.object_id = object_id


class _GraspMetricsProxy:
    def __init__(self, object_id: str) -> None:
        self.object_id = object_id
        self.metrics = {}

    def get_grasp_validation_target(self) -> str:
        return self.object_id

    def set_grasp_metrics(self, metrics: dict) -> None:
        self.metrics = metrics


def test_consumed_snack_is_hidden_and_has_no_collision() -> None:
    model = mujoco.MjModel.from_xml_path(str(OFFICE_SCENE_PATH))
    proxy = _VisibilityProxy("bread_snack", False)
    server = MujocoServer.__new__(MujocoServer)
    server.mjmodel = model
    server.data_proxies = proxy
    server._object_visibility_state = {}
    server._object_geom_defaults = {}
    body_id = model.body("bread_snack").id
    geom_ids = [
        geom_id for geom_id in range(model.ngeom) if int(model.geom_bodyid[geom_id]) == body_id
    ]
    original = {
        geom_id: (
            float(model.geom_rgba[geom_id, 3]),
            int(model.geom_contype[geom_id]),
            int(model.geom_conaffinity[geom_id]),
        )
        for geom_id in geom_ids
    }

    server._apply_object_visibility()

    assert geom_ids
    assert all(model.geom_rgba[geom_id, 3] == 0 for geom_id in geom_ids)
    assert all(model.geom_contype[geom_id] == 0 for geom_id in geom_ids)
    assert all(model.geom_conaffinity[geom_id] == 0 for geom_id in geom_ids)

    proxy.visibility["bread_snack"] = True
    server._apply_object_visibility()

    for geom_id in geom_ids:
        assert (
            float(model.geom_rgba[geom_id, 3]),
            int(model.geom_contype[geom_id]),
            int(model.geom_conaffinity[geom_id]),
        ) == original[geom_id]


def test_attached_snack_follows_grasp_center() -> None:
    model = mujoco.MjModel.from_xml_path(str(OFFICE_SCENE_PATH))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    server = MujocoServer.__new__(MujocoServer)
    server.mjmodel = model
    server.data_proxies = _GraspProxy("bread_snack")
    server._grasp_attachment_object = ""
    server._grasp_attachment_transform = None
    server._grasp_attachment_gravcomp = None
    server._object_geom_defaults = {}
    initial_object_z = float(data.body("bread_snack").xpos[2])

    server._apply_grasp_attachment(data)
    lift_joint = model.joint("joint_lift")
    data.qpos[lift_joint.qposadr[0]] += 0.2
    mujoco.mj_forward(model, data)
    server._apply_grasp_attachment(data)
    mujoco.mj_forward(model, data)

    assert data.body("bread_snack").xpos[2] == pytest.approx(initial_object_z + 0.2)


def test_grasp_metrics_reject_object_without_finger_contacts() -> None:
    model = mujoco.MjModel.from_xml_path(str(OFFICE_SCENE_PATH))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    proxy = _GraspMetricsProxy("cereal_box")
    server = MujocoServer.__new__(MujocoServer)
    server.mjmodel = model
    server.data_proxies = proxy

    server._update_grasp_metrics(data)

    assert proxy.metrics["center_distance_m"] > 0.1
    assert proxy.metrics["left_finger_contacts"] == 0
    assert proxy.metrics["right_finger_contacts"] == 0
    assert proxy.metrics["bilateral_contact"] is False
