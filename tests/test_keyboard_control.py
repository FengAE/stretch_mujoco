from pathlib import Path

import mujoco
import pytest

from stretch_mujoco.keyboard_control import (
    actuator_for_joint,
    change_joint_target,
    drive_planar_base,
    reset_to_home,
    step_for_period,
)


ASSETS = Path(__file__).resolve().parents[1] / "stretch_mujoco" / "models" / "assets"


@pytest.mark.parametrize(
    ("robot", "base_joints"),
    [
        ("google_robot", ("base_x", "base_y", "base_theta")),
        ("stanford_tidybot", ("joint_x", "joint_y", "joint_th")),
    ],
)
def test_wasd_moves_mobile_robot_in_stretch_scene(robot, base_joints) -> None:
    model = mujoco.MjModel.from_xml_path(str(ASSETS / robot / "scene.xml"))
    data = mujoco.MjData(model)
    reset_to_home(model, data)

    for body_name in ("table", "object1", "object2"):
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name) >= 0

    for _ in range(5):
        drive_planar_base(
            model,
            data,
            "w",
            x_joint=base_joints[0],
            y_joint=base_joints[1],
            heading_joint=base_joints[2],
        )
    for _ in range(50):
        step_for_period(model, data, 0.02)

    assert data.ctrl[actuator_for_joint(model, base_joints[0])] == pytest.approx(0.2)
    assert float(data.joint(base_joints[0]).qpos[0]) > 0.15


def test_unlimited_tidybot_actuator_is_not_clamped_to_zero() -> None:
    model = mujoco.MjModel.from_xml_path(str(ASSETS / "stanford_tidybot" / "scene.xml"))
    data = mujoco.MjData(model)
    reset_to_home(model, data)

    target = change_joint_target(model, data, "joint_1", 0.15)

    assert not model.actuator_ctrllimited[actuator_for_joint(model, "joint_1")]
    assert target == pytest.approx(0.15)
