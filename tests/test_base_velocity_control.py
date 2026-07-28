from types import SimpleNamespace

import mujoco
import numpy as np

from stretch_mujoco import config, utils
from stretch_mujoco.mujoco_server import BaseController


def test_base_velocity_accounts_for_actuator_gear():
    model = mujoco.MjModel.from_xml_path("stretch_mujoco/models/scene.xml")
    data = mujoco.MjData(model)
    ids = [model.actuator(name).id for name in ("left_wheel_vel", "right_wheel_vel")]
    model.actuator_ctrlrange[ids] = (-40.0, 40.0)
    controller = BaseController(SimpleNamespace(mjmodel=model, mjdata=data))

    controller._set_base_velocity(0.3, 0.4)

    wheel_speeds = np.array(utils.diff_drive_inv_kinematics(0.3, 0.4))
    assert np.allclose(data.ctrl[ids], wheel_speeds * model.actuator_gear[ids, 0])
    assert config.base_motion["default_x_vel"] == 0.3
