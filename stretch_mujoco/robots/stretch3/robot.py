"""Stretch3RobotSimulator — concrete RobotSimulator for the Stretch 3."""

from __future__ import annotations

from typing import Any

import numpy as np
from mujoco._structs import MjModel

from stretch_mujoco.datamodels.status_command import CommandKeyframe, StatusCommand
from stretch_mujoco.enums.actuators import Actuators
from stretch_mujoco.enums.stretch_cameras import StretchCameras
from stretch_mujoco.enums.stretch_sensors import StretchSensors
from stretch_mujoco.robots.base import (
    ActuatorType,
    RobotActuators,
    RobotBodySites,
    RobotCameras,
    RobotSensors,
    RobotSimulator,
)
from stretch_mujoco.robots.stretch3.sites import Stretch3Sites


class Stretch3RobotSimulator(RobotSimulator):
    """Stretch 3 robot simulator conforming to the ``RobotSimulator`` ABC.

    Internally delegates to the proven ``StretchMujocoSimulator``
    implementation while exposing the new abstract interface.
    """

    # -- robot identity --------------------------------------------------

    @property
    def Actuators(self) -> type[RobotActuators]:
        return Actuators

    @property
    def Cameras(self) -> type[RobotCameras]:
        return StretchCameras

    @property
    def Sensors(self) -> type[RobotSensors]:
        return StretchSensors

    @property
    def Sites(self) -> type[RobotBodySites]:
        return Stretch3Sites

    # ------------------------------------------------------------------
    # Internal _sim accessor — lazily creates the underlying simulator
    # ------------------------------------------------------------------

    def __init__(
        self,
        scene_xml_path: str | None = None,
        model: MjModel | None = None,
        camera_hz: float = 30,
        cameras_to_use: list[StretchCameras] | None = None,
        start_translation: list | None = None,
        start_rotation_quat: list | None = None,
    ) -> None:
        self._scene_xml_path = scene_xml_path
        self._model = model
        self._camera_hz = camera_hz
        self._cameras_to_use = cameras_to_use or []
        self._start_translation = start_translation
        self._start_rotation_quat = start_rotation_quat
        self._sim: Any = None  # StretchMujocoSimulator, lazy-created in start()

    @property
    def _internal(self) -> Any:
        if self._sim is None:
            raise RuntimeError("Simulator not started — call start() first")
        return self._sim

    # -- lifecycle -------------------------------------------------------

    def start(
        self,
        show_viewer_ui: bool = False,
        headless: bool = False,
        use_passive_viewer: bool = True,
    ) -> None:
        from stretch_mujoco.stretch_mujoco_simulator import StretchMujocoSimulator

        self._sim = StretchMujocoSimulator(
            scene_xml_path=self._scene_xml_path,
            model=self._model,
            camera_hz=self._camera_hz,
            cameras_to_use=list(self._cameras_to_use),
            start_translation=self._start_translation,
            start_rotation_quat=self._start_rotation_quat,
        )
        self._sim.start(
            show_viewer_ui=show_viewer_ui,
            headless=headless,
            use_passive_viewer=use_passive_viewer,
        )

    def stop(self) -> None:
        if self._sim is not None:
            self._sim.stop()

    def is_running(self) -> bool:
        return self._sim is not None and self._sim.is_running()

    # -- movement --------------------------------------------------------

    def move_to(self, actuator: RobotActuators, pos: float) -> None:
        if not isinstance(actuator, Actuators):
            raise TypeError(f"Expected Stretch Actuators, got {type(actuator)}")
        self._internal.move_to(actuator, pos)

    def move_by(self, actuator: RobotActuators, pos: float) -> None:
        if not isinstance(actuator, Actuators):
            raise TypeError(f"Expected Stretch Actuators, got {type(actuator)}")
        self._internal.move_by(actuator, pos)

    def set_base_velocity(
        self, v_linear: float, omega: float, v_lateral: float = 0.0
    ) -> None:
        if not np.isclose(v_lateral, 0.0):
            raise ValueError("Stretch 3 is differential-drive and cannot move laterally")
        self._internal.set_base_velocity(v_linear, omega)

    def home(self) -> None:
        self._internal.home()

    def stow(self) -> None:
        self._internal.stow()

    def wait_until_at_setpoint(
        self,
        actuator: RobotActuators,
        timeout: float = 5.0,
        position_tolerance: float = 0.05,
    ) -> bool:
        if not isinstance(actuator, Actuators):
            raise TypeError(f"Expected Stretch Actuators, got {type(actuator)}")
        return self._internal.wait_until_at_setpoint(
            actuator, timeout=timeout, position_tolerance=position_tolerance
        )

    def wait_while_is_moving(
        self,
        actuator: RobotActuators,
        timeout: float | None = 5.0,
        check_interval: float = 0.1,
        position_tolerance: float = 0.0005,
    ) -> bool:
        if not isinstance(actuator, Actuators):
            raise TypeError(f"Expected Stretch Actuators, got {type(actuator)}")
        return self._internal.wait_while_is_moving(
            actuator,
            timeout=timeout,
            check_interval=check_interval,
            position_tolerance=position_tolerance,
        )

    def is_reached_set_position(
        self,
        actuator: RobotActuators,
        position_tolerance: float = 0.05,
    ) -> bool:
        if not isinstance(actuator, Actuators):
            raise TypeError(f"Expected Stretch Actuators, got {type(actuator)}")
        return self._internal.is_reached_set_position(
            actuator, position_tolerance=position_tolerance
        )

    # -- state queries ---------------------------------------------------

    def pull_status(self) -> Any:
        return self._internal.pull_status()

    def get_base_pose(self) -> tuple[float, float, float]:
        return self._internal.get_base_pose()

    def get_ee_pose(self) -> np.ndarray:
        return self._internal.get_ee_pose()

    def get_link_pose(self, link_name: str) -> np.ndarray:
        return self._internal.get_link_pose(link_name)

    def get_site_pose(self, site: RobotBodySites) -> np.ndarray:
        if not isinstance(site, Stretch3Sites):
            raise TypeError(f"Expected Stretch3Sites, got {type(site)}")
        return self._internal.get_link_pose(site.site_ref.name)

    # -- camera ----------------------------------------------------------

    def pull_camera_data(self) -> Any:
        return self._internal.pull_camera_data()

    def get_camera_intrinsics(self, camera: RobotCameras) -> np.ndarray:
        if not isinstance(camera, StretchCameras):
            raise TypeError(f"Expected StretchCameras, got {type(camera)}")
        return self._internal.pull_camera_data().get_camera_data.camera_intrinsic_matrix

    # -- sensors ---------------------------------------------------------

    def pull_sensor_data(self) -> Any:
        return self._internal.pull_sensor_data()

    @property
    def available_sensors(self) -> list[RobotSensors]:
        return StretchSensors.all()

    # -- grasping --------------------------------------------------------

    def attach_object_to_gripper(self, object_id: str) -> None:
        self._internal.attach_object_to_gripper(object_id)

    def release_grasped_object(self) -> None:
        self._internal.release_grasped_object()

    # -- utility ---------------------------------------------------------

    def set_object_visibility(self, object_id: str, visible: bool) -> None:
        self._internal.set_object_visibility(object_id, visible)

    def pull_joint_limits(self) -> dict[RobotActuators, tuple[float, float]]:
        raw = self._internal.pull_joint_limits()
        return {k: v for k, v in raw.items()}

    def add_world_frame(
        self,
        position: tuple[float, float, float],
        rotation: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None:
        self._internal.add_world_frame(position, rotation)
