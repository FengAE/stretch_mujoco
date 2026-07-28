"""GoogleRobotSimulator — concrete RobotSimulator for the Google Robot."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from mujoco._structs import MjModel

from stretch_mujoco.robots.base import (
    RobotActuators,
    RobotBodySites,
    RobotCameras,
    RobotSensors,
    RobotSimulator,
)
from stretch_mujoco.robots.google_robot.actuators import GoogleRobotActuators
from stretch_mujoco.robots.google_robot.camera_data import StatusGoogleRobotCameraData
from stretch_mujoco.robots.google_robot.cameras import GoogleRobotCameras
from stretch_mujoco.robots.google_robot.config import (
    DEFAULT_BASE_MAX_ANGULAR_VEL,
    DEFAULT_BASE_MAX_LINEAR_VEL,
)
from stretch_mujoco.robots.google_robot.joints import StatusGoogleRobotJoints
from stretch_mujoco.robots.google_robot.sensor_data import StatusGoogleRobotSensorData
from stretch_mujoco.robots.google_robot.sensors import GoogleRobotSensors
from stretch_mujoco.robots.google_robot.sites import GoogleRobotSites

_MODELS_DIR = Path(__file__).resolve().parents[2] / "models" / "assets"
_DEFAULT_SCENE = _MODELS_DIR / "google_robot" / "scene.xml"


class GoogleRobotSimulator(RobotSimulator):
    """Google Robot simulator with planar base, arm, gripper, head, and sensors.

    Uses the generic position-controlled robot backend. The model combines
    Menagerie geometry with public SimplerEnv URDF frames.
    """

    # -- robot identity --------------------------------------------------

    @property
    def Actuators(self) -> type[RobotActuators]:
        return GoogleRobotActuators

    @property
    def Cameras(self) -> type[RobotCameras]:
        return GoogleRobotCameras

    @property
    def Sensors(self) -> type[RobotSensors]:
        return GoogleRobotSensors

    @property
    def Sites(self) -> type[RobotBodySites]:
        return GoogleRobotSites

    # ------------------------------------------------------------------

    def __init__(
        self,
        scene_xml_path: str | None = None,
        model: MjModel | None = None,
        camera_hz: float = 30,
        cameras_to_use: list | None = None,
        start_translation: list | None = None,
        start_rotation_quat: list | None = None,
    ) -> None:
        self._scene_xml_path = scene_xml_path or str(_DEFAULT_SCENE)
        self._model = model
        self._camera_hz = camera_hz
        self._cameras_to_use = cameras_to_use or []
        self._start_translation = start_translation
        self._start_rotation_quat = start_rotation_quat
        self._sim: Any = None

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
        from stretch_mujoco.robots.generic_simulator import GenericRobotBackend

        self._sim = GenericRobotBackend(
            scene_xml_path=self._scene_xml_path,
            model=self._model,
            actuators=GoogleRobotActuators,
            status_type=StatusGoogleRobotJoints,
            camera_data_type=StatusGoogleRobotCameraData,
            sensor_data_type=StatusGoogleRobotSensorData,
            sensors=GoogleRobotSensors,
            camera_hz=self._camera_hz,
            cameras_to_use=list(self._cameras_to_use),
            base_actuator_names=("base_x", "base_y", "base_theta"),
            grasp_ref=GoogleRobotSites.GRIPPER.site_ref,
            start_translation=self._start_translation,
            start_rotation_quat=self._start_rotation_quat,
            max_linear_velocity=DEFAULT_BASE_MAX_LINEAR_VEL,
            max_angular_velocity=DEFAULT_BASE_MAX_ANGULAR_VEL,
            initial_joint_positions={
                "joint_head_pan": -0.00285961,
                "joint_head_tilt": 0.9351361,
            },
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
        if not isinstance(actuator, GoogleRobotActuators):
            raise TypeError(f"Expected GoogleRobotActuators, got {type(actuator)}")
        self._internal.move_to(actuator, pos)

    def move_by(self, actuator: RobotActuators, pos: float) -> None:
        if not isinstance(actuator, GoogleRobotActuators):
            raise TypeError(f"Expected GoogleRobotActuators, got {type(actuator)}")
        self._internal.move_by(actuator, pos)

    def set_base_velocity(self, v_linear: float, omega: float, v_lateral: float = 0.0) -> None:
        self._internal.set_base_velocity(v_linear, omega, v_lateral)

    def home(self) -> None:
        raise NotImplementedError("Google Robot has no 'home' keyframe defined")

    def stow(self) -> None:
        raise NotImplementedError("Google Robot has no 'stow' keyframe defined")

    def wait_until_at_setpoint(
        self,
        actuator: RobotActuators,
        timeout: float = 5.0,
        position_tolerance: float = 0.05,
    ) -> bool:
        if not isinstance(actuator, GoogleRobotActuators):
            raise TypeError(f"Expected GoogleRobotActuators, got {type(actuator)}")
        return self._internal.wait_until_at_setpoint(actuator, timeout, position_tolerance)

    def wait_while_is_moving(
        self,
        actuator: RobotActuators,
        timeout: float | None = 5.0,
        check_interval: float = 0.1,
        position_tolerance: float = 0.0005,
    ) -> bool:
        if not isinstance(actuator, GoogleRobotActuators):
            raise TypeError(f"Expected GoogleRobotActuators, got {type(actuator)}")
        return self._internal.wait_while_is_moving(
            actuator, timeout, check_interval, position_tolerance
        )

    def is_reached_set_position(
        self,
        actuator: RobotActuators,
        position_tolerance: float = 0.05,
    ) -> bool:
        if not isinstance(actuator, GoogleRobotActuators):
            raise TypeError(f"Expected GoogleRobotActuators, got {type(actuator)}")
        return self._internal.is_reached_set_position(actuator, position_tolerance)

    # -- state queries ---------------------------------------------------

    def pull_status(self) -> Any:
        return self._internal.pull_status()

    def get_base_pose(self) -> tuple[float, float, float]:
        return self._internal.get_base_pose()

    def get_ee_pose(self) -> np.ndarray:
        ref = GoogleRobotSites.GRIPPER.site_ref
        return self._internal.get_pose(ref.name, ref.mjcf_kind)

    def get_link_pose(self, link_name: str) -> np.ndarray:
        return self._internal.get_pose(link_name)

    def get_site_pose(self, site: RobotBodySites) -> np.ndarray:
        if not isinstance(site, GoogleRobotSites):
            raise TypeError(f"Expected GoogleRobotSites, got {type(site)}")
        return self._internal.get_pose(site.site_ref.name, site.site_ref.mjcf_kind)

    # -- camera ----------------------------------------------------------

    def pull_camera_data(self) -> Any:
        return self._internal.pull_camera_data()

    # -- sensors ---------------------------------------------------------

    def pull_sensor_data(self) -> Any:
        return self._internal.pull_sensor_data()

    @property
    def available_sensors(self) -> list[RobotSensors]:
        return GoogleRobotSensors.from_mjmodel(self._internal.mjmodel)

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
