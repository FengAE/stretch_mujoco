"""Google Robot head-camera data."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import cv2

from stretch_mujoco.robots.base import RobotCameraData, RobotCameras
from stretch_mujoco.robots.google_robot.cameras import GoogleRobotCameras
from stretch_mujoco.utils import dataclass_from_dict


@dataclass
class StatusGoogleRobotCameraData(RobotCameraData):
    """Camera-data snapshot for the SimplerEnv-compatible head camera."""

    time: float = 0.0
    fps: float = 0.0
    overhead_camera_rgb: np.ndarray | None = None
    overhead_camera_depth: np.ndarray | None = None

    def get_camera_data(
        self,
        camera: RobotCameras,
        *,
        auto_rotate: bool = True,
        auto_correct_rgb: bool = True,
        use_depth_color_map: bool = False,
    ) -> np.ndarray:
        if not isinstance(camera, GoogleRobotCameras):
            raise TypeError(f"Expected GoogleRobotCameras, got {type(camera)}")
        data = (
            self.overhead_camera_depth
            if camera == GoogleRobotCameras.overhead_depth
            else self.overhead_camera_rgb
        )
        if data is None:
            raise ValueError(f"No data for camera {camera.name}")
        if camera.is_depth:
            return data
        return cv2.cvtColor(data, cv2.COLOR_RGB2BGR) if auto_correct_rgb else data

    def set_camera_data(self, camera: RobotCameras, data: np.ndarray) -> None:
        if not isinstance(camera, GoogleRobotCameras):
            raise TypeError(f"Expected GoogleRobotCameras, got {type(camera)}")
        if camera == GoogleRobotCameras.overhead_camera:
            self.overhead_camera_rgb = data
        elif camera == GoogleRobotCameras.overhead_depth:
            self.overhead_camera_depth = data
        else:
            raise ValueError(f"Unsupported Google Robot camera {camera}")

    def get_intrinsic_matrix(self, camera: RobotCameras) -> np.ndarray:
        if not isinstance(camera, GoogleRobotCameras):
            raise TypeError(f"Expected GoogleRobotCameras, got {type(camera)}")
        return np.asarray(camera.initial_camera_settings.get_intrinsic_params_k()).reshape(3, 3)

    def get_all(self, **kwargs: Any) -> dict[RobotCameras, np.ndarray]:
        result: dict[RobotCameras, np.ndarray] = {}
        for camera in GoogleRobotCameras.all():
            try:
                result[camera] = self.get_camera_data(camera, **kwargs)
            except ValueError:
                pass
        return result

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def copy(self) -> "StatusGoogleRobotCameraData":
        return StatusGoogleRobotCameraData.from_dict(copy.copy(self.to_dict()))

    @staticmethod
    def from_dict(dict_data: dict[str, Any]) -> "StatusGoogleRobotCameraData":
        return dataclass_from_dict(StatusGoogleRobotCameraData, dict_data)  # type: ignore[return-type]

    @staticmethod
    def default() -> "StatusGoogleRobotCameraData":
        return StatusGoogleRobotCameraData()
