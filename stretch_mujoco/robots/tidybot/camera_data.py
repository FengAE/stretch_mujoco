"""TidyBot camera data container."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from typing import Any

import cv2
import numpy as np

from stretch_mujoco.robots.base import RobotCameraData, RobotCameras
from stretch_mujoco.robots.tidybot.cameras import TidyBotCameras
from stretch_mujoco.utils import dataclass_from_dict


@dataclass
class StatusTidyBotCameraData(RobotCameraData):
    """Camera-data snapshot for Stanford TidyBot (2 RGB cameras)."""

    time: float = 0.0
    fps: float = 0.0

    cam_base_rgb: np.ndarray | None = None
    cam_wrist_rgb: np.ndarray | None = None
    cam_base_depth: np.ndarray | None = None
    cam_wrist_depth: np.ndarray | None = None

    # -- RobotCameraData ABC --------------------------------------------

    def get_camera_data(
        self,
        camera: RobotCameras,
        *,
        auto_rotate: bool = True,
        auto_correct_rgb: bool = True,
        use_depth_color_map: bool = False,
    ) -> np.ndarray:
        if not isinstance(camera, TidyBotCameras):
            raise TypeError(f"Expected TidyBotCameras, got {type(camera)}")
        data: np.ndarray | None = None
        if camera == TidyBotCameras.base:
            data = self.cam_base_rgb
        elif camera == TidyBotCameras.wrist:
            data = self.cam_wrist_rgb
        elif camera == TidyBotCameras.base_depth:
            data = self.cam_base_depth
        elif camera == TidyBotCameras.wrist_depth:
            data = self.cam_wrist_depth
        if data is None:
            raise ValueError(f"No data for camera {camera.name}")
        if auto_correct_rgb and camera.is_rgb:
            data = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
        return data

    def set_camera_data(self, camera: RobotCameras, data: np.ndarray) -> None:
        if not isinstance(camera, TidyBotCameras):
            raise TypeError(f"Expected TidyBotCameras, got {type(camera)}")
        if camera == TidyBotCameras.base:
            self.cam_base_rgb = data
        elif camera == TidyBotCameras.wrist:
            self.cam_wrist_rgb = data
        elif camera == TidyBotCameras.base_depth:
            self.cam_base_depth = data
        elif camera == TidyBotCameras.wrist_depth:
            self.cam_wrist_depth = data
        else:
            raise NotImplementedError(f"Camera {camera} not supported")

    def get_intrinsic_matrix(self, camera: RobotCameras) -> np.ndarray:
        if not isinstance(camera, TidyBotCameras):
            raise TypeError(f"Expected TidyBotCameras, got {type(camera)}")
        settings = camera.initial_camera_settings
        return np.asarray(settings.get_intrinsic_params_k()).reshape(3, 3)

    def get_all(self, **kwargs: Any) -> dict[RobotCameras, np.ndarray]:
        result: dict[RobotCameras, np.ndarray] = {}
        for camera in TidyBotCameras.all():
            try:
                result[camera] = self.get_camera_data(camera, **kwargs)
            except ValueError:
                pass
        return result

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def copy(self) -> "StatusTidyBotCameraData":
        return StatusTidyBotCameraData.from_dict(copy.copy(self.to_dict()))

    @staticmethod
    def from_dict(dict_data: dict[str, Any]) -> "StatusTidyBotCameraData":
        return dataclass_from_dict(StatusTidyBotCameraData, dict_data)  # type: ignore[return-type]

    @staticmethod
    def default() -> "StatusTidyBotCameraData":
        return StatusTidyBotCameraData()
