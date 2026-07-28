"""Google Robot head camera from SimplerEnv's real-to-sim configuration."""

from stretch_mujoco.datamodels.camera import CameraSettings
from stretch_mujoco.robots.base import RobotCameras
from stretch_mujoco.robots.google_robot.config import (
    HEAD_CAMERA_FOCAL,
    HEAD_CAMERA_FOVY,
    HEAD_CAMERA_HEIGHT,
    HEAD_CAMERA_PRINCIPAL_POINT,
    HEAD_CAMERA_WIDTH,
)


class GoogleRobotCameras(RobotCameras):
    """Head-mounted RGB camera used by SimplerEnv's Google Robot."""

    overhead_camera = 0
    overhead_depth = 1

    @property
    def camera_name_in_mjcf(self) -> str:
        return "overhead_camera"

    @property
    def is_depth(self) -> bool:
        return self == GoogleRobotCameras.overhead_depth

    @property
    def is_simulated(self) -> bool:
        return self.is_depth

    @property
    def initial_camera_settings(self) -> CameraSettings:
        return CameraSettings(
            field_of_view_vertical_in_degrees=HEAD_CAMERA_FOVY,
            focal=HEAD_CAMERA_FOCAL,
            width=HEAD_CAMERA_WIDTH,
            height=HEAD_CAMERA_HEIGHT,
            principal_point=HEAD_CAMERA_PRINCIPAL_POINT,
        )

    @staticmethod
    def all() -> list["GoogleRobotCameras"]:
        return list(GoogleRobotCameras)

    @staticmethod
    def rgb() -> list["GoogleRobotCameras"]:
        return [GoogleRobotCameras.overhead_camera]

    @staticmethod
    def depth() -> list["GoogleRobotCameras"]:
        return [GoogleRobotCameras.overhead_depth]

    @staticmethod
    def none() -> list["GoogleRobotCameras"]:
        return []
