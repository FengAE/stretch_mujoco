"""Stanford TidyBot cameras — base camera + wrist camera."""

from stretch_mujoco.datamodels.camera import CameraSettings
from stretch_mujoco.robots.base import RobotCameras
from stretch_mujoco.robots.tidybot.config import (
    EGOCENTRIC_CAMERA_DISTORTION,
    EGOCENTRIC_CAMERA_FOCAL,
    EGOCENTRIC_CAMERA_FOVY,
    EGOCENTRIC_CAMERA_PRINCIPAL_POINT,
    EGOCENTRIC_CAMERA_SIZE,
)


class TidyBotCameras(RobotCameras):
    """Two cameras on the Stanford TidyBot.

    * ``base``  — real C930e calibration scaled to 640×360
    * ``wrist`` — simulation-only end-effector view at 640×480
    """

    base = 0
    wrist = 1
    base_depth = 2
    wrist_depth = 3

    @property
    def camera_name_in_mjcf(self) -> str:
        return "base" if self in {TidyBotCameras.base, TidyBotCameras.base_depth} else "wrist"

    @property
    def is_depth(self) -> bool:
        return self in {TidyBotCameras.base_depth, TidyBotCameras.wrist_depth}

    @property
    def is_simulated(self) -> bool:
        return self != TidyBotCameras.base

    @property
    def initial_camera_settings(self) -> CameraSettings:
        if self in {TidyBotCameras.base, TidyBotCameras.base_depth}:
            return CameraSettings(
                field_of_view_vertical_in_degrees=EGOCENTRIC_CAMERA_FOVY,
                focal=EGOCENTRIC_CAMERA_FOCAL,
                width=EGOCENTRIC_CAMERA_SIZE[0],
                height=EGOCENTRIC_CAMERA_SIZE[1],
                principal_point=EGOCENTRIC_CAMERA_PRINCIPAL_POINT,
                distortion_params=EGOCENTRIC_CAMERA_DISTORTION,
            )
        if self in {TidyBotCameras.wrist, TidyBotCameras.wrist_depth}:
            return CameraSettings(
                field_of_view_vertical_in_degrees=41.83792730009236,
                focal=(622.3820289447968, 622.3820289447968),
                width=640,
                height=480,
            )
        raise NotImplementedError(f"Camera {self} settings not defined")

    @staticmethod
    def all() -> list["TidyBotCameras"]:
        return [c for c in TidyBotCameras]

    @staticmethod
    def rgb() -> list["TidyBotCameras"]:
        return [TidyBotCameras.base, TidyBotCameras.wrist]

    @staticmethod
    def depth() -> list["TidyBotCameras"]:
        return [TidyBotCameras.base_depth, TidyBotCameras.wrist_depth]

    @staticmethod
    def none() -> list["TidyBotCameras"]:
        return []
