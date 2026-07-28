"""Stanford TidyBot sensors — empty (no sensors defined in the current XML)."""

from typing import Any

from stretch_mujoco.datamodels.sensors import SensorMetadata, SensorType
from stretch_mujoco.robots.base import RobotSensors


class TidyBotSensors(RobotSensors):
    """Sensor enumeration for TidyBot — currently empty.

    Sensors (IMU, lidar, etc.) can be added to the robot XML and
    registered here when needed.
    """

    @property
    def sensor_name_in_mjcf(self) -> str:
        raise NotImplementedError("TidyBot has no sensors defined")

    @property
    def metadata(self) -> SensorMetadata:
        raise NotImplementedError("TidyBot has no sensors defined")

    @staticmethod
    def all() -> list["TidyBotSensors"]:
        return []

    @staticmethod
    def none() -> list["TidyBotSensors"]:
        return []

    @staticmethod
    def from_mjmodel(mjmodel: Any) -> list["TidyBotSensors"]:
        return []
