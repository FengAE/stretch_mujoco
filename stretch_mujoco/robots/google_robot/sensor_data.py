"""Google Robot sensor snapshot."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from stretch_mujoco.datamodels.sensors import SensorMetadata
from stretch_mujoco.robots.base import RobotSensorData, RobotSensors
from stretch_mujoco.robots.google_robot.sensors import GoogleRobotSensors
from stretch_mujoco.utils import dataclass_from_dict


def _zeros3() -> np.ndarray:
    return np.zeros(3, dtype=float)


@dataclass
class StatusGoogleRobotSensorData(RobotSensorData):
    time: float = 0.0
    fps: float = 0.0
    base_imu_gyro: np.ndarray = field(default_factory=_zeros3)
    base_imu_accel: np.ndarray = field(default_factory=_zeros3)
    neck_imu_gyro: np.ndarray = field(default_factory=_zeros3)
    neck_imu_accel: np.ndarray = field(default_factory=_zeros3)
    time_of_flight: np.ndarray = field(default_factory=lambda: np.zeros(8, dtype=float))
    cliff: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=float))

    def get_data(self, sensor: RobotSensors) -> np.ndarray:
        if not isinstance(sensor, GoogleRobotSensors):
            raise TypeError(f"Expected GoogleRobotSensors, got {type(sensor)}")
        return np.asarray(getattr(self, sensor.value))

    def set_data(self, sensor: RobotSensors, value: np.ndarray) -> None:
        if not isinstance(sensor, GoogleRobotSensors):
            raise TypeError(f"Expected GoogleRobotSensors, got {type(sensor)}")
        setattr(self, sensor.value, np.asarray(value).copy())

    def get_metadata(self, sensor: RobotSensors) -> SensorMetadata:
        if not isinstance(sensor, GoogleRobotSensors):
            raise TypeError(f"Expected GoogleRobotSensors, got {type(sensor)}")
        return sensor.metadata

    def get_all(self) -> dict[RobotSensors, np.ndarray]:
        return {sensor: self.get_data(sensor) for sensor in GoogleRobotSensors.all()}

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def copy(self) -> "StatusGoogleRobotSensorData":
        return StatusGoogleRobotSensorData.from_dict(copy.deepcopy(self.to_dict()))

    @staticmethod
    def from_dict(dict_data: dict[str, Any]) -> "StatusGoogleRobotSensorData":
        return dataclass_from_dict(StatusGoogleRobotSensorData, dict_data)  # type: ignore[return-value]

    @staticmethod
    def default() -> "StatusGoogleRobotSensorData":
        return StatusGoogleRobotSensorData()
