"""TidyBot sensor data — empty (no sensors)."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from stretch_mujoco.datamodels.sensors import SensorMetadata
from stretch_mujoco.robots.base import RobotSensorData, RobotSensors
from stretch_mujoco.utils import dataclass_from_dict


@dataclass
class StatusTidyBotSensorData(RobotSensorData):
    """Sensor-data snapshot — always empty for TidyBot."""

    time: float = 0.0
    fps: float = 0.0

    def get_data(self, sensor: RobotSensors) -> np.ndarray:
        raise ValueError("TidyBot has no sensors")

    def set_data(self, sensor: RobotSensors, value: np.ndarray) -> None:
        raise ValueError("TidyBot has no sensors")

    def get_metadata(self, sensor: RobotSensors) -> SensorMetadata:
        raise ValueError("TidyBot has no sensors")

    def get_all(self) -> dict[RobotSensors, np.ndarray]:
        return {}

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def copy(self) -> "StatusTidyBotSensorData":
        return StatusTidyBotSensorData.from_dict(copy.copy(self.to_dict()))

    @staticmethod
    def from_dict(dict_data: dict[str, Any]) -> "StatusTidyBotSensorData":
        return dataclass_from_dict(StatusTidyBotSensorData, dict_data)  # type: ignore[return-type]

    @staticmethod
    def default() -> "StatusTidyBotSensorData":
        return StatusTidyBotSensorData()
