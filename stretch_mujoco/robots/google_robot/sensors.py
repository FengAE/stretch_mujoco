"""Google Robot sensors represented by the SimplerEnv URDF mounts."""

from __future__ import annotations

from typing import Any

import mujoco

from stretch_mujoco.datamodels.sensors import SensorMetadata, SensorType
from stretch_mujoco.robots.base import RobotSensors


class GoogleRobotSensors(RobotSensors):
    BASE_IMU_GYRO = "base_imu_gyro"
    BASE_IMU_ACCEL = "base_imu_accel"
    NECK_IMU_GYRO = "neck_imu_gyro"
    NECK_IMU_ACCEL = "neck_imu_accel"
    TIME_OF_FLIGHT = "time_of_flight"
    CLIFF = "cliff"

    @property
    def sensor_name_in_mjcf(self) -> str:
        return self.value

    @property
    def metadata(self) -> SensorMetadata:
        metadata = {
            GoogleRobotSensors.BASE_IMU_GYRO: SensorMetadata(
                SensorType.IMU_GYRO, (3,), units="rad/s", frame="base_imu_site"
            ),
            GoogleRobotSensors.BASE_IMU_ACCEL: SensorMetadata(
                SensorType.IMU_ACCEL, (3,), units="m/s^2", frame="base_imu_site"
            ),
            GoogleRobotSensors.NECK_IMU_GYRO: SensorMetadata(
                SensorType.IMU_GYRO, (3,), units="rad/s", frame="neck_imu_site"
            ),
            GoogleRobotSensors.NECK_IMU_ACCEL: SensorMetadata(
                SensorType.IMU_ACCEL, (3,), units="m/s^2", frame="neck_imu_site"
            ),
            GoogleRobotSensors.TIME_OF_FLIGHT: SensorMetadata(
                SensorType.RANGEFINDER,
                (8,),
                units="m",
                frame="base_link",
                description="Eight perimeter time-of-flight rays",
            ),
            GoogleRobotSensors.CLIFF: SensorMetadata(
                SensorType.RANGEFINDER,
                (2,),
                units="m",
                frame="base_link",
                description="Two rear downward-facing cliff rays",
            ),
        }
        return metadata[self]

    @property
    def is_replicated(self) -> bool:
        return self in {
            GoogleRobotSensors.TIME_OF_FLIGHT,
            GoogleRobotSensors.CLIFF,
        }

    def get_replicated_names(self, resolution: int = 0) -> list[str]:
        _ = resolution
        if self == GoogleRobotSensors.TIME_OF_FLIGHT:
            return [
                "tof_right_1",
                "tof_right_2",
                "tof_right_back_1",
                "tof_right_back_2",
                "tof_left_1",
                "tof_left_2",
                "tof_left_back_1",
                "tof_left_back_2",
            ]
        if self == GoogleRobotSensors.CLIFF:
            return ["cliff_rear_right", "cliff_rear_left"]
        return super().get_replicated_names(resolution)

    @staticmethod
    def all() -> list["GoogleRobotSensors"]:
        return list(GoogleRobotSensors)

    @staticmethod
    def none() -> list["GoogleRobotSensors"]:
        return []

    @staticmethod
    def from_mjmodel(mjmodel: Any) -> list["GoogleRobotSensors"]:
        available: list[GoogleRobotSensors] = []
        for sensor in GoogleRobotSensors:
            names = (
                sensor.get_replicated_names()
                if sensor.is_replicated
                else [sensor.sensor_name_in_mjcf]
            )
            if all(
                mujoco.mj_name2id(mjmodel, mujoco.mjtObj.mjOBJ_SENSOR, name) >= 0 for name in names
            ):
                available.append(sensor)
        return available
