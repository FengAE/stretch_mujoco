from enum import Enum
from functools import cache

import mujoco
import mujoco._structs
import numpy as np

from stretch_mujoco.datamodels.sensors import SensorMetadata, SensorType
from stretch_mujoco.robots.base import RobotSensors


class StretchSensors(RobotSensors):
    """
    An enum of the sensors available to the simulation.
    """

    base_gyro = 0
    base_accel = 1
    base_lidar = 2

    # -- RobotSensors ABC interface ------------------------------------

    @property
    def sensor_name_in_mjcf(self) -> str:
        return self.name

    @property
    def metadata(self) -> SensorMetadata:
        if self == StretchSensors.base_gyro:
            return SensorMetadata(
                sensor_type=SensorType.IMU_GYRO,
                shape=(3,),
                units="rad/s",
                frame="base",
                description="3-axis gyroscope at base_imu site",
            )
        if self == StretchSensors.base_accel:
            return SensorMetadata(
                sensor_type=SensorType.IMU_ACCEL,
                shape=(3,),
                units="m/s²",
                frame="base",
                description="3-axis accelerometer at base_imu site",
            )
        if self == StretchSensors.base_lidar:
            return SensorMetadata(
                sensor_type=SensorType.LIDAR_2D,
                shape=(360,),
                units="m",
                frame="base",
                description="360-point 2D lidar rangefinder array, 10m max range",
            )
        return SensorMetadata(sensor_type=SensorType.CUSTOM, shape=(), description="unknown")

    @property
    def is_replicated(self) -> bool:
        return self == StretchSensors.base_lidar

    def get_replicated_names(self, resolution: int = 360) -> list[str]:
        if self == StretchSensors.base_lidar:
            num_digits = len(str(resolution))
            return [
                f"{self.name}{str(i).zfill(num_digits)}"
                for i in range(resolution)
            ]
        return [self.sensor_name_in_mjcf]

    @staticmethod
    def all() -> list["StretchSensors"]:
        """
        Returns all the available sensors
        """
        return [sensor for sensor in StretchSensors]

    @staticmethod
    def none() -> list["StretchSensors"]:
        """
        Short-hand for not using any sensor.
        """
        return []

    @staticmethod
    @cache
    def lidar_names(resolution: int = 720):
        """
        Mujoco names replicated rangefinders using the base_lidar000 -> base_lidar719 nominclature. We need to poll each one individually.
        """
        num_digits = len(str(resolution))
        return [
            f"{StretchSensors.base_lidar.name}{str(i).zfill(num_digits)}" for i in range(resolution)
        ]
    
    @staticmethod
    def from_mjmodel(mjmodel: mujoco._structs.MjModel) -> "list[StretchSensors]":
        """Get all the sensors in an mjmodel. We don't have the spec, only the compiled model. We're gonna try to find all the sensors."""
        sensors: set[StretchSensors] = set()
        remaining_sensors = [s for s in StretchSensors]
        try:
            index = 0
            while True:
                # We have no way of pulling the number of sensors via API. 
                # When we exceed the sensors in mjmodel.sensor, an IndexError will be thrown.
                name = mjmodel.sensor(index).name
                index += 1
                for sensor in remaining_sensors:
                    # base_lidar is replicated, so it's called base_lidar000 -> base_lidar359 in this list, this is why we're using `sensor.name in name` below:
                    if sensor.name in name:
                        sensors.add(sensor)
                        remaining_sensors.remove(sensor)

                if len(remaining_sensors) == 0:
                    break

        except IndexError: ...

        return list(sensors)

