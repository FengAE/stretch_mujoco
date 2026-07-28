"""Shared sensor metadata types used across all robot sensor enums."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class SensorType(str, Enum):
    """Standard taxonomy for robot sensors.

    Consumers use this to interpret raw ``np.ndarray`` data without
    needing robot-specific knowledge.
    """

    IMU_GYRO = "imu_gyro"
    """3-axis angular velocity [rad/s]."""

    IMU_ACCEL = "imu_accel"
    """3-axis linear acceleration [m/s²]."""

    IMU_COMBINED = "imu_combined"
    """6-axis IMU (gyro + accel packed together)."""

    LIDAR_2D = "lidar_2d"
    """Planar rangefinder array, shape (N,) in metres."""

    LIDAR_3D = "lidar_3d"
    """3-D point-cloud, shape (N, 3) in metres."""

    JOINT_TORQUE = "joint_torque"
    """Per-joint torque [N·m]."""

    FT_WRIST = "ft_wrist"
    """6-axis force-torque at the wrist frame."""

    FT_BASE = "ft_base"
    """6-axis force-torque at the base frame."""

    CONTACT = "contact"
    """Binary contact / bumper."""

    RANGEFINDER = "rangefinder"
    """Single-beam range sensor [m]."""

    ODOMETRY = "odometry"
    """Computed odometry (may be derived from joints + IMU)."""

    CUSTOM = "custom"
    """Robot-specific sensor not covered by the standard taxonomy."""


@dataclass(frozen=True)
class SensorMetadata:
    """Describes a sensor's output so consumers can interpret raw data.

    Every ``RobotSensors`` enum member exposes one of these via its
    ``metadata`` property.
    """

    sensor_type: SensorType
    shape: tuple[int, ...]
    dtype: type = np.float64
    units: str = ""
    frame: str = "sensor"
    description: str = ""


@dataclass(frozen=True)
class LidarConfig:
    """Configuration for a 2-D lidar modelled as replicated rangefinders."""

    resolution: int
    """Number of rangefinder beams (e.g. 360 or 720)."""

    fov_degrees: float = 360.0
    """Angular coverage in degrees."""

    min_range: float = 0.0
    """Minimum measurable distance [m]."""

    max_range: float = 10.0
    """Maximum measurable distance [m]."""
