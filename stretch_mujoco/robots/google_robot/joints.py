"""Google Robot joint-status dataclass."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from typing import Any

from stretch_mujoco.datamodels.common import PositionVelocity
from stretch_mujoco.robots.base import RobotStatus
from stretch_mujoco.utils import dataclass_from_dict


@dataclass
class StatusGoogleRobotJoints(RobotStatus):
    """Joint-state snapshot for the Google Robot's 14 actuators."""

    time: float = 0.0
    fps: float = 0.0
    sim_to_real_time_ratio_msg: str = ""

    base_x: PositionVelocity = field(default_factory=PositionVelocity.default)
    base_y: PositionVelocity = field(default_factory=PositionVelocity.default)
    base_theta: PositionVelocity = field(default_factory=PositionVelocity.default)
    joint_torso: PositionVelocity = field(default_factory=PositionVelocity.default)
    joint_shoulder: PositionVelocity = field(default_factory=PositionVelocity.default)
    joint_bicep: PositionVelocity = field(default_factory=PositionVelocity.default)
    joint_elbow: PositionVelocity = field(default_factory=PositionVelocity.default)
    joint_forearm: PositionVelocity = field(default_factory=PositionVelocity.default)
    joint_wrist: PositionVelocity = field(default_factory=PositionVelocity.default)
    joint_gripper: PositionVelocity = field(default_factory=PositionVelocity.default)
    joint_finger_right: PositionVelocity = field(default_factory=PositionVelocity.default)
    joint_finger_left: PositionVelocity = field(default_factory=PositionVelocity.default)
    joint_head_pan: PositionVelocity = field(default_factory=PositionVelocity.default)
    joint_head_tilt: PositionVelocity = field(default_factory=PositionVelocity.default)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def copy(self) -> "StatusGoogleRobotJoints":
        return StatusGoogleRobotJoints.from_dict(copy.copy(self.to_dict()))

    @staticmethod
    def from_dict(dict_data: dict[str, Any]) -> "StatusGoogleRobotJoints":
        return dataclass_from_dict(StatusGoogleRobotJoints, dict_data)  # type: ignore[return-type]

    @staticmethod
    def default() -> "StatusGoogleRobotJoints":
        return StatusGoogleRobotJoints()
