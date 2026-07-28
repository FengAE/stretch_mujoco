"""TidyBot joint-status dataclass."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from typing import Any

from stretch_mujoco.datamodels.common import PositionVelocity
from stretch_mujoco.robots.base import RobotStatus
from stretch_mujoco.utils import dataclass_from_dict


@dataclass
class StatusTidyBotJoints(RobotStatus):
    """Joint-state snapshot for the 11-DoF Stanford TidyBot."""

    time: float = 0.0
    fps: float = 0.0
    sim_to_real_time_ratio_msg: str = ""

    joint_x: PositionVelocity = field(default_factory=PositionVelocity.default)
    joint_y: PositionVelocity = field(default_factory=PositionVelocity.default)
    joint_th: PositionVelocity = field(default_factory=PositionVelocity.default)
    joint_1: PositionVelocity = field(default_factory=PositionVelocity.default)
    joint_2: PositionVelocity = field(default_factory=PositionVelocity.default)
    joint_3: PositionVelocity = field(default_factory=PositionVelocity.default)
    joint_4: PositionVelocity = field(default_factory=PositionVelocity.default)
    joint_5: PositionVelocity = field(default_factory=PositionVelocity.default)
    joint_6: PositionVelocity = field(default_factory=PositionVelocity.default)
    joint_7: PositionVelocity = field(default_factory=PositionVelocity.default)
    # General (tendon) actuator — stored as a position-only value
    fingers_actuator: PositionVelocity = field(default_factory=PositionVelocity.default)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def copy(self) -> "StatusTidyBotJoints":
        return StatusTidyBotJoints.from_dict(copy.copy(self.to_dict()))

    @staticmethod
    def from_dict(dict_data: dict[str, Any]) -> "StatusTidyBotJoints":
        return dataclass_from_dict(StatusTidyBotJoints, dict_data)  # type: ignore[return-type]

    @staticmethod
    def default() -> "StatusTidyBotJoints":
        return StatusTidyBotJoints()
