"""Shared simple types used across all robot models."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PositionVelocity:
    """Joint position and velocity pair."""

    pos: float
    vel: float

    @staticmethod
    def default() -> "PositionVelocity":
        return PositionVelocity(0.0, 0.0)


@dataclass
class BaseStatus:
    """Planar base pose and velocity."""

    x: float
    y: float
    theta: float
    x_vel: float
    theta_vel: float

    @staticmethod
    def default() -> "BaseStatus":
        return BaseStatus(0.0, 0.0, 0.0, 0.0, 0.0)
