"""Multi-robot abstraction layer.

Usage::

    from stretch_mujoco.robots import create_simulator, RobotType

    sim = create_simulator(RobotType.STRETCH3, scene_xml_path="scene.xml")
    sim.start(headless=True)
    sim.move_to(sim.Actuators.lift, 0.8)
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any


class RobotType(str, Enum):
    """Supported robot models."""

    STRETCH3 = "stretch3"
    GOOGLE_ROBOT = "google_robot"
    TIDYBOT = "tidybot"


def create_simulator(
    robot_type: RobotType | str,
    scene_xml_path: str | None = None,
    model: Any = None,
    camera_hz: float = 30,
    cameras_to_use: list | None = None,
    start_translation: list | None = None,
    start_rotation_quat: list | None = None,
    **kwargs: Any,
):
    """Factory that returns the right simulator for the given robot type.

    Parameters
    ----------
    robot_type:
        One of ``"stretch3"``, ``"google_robot"``, or ``"tidybot"``.
    scene_xml_path:
        Path to the MuJoCo scene XML.  If *None*, the robot-specific
        default scene is used when available.
    model:
        Optional pre-compiled ``MjModel``.
    camera_hz:
        Target camera render frequency.
    cameras_to_use:
        List of robot-specific camera enum members.  Pass an empty list
        to disable cameras.
    start_translation:
        Initial (x, y, z) base translation override.
    start_rotation_quat:
        Initial (w, x, y, z) base rotation override.
    **kwargs:
        Forwarded to the concrete simulator constructor.
    """
    if isinstance(robot_type, str):
        robot_type = RobotType(robot_type)

    cameras_to_use = cameras_to_use or []

    if robot_type == RobotType.STRETCH3:
        from stretch_mujoco.robots.stretch3.robot import Stretch3RobotSimulator  # noqa: F811

        return Stretch3RobotSimulator(
            scene_xml_path=scene_xml_path,
            model=model,
            camera_hz=camera_hz,
            cameras_to_use=cameras_to_use,
            start_translation=start_translation,
            start_rotation_quat=start_rotation_quat,
            **kwargs,
        )

    if robot_type == RobotType.GOOGLE_ROBOT:
        from stretch_mujoco.robots.google_robot.robot import GoogleRobotSimulator

        return GoogleRobotSimulator(
            scene_xml_path=scene_xml_path,
            model=model,
            camera_hz=camera_hz,
            cameras_to_use=cameras_to_use,
            start_translation=start_translation,
            start_rotation_quat=start_rotation_quat,
            **kwargs,
        )

    if robot_type == RobotType.TIDYBOT:
        from stretch_mujoco.robots.tidybot.robot import TidyBotRobotSimulator

        return TidyBotRobotSimulator(
            scene_xml_path=scene_xml_path,
            model=model,
            camera_hz=camera_hz,
            cameras_to_use=cameras_to_use,
            start_translation=start_translation,
            start_rotation_quat=start_rotation_quat,
            **kwargs,
        )

    raise ValueError(f"Unknown robot type: {robot_type}")


# Re-export key types for convenience
from stretch_mujoco.robots.base import (  # noqa: E402, F401
    ActuatorType,
    BodySiteRef,
    RobotActuators,
    RobotBaseController,
    RobotBodySites,
    RobotCameraData,
    RobotCameras,
    RobotSensorData,
    RobotSensors,
    RobotSimulator,
    RobotStatus,
)
