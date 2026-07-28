"""Google Robot actuators — planar base, arm, gripper, and actuated head."""

from stretch_mujoco.robots.base import ActuatorType, RobotActuators, RobotStatus


class GoogleRobotActuators(RobotActuators):
    """14-actuator position-controlled Google Robot.

    Base:   ``base_x`` (slide), ``base_y`` (slide), ``base_theta`` (hinge)
    Arm:    ``joint_torso``, ``joint_shoulder``, ``joint_bicep``,
            ``joint_elbow``, ``joint_forearm``, ``joint_wrist``,
            ``joint_gripper``
    Gripper:``joint_finger_right``, ``joint_finger_left``
    Head:   ``joint_head_pan``, ``joint_head_tilt``
    """

    base_x = 0
    base_y = 1
    base_theta = 2
    joint_torso = 3
    joint_shoulder = 4
    joint_bicep = 5
    joint_elbow = 6
    joint_forearm = 7
    joint_wrist = 8
    joint_gripper = 9
    joint_finger_right = 10
    joint_finger_left = 11
    joint_head_pan = 12
    joint_head_tilt = 13

    # -- RobotActuators ABC ---------------------------------------------

    def get_joint_names_in_mjcf(self) -> list[str]:
        return [self.name]

    def actuator_type(self) -> ActuatorType:
        return ActuatorType.POSITION

    @property
    def has_position_control(self) -> bool:
        return True

    @property
    def has_velocity_control(self) -> bool:
        return False

    def is_base_actuator(self) -> bool:
        return self in {
            GoogleRobotActuators.base_x,
            GoogleRobotActuators.base_y,
            GoogleRobotActuators.base_theta,
        }

    def get_position(self, status: RobotStatus) -> float:
        _map = {
            GoogleRobotActuators.base_x: status.base_x,
            GoogleRobotActuators.base_y: status.base_y,
            GoogleRobotActuators.base_theta: status.base_theta,
            GoogleRobotActuators.joint_torso: status.joint_torso,
            GoogleRobotActuators.joint_shoulder: status.joint_shoulder,
            GoogleRobotActuators.joint_bicep: status.joint_bicep,
            GoogleRobotActuators.joint_elbow: status.joint_elbow,
            GoogleRobotActuators.joint_forearm: status.joint_forearm,
            GoogleRobotActuators.joint_wrist: status.joint_wrist,
            GoogleRobotActuators.joint_gripper: status.joint_gripper,
            GoogleRobotActuators.joint_finger_right: status.joint_finger_right,
            GoogleRobotActuators.joint_finger_left: status.joint_finger_left,
            GoogleRobotActuators.joint_head_pan: status.joint_head_pan,
            GoogleRobotActuators.joint_head_tilt: status.joint_head_tilt,
        }
        val = _map.get(self)
        if val is None:
            raise NotImplementedError(f"get_position not implemented for {self}")
        return float(val) if isinstance(val, (int, float)) else float(val.pos)

    def get_velocity(self, status: RobotStatus) -> float:
        value = getattr(status, self.name)
        return float(value.vel) if hasattr(value, "vel") else 0.0

    def get_position_relative(self, status: RobotStatus) -> tuple[float, float, float]:
        if self in {
            GoogleRobotActuators.base_x,
            GoogleRobotActuators.base_y,
            GoogleRobotActuators.base_theta,
        }:
            return (
                float(status.base_x.pos) if hasattr(status.base_x, "pos") else float(status.base_x),
                float(status.base_y.pos) if hasattr(status.base_y, "pos") else float(status.base_y),
                (
                    float(status.base_theta.pos)
                    if hasattr(status.base_theta, "pos")
                    else float(status.base_theta)
                ),
            )
        raise ValueError(f"get_position_relative not valid for {self} — not a base actuator")

    def get_velocity_relative(self, status: RobotStatus) -> tuple[float, float, float]:
        return (0.0, 0.0, 0.0)

    @staticmethod
    def get_actuator_by_joint_names_in_mjcf(joint_name: str) -> "GoogleRobotActuators":
        for actuator in GoogleRobotActuators:
            if actuator.name == joint_name:
                return actuator
        raise KeyError(f"No GoogleRobot actuator for MuJoCo joint '{joint_name}'")

    @staticmethod
    def all() -> list["GoogleRobotActuators"]:
        return [a for a in GoogleRobotActuators]
