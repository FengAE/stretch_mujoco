"""Stanford TidyBot actuators — 11 actuators including a tendon-driven gripper."""

from stretch_mujoco.robots.base import ActuatorType, RobotActuators, RobotStatus


class TidyBotActuators(RobotActuators):
    """11 actuators for the Stanford TidyBot.

    Base:      ``joint_x`` (slide), ``joint_y`` (slide), ``joint_th`` (hinge)
    Arm:       ``joint_1`` … ``joint_7`` — 7-DoF Kinova Gen3
    Gripper:   ``fingers_actuator`` — **general** actuator (0-255 ctrl)
               driving a tendon-coupled Robotiq 2F-85.

    The arm joints use two actuator classes defined in the MJCF:
    * ``large_actuator`` (kp=2000, kv=100, forcerange=±105) — joints 1-4
    * ``small_actuator`` (kp=500,  kv=50,  forcerange=±52)  — joints 5-7
    """

    joint_x = 0
    joint_y = 1
    joint_th = 2
    joint_1 = 3
    joint_2 = 4
    joint_3 = 5
    joint_4 = 6
    joint_5 = 7
    joint_6 = 8
    joint_7 = 9
    fingers_actuator = 10

    # -- RobotActuators ABC ---------------------------------------------

    def get_joint_names_in_mjcf(self) -> list[str]:
        # fingers_actuator drives a tendon, not a joint
        if self == TidyBotActuators.fingers_actuator:
            return []  # tendon — no direct joint
        return [self.name]

    def actuator_type(self) -> ActuatorType:
        if self == TidyBotActuators.fingers_actuator:
            return ActuatorType.GENERAL
        return ActuatorType.POSITION

    @property
    def has_position_control(self) -> bool:
        return self != TidyBotActuators.fingers_actuator

    @property
    def has_velocity_control(self) -> bool:
        return False

    def is_base_actuator(self) -> bool:
        return self in {
            TidyBotActuators.joint_x,
            TidyBotActuators.joint_y,
            TidyBotActuators.joint_th,
        }

    def get_position(self, status: RobotStatus) -> float:
        _map = {
            TidyBotActuators.joint_x: status.joint_x,
            TidyBotActuators.joint_y: status.joint_y,
            TidyBotActuators.joint_th: status.joint_th,
            TidyBotActuators.joint_1: status.joint_1,
            TidyBotActuators.joint_2: status.joint_2,
            TidyBotActuators.joint_3: status.joint_3,
            TidyBotActuators.joint_4: status.joint_4,
            TidyBotActuators.joint_5: status.joint_5,
            TidyBotActuators.joint_6: status.joint_6,
            TidyBotActuators.joint_7: status.joint_7,
            TidyBotActuators.fingers_actuator: status.fingers_actuator,
        }
        val = _map.get(self)
        if val is None:
            raise NotImplementedError(f"get_position not implemented for {self}")
        return float(val) if isinstance(val, (int, float)) else float(val.pos)

    def get_velocity(self, status: RobotStatus) -> float:
        value = getattr(status, self.name)
        return float(value.vel) if hasattr(value, "vel") else 0.0

    def get_position_relative(self, status: RobotStatus) -> tuple[float, float, float]:
        if self.is_base_actuator():
            return (
                (
                    float(status.joint_x.pos)
                    if hasattr(status.joint_x, "pos")
                    else float(status.joint_x)
                ),
                (
                    float(status.joint_y.pos)
                    if hasattr(status.joint_y, "pos")
                    else float(status.joint_y)
                ),
                (
                    float(status.joint_th.pos)
                    if hasattr(status.joint_th, "pos")
                    else float(status.joint_th)
                ),
            )
        raise ValueError(f"get_position_relative not valid for {self}")

    def get_velocity_relative(self, status: RobotStatus) -> tuple[float, float, float]:
        return (0.0, 0.0, 0.0)

    @staticmethod
    def get_actuator_by_joint_names_in_mjcf(joint_name: str) -> "TidyBotActuators":
        for actuator in TidyBotActuators:
            if actuator.name == joint_name:
                return actuator
        raise KeyError(f"No TidyBot actuator for MuJoCo joint '{joint_name}'")

    @staticmethod
    def all() -> list["TidyBotActuators"]:
        return [a for a in TidyBotActuators]
