"""Google Robot hardware configuration."""

# -- base ----------------------------------------------------------------
# Omnidirectional base — position-controlled planar joints.
DEFAULT_BASE_MAX_LINEAR_VEL: float = 2.0  # m/s
DEFAULT_BASE_MAX_ANGULAR_VEL: float = 3.0  # rad/s

# -- arm -----------------------------------------------------------------
# 7-DoF articulated arm + gripper rotation + 2 fingers.
# Joint limits match the ctrlrange values in robot.xml.
JOINT_TORSO_RANGE: tuple[float, float] = (-4.49, 1.35)
JOINT_SHOULDER_RANGE: tuple[float, float] = (-2.66, 3.18)
JOINT_BICEP_RANGE: tuple[float, float] = (-2.13, 3.71)
JOINT_ELBOW_RANGE: tuple[float, float] = (-2.05, 3.79)
JOINT_FOREARM_RANGE: tuple[float, float] = (-2.92, 2.92)
JOINT_WRIST_RANGE: tuple[float, float] = (-1.79, 1.79)
JOINT_GRIPPER_ROT_RANGE: tuple[float, float] = (-4.49, 1.35)
FINGER_RANGE: tuple[float, float] = (0.01, 1.3)
JOINT_HEAD_PAN_RANGE: tuple[float, float] = (-3.79, 2.22)
JOINT_HEAD_TILT_RANGE: tuple[float, float] = (-1.17, 1.17)

# SimplerEnv GoogleRobotManualTunedIntrinsicConfig.
HEAD_CAMERA_WIDTH: int = 640
HEAD_CAMERA_HEIGHT: int = 512
HEAD_CAMERA_FOCAL: tuple[float, float] = (425.0, 413.1)
HEAD_CAMERA_PRINCIPAL_POINT: tuple[float, float] = (305.0, 233.0)
HEAD_CAMERA_FOVY: float = 63.57337709836487

# -- robot XML paths ----------------------------------------------------
ROBOT_XML_NAME: str = "robot.xml"
SCENE_XML_NAME: str = "scene.xml"
