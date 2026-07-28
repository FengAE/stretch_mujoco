"""Stanford TidyBot hardware configuration."""

# -- base ----------------------------------------------------------------
DEFAULT_BASE_MAX_LINEAR_VEL: float = 2.0  # m/s
DEFAULT_BASE_MAX_ANGULAR_VEL: float = 3.0  # rad/s

# Official TidyBot robot #2 C930e calibration (serial 44251E9E), scaled from
# 1280x720 to the 640x360 stream produced by robot/camera.py.
EGOCENTRIC_CAMERA_SERIAL: str = "44251E9E"
EGOCENTRIC_CAMERA_SIZE: tuple[int, int] = (640, 360)
EGOCENTRIC_CAMERA_FOCAL: tuple[float, float] = (
    377.89521770116426,
    378.16364930400925,
)
EGOCENTRIC_CAMERA_PRINCIPAL_POINT: tuple[float, float] = (
    319.9291029848687,
    180.4933473081451,
)
EGOCENTRIC_CAMERA_DISTORTION: tuple[float, ...] = (
    0.09860833831124165,
    -0.24087309592232233,
    -0.00035561249192511416,
    -0.00031414951926516975,
    0.1176028326156184,
)
EGOCENTRIC_CAMERA_FOVY: float = 50.9074395456145

# -- arm (Kinova Gen3) ---------------------------------------------------
# joint_1: large_actuator, no explicit ctrlrange
JOINT_2_RANGE: tuple[float, float] = (-2.2497294058206907, 2.2497294058206907)
JOINT_4_RANGE: tuple[float, float] = (-2.5795966344476193, 2.5795966344476193)
JOINT_6_RANGE: tuple[float, float] = (-2.0996310901491784, 2.0996310901491784)

# -- gripper (Robotiq 2F-85) --------------------------------------------
FINGERS_ACTR_CTRLRANGE: tuple[float, float] = (0.0, 255.0)
FINGERS_ACTR_FORCERANGE: tuple[float, float] = (-5.0, 5.0)

# -- keyframes ----------------------------------------------------------
KEYFRAME_HOME: str = "home"
KEYFRAME_RETRACT: str = "retract"

# -- robot XML paths ----------------------------------------------------
ROBOT_XML_NAME: str = "tidybot.xml"
SCENE_XML_NAME: str = "scene.xml"
