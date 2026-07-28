"""Stretch 3 hardware configuration parameters."""

# -- wheels ---------------------------------------------------------------
WHEEL_DIAMETER: float = 0.1016  # m
WHEEL_SEPARATION: float = 0.3153  # m

# -- gripper --------------------------------------------------------------
GRIPPER_MIN_MAX: tuple[float, float] = (-0.376, 0.56)  # real range
SIM_GRIPPER_MIN_MAX: tuple[float, float] = (-0.02, 0.04)  # simulation range

# -- depth cameras --------------------------------------------------------
DEPTH_LIMITS: dict[str, float] = {
    "d405": 1.0,
    "d435i": 10.0,
}

# -- base motion ----------------------------------------------------------
BASE_MOTION_TIMEOUT: float = 15.0
BASE_DEFAULT_X_VEL: float = 0.3  # m/s
BASE_DEFAULT_R_VEL: float = 1.0  # rad/s

# -- robot settings (aggregated for backward compatibility) ----------------
robot_settings = {
    "wheel_diameter": WHEEL_DIAMETER,
    "wheel_separation": WHEEL_SEPARATION,
    "gripper_min_max": GRIPPER_MIN_MAX,
    "sim_gripper_min_max": SIM_GRIPPER_MIN_MAX,
}

depth_limits = DEPTH_LIMITS
base_motion = {
    "timeout": BASE_MOTION_TIMEOUT,
    "default_x_vel": BASE_DEFAULT_X_VEL,
    "default_r_vel": BASE_DEFAULT_R_VEL,
}
