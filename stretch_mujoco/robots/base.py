"""Abstract base classes for multi-robot simulator abstraction.

Every robot supported by the framework must provide concrete
implementations of the enums and dataclasses defined here.
"""

from __future__ import annotations

from abc import ABC, ABCMeta, abstractmethod
from dataclasses import dataclass
from enum import Enum, EnumMeta
from typing import Any, Callable, Generic, TypeVar

import numpy as np

from stretch_mujoco.datamodels.camera import CameraSettings
from stretch_mujoco.datamodels.sensors import SensorMetadata

# ---------------------------------------------------------------------------
# Type variables for generic robot typing
# ---------------------------------------------------------------------------

ActuatorT = TypeVar("ActuatorT", bound="RobotActuators")
CameraT = TypeVar("CameraT", bound="RobotCameras")
SensorT = TypeVar("SensorT", bound="RobotSensors")
SiteT = TypeVar("SiteT", bound="RobotBodySites")
StatusT = TypeVar("StatusT", bound="RobotStatus")
CameraDataT = TypeVar("CameraDataT", bound="RobotCameraData")
SensorDataT = TypeVar("SensorDataT", bound="RobotSensorData")


class ABCEnumMeta(ABCMeta, EnumMeta):
    """Metaclass for enums that also enforce abstract interface methods."""


# ===================================================================
# Actuator type taxonomy
# ===================================================================


class ActuatorType(str, Enum):
    """Kinds of actuators a robot joint may expose."""

    POSITION = "position"
    """Standard position servo with kp/kv gains."""

    VELOCITY = "velocity"
    """Velocity-controlled actuator (e.g. diff-drive wheels)."""

    GENERAL = "general"
    """General actuator (e.g. tendon-driven gripper with raw 0-255 ctrl)."""

    MOTOR = "motor"
    """Torque / current-controlled motor (reserved for future use)."""


# ===================================================================
# Body sites
# ===================================================================


@dataclass(frozen=True)
class BodySiteRef:
    """Describes a named reference frame on a robot body.

    Used for forward-kinematics queries, grasping, and camera mounts.
    """

    name: str
    """MuJoCo body or site name (e.g. ``"link_grasp_center"``)."""

    mjcf_kind: str = "body"
    """``"body"`` or ``"site"``."""

    purpose: str = "generic"
    """Semantic tag: ``"ee_grasp"``, ``"camera"``, ``"imu"``, ``"lidar"``, ``"marker"``."""


class RobotBodySites(Enum, metaclass=ABCEnumMeta):
    """Named reference frames that every robot implementation must enumerate."""

    @property
    @abstractmethod
    def site_ref(self) -> BodySiteRef:
        """Metadata for this site."""

    @staticmethod
    @abstractmethod
    def grasp_sites() -> list["RobotBodySites"]:
        """Sites usable as end-effector grasp frames."""

    @staticmethod
    @abstractmethod
    def camera_sites() -> list["RobotBodySites"]:
        """Sites where cameras are mounted."""

    @staticmethod
    @abstractmethod
    def all() -> list["RobotBodySites"]:
        """Every registered site."""


# ===================================================================
# Actuators
# ===================================================================


class RobotActuators(Enum, metaclass=ABCEnumMeta):
    """Actuator / joint enumeration that every robot must provide."""

    # -- identification -------------------------------------------------

    @abstractmethod
    def get_joint_names_in_mjcf(self) -> list[str]:
        """MuJoCo joint name(s) driven by this actuator."""

    @abstractmethod
    def actuator_type(self) -> ActuatorType:
        """Whether this is position-, velocity-, general-, or motor-controlled."""

    @abstractmethod
    def is_base_actuator(self) -> bool:
        """True for actuators that move the mobile base."""

    @property
    @abstractmethod
    def has_position_control(self) -> bool:
        """True if this actuator supports absolute position targets."""

    @property
    @abstractmethod
    def has_velocity_control(self) -> bool:
        """True if this actuator supports velocity control."""

    # -- position / velocity queries ------------------------------------

    @abstractmethod
    def get_position(self, status: "RobotStatus") -> float:
        """Return the scalar joint position (not valid for base actuators)."""

    @abstractmethod
    def get_velocity(self, status: "RobotStatus") -> float:
        """Return the scalar joint velocity (not valid for base actuators)."""

    @abstractmethod
    def get_position_relative(self, status: "RobotStatus") -> tuple[float, float, float]:
        """Return base pose (x, y, theta) — only valid for base actuators."""

    @abstractmethod
    def get_velocity_relative(self, status: "RobotStatus") -> tuple[float, float, float]:
        """Return base velocity (x_vel, y_vel, theta_vel) — only valid for base actuators."""

    # -- range / limits -------------------------------------------------

    def get_ctrlrange(self, model: Any) -> tuple[float, float] | None:
        """Return (min, max) from the MuJoCo actuator, or None if unlimited.

        The default implementation returns ``None``.  Robot-specific
        enums may override this to query the model at runtime.
        """
        return None

    # -- resolution -----------------------------------------------------

    @staticmethod
    @abstractmethod
    def get_actuator_by_joint_names_in_mjcf(joint_name: str) -> "RobotActuators":
        """Reverse-lookup: MuJoCo joint name → actuator enum member."""

    @staticmethod
    @abstractmethod
    def all() -> list["RobotActuators"]:
        """Every registered actuator."""


# ===================================================================
# Cameras
# ===================================================================


class RobotCameras(Enum, metaclass=ABCEnumMeta):
    """Camera enumeration that every robot must provide."""

    @property
    @abstractmethod
    def camera_name_in_mjcf(self) -> str:
        """Name of the MuJoCo camera element."""

    @property
    @abstractmethod
    def is_depth(self) -> bool:
        """True for depth cameras."""

    @property
    def is_rgb(self) -> bool:
        """True for RGB / monocular cameras (default: not depth)."""
        return not self.is_depth

    @property
    def is_simulated(self) -> bool:
        """Whether this camera exists only in simulation."""
        return False

    @property
    def post_processing_callback(self) -> Callable[[np.ndarray], np.ndarray] | None:
        """Optional per-camera post-processing hook (e.g. depth clipping)."""
        return None

    @property
    @abstractmethod
    def initial_camera_settings(self) -> CameraSettings:
        """Intrinsic and rendering settings for this camera."""

    @staticmethod
    @abstractmethod
    def all() -> list["RobotCameras"]:
        """Every registered camera."""

    @staticmethod
    @abstractmethod
    def rgb() -> list["RobotCameras"]:
        """RGB-only subset."""

    @staticmethod
    @abstractmethod
    def depth() -> list["RobotCameras"]:
        """Depth-only subset."""

    @staticmethod
    def none() -> list["RobotCameras"]:
        """Convenience for robots without cameras."""
        return []


# ===================================================================
# Sensors
# ===================================================================


class RobotSensors(Enum, metaclass=ABCEnumMeta):
    """Sensor enumeration that every robot must provide."""

    @property
    @abstractmethod
    def sensor_name_in_mjcf(self) -> str:
        """MuJoCo sensor name, or prefix for replicated sensors (e.g. lidar)."""

    @property
    @abstractmethod
    def metadata(self) -> SensorMetadata:
        """Shape, dtype, units, and frame for this sensor's raw data."""

    @property
    def is_replicated(self) -> bool:
        """True when the sensor is modelled as N individual MuJoCo sensors
        (e.g. a 2-D lidar as an array of rangefinders)."""
        return False

    def get_replicated_names(self, resolution: int) -> list[str]:
        """Return all MuJoCo sensor names when ``is_replicated`` is True."""
        _ = resolution
        return [self.sensor_name_in_mjcf]

    @staticmethod
    @abstractmethod
    def all() -> list["RobotSensors"]:
        """Every registered sensor."""

    @staticmethod
    def none() -> list["RobotSensors"]:
        """Convenience for robots without sensors."""
        return []

    @staticmethod
    @abstractmethod
    def from_mjmodel(mjmodel: Any) -> list["RobotSensors"]:
        """Auto-detect which sensors actually exist in an MjModel."""


# ===================================================================
# Status / data containers
# ===================================================================


class RobotStatus(ABC):
    """Joint-state snapshot returned by ``pull_status()``."""

    time: float
    fps: float
    sim_to_real_time_ratio_msg: str

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict."""

    @staticmethod
    @abstractmethod
    def from_dict(dict_data: dict[str, Any]) -> "RobotStatus":
        """Deserialize from a plain dict."""

    @staticmethod
    @abstractmethod
    def default() -> "RobotStatus":
        """Return a zero / empty instance."""


class RobotCameraData(ABC):
    """Camera-data snapshot returned by ``pull_camera_data()``."""

    time: float
    fps: float

    @abstractmethod
    def get_camera_data(
        self,
        camera: RobotCameras,
        *,
        auto_rotate: bool = True,
        auto_correct_rgb: bool = True,
        use_depth_color_map: bool = False,
    ) -> np.ndarray:
        """Return pixel data for *camera*, applying optional corrections."""

    @abstractmethod
    def set_camera_data(self, camera: RobotCameras, data: np.ndarray) -> None:
        """Store pixel data for *camera*."""

    @abstractmethod
    def get_intrinsic_matrix(self, camera: RobotCameras) -> np.ndarray:
        """Return the 3×3 intrinsic matrix K for *camera*."""

    @abstractmethod
    def get_all(self, **kwargs: Any) -> dict[RobotCameras, np.ndarray]:
        """Return all available camera images."""

    @staticmethod
    @abstractmethod
    def default() -> "RobotCameraData":
        """Return a zero / empty instance."""

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict."""

    @abstractmethod
    def copy(self) -> "RobotCameraData":
        """Deep-copy this snapshot."""


class RobotSensorData(ABC):
    """Sensor-data snapshot returned by ``pull_sensor_data()``."""

    time: float
    fps: float

    @abstractmethod
    def get_data(self, sensor: RobotSensors) -> np.ndarray:
        """Return the raw sensor reading."""

    @abstractmethod
    def set_data(self, sensor: RobotSensors, value: np.ndarray) -> None:
        """Store a raw sensor reading."""

    @abstractmethod
    def get_metadata(self, sensor: RobotSensors) -> SensorMetadata:
        """Return the SensorMetadata for *sensor*."""

    @abstractmethod
    def get_all(self) -> dict[RobotSensors, np.ndarray]:
        """Return all available sensor readings."""

    @staticmethod
    @abstractmethod
    def default() -> "RobotSensorData":
        """Return a zero / empty instance."""

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict."""

    @abstractmethod
    def copy(self) -> "RobotSensorData":
        """Deep-copy this snapshot."""


# ===================================================================
# Base controller
# ===================================================================


class RobotBaseController(ABC):
    """Abstract base locomotion controller.

    Concrete implementations handle diff-drive, omnidirectional, Ackermann, etc.
    """

    @abstractmethod
    def push_command(self, command: Any) -> None:
        """Push a movement command to the base."""

    @abstractmethod
    def update(self) -> None:
        """Called every physics step to write MuJoCo ctrl values."""

    @abstractmethod
    def get_base_pose(self) -> np.ndarray:
        """Return se(2) pose as ``[x, y, theta]``."""

    @abstractmethod
    def handle_move_by(self, command: Any) -> None:
        """Execute a relative displacement command."""

    @abstractmethod
    def _set_base_velocity(self, v_linear: float, omega: float) -> None:
        """Set instantaneous base velocity."""

    @abstractmethod
    def stop(self) -> None:
        """Stop all base motion."""


# ===================================================================
# Top-level simulator interface
# ===================================================================


class RobotSimulator(ABC):
    """Unified interface for any robot in MuJoCo.

    Concrete classes provide the robot-specific enums and wire up the
    MuJoCo server process.
    """

    # -- lifecycle -------------------------------------------------------

    @abstractmethod
    def start(
        self,
        show_viewer_ui: bool = False,
        headless: bool = False,
        use_passive_viewer: bool = True,
    ) -> None:
        """Launch the MuJoCo simulation process."""

    @abstractmethod
    def stop(self) -> None:
        """Gracefully terminate the simulation."""

    @abstractmethod
    def is_running(self) -> bool:
        """Return True while the MuJoCo process is alive."""

    # -- movement --------------------------------------------------------

    @abstractmethod
    def move_to(self, actuator: RobotActuators, pos: float) -> None:
        """Command an absolute position target.

        For *general* actuators (e.g. TidyBot tendon gripper) *pos* is
        interpreted as a raw ctrl value.
        """

    @abstractmethod
    def move_by(self, actuator: RobotActuators, pos: float) -> None:
        """Command a relative position increment."""

    @abstractmethod
    def set_base_velocity(self, v_linear: float, omega: float, v_lateral: float = 0.0) -> None:
        """Command base velocity; omnidirectional bases also accept lateral speed."""

    @abstractmethod
    def home(self) -> None:
        """Move the robot to its home keyframe (if defined)."""

    @abstractmethod
    def stow(self) -> None:
        """Move the robot to its stow keyframe (if defined)."""

    @abstractmethod
    def wait_until_at_setpoint(
        self,
        actuator: RobotActuators,
        timeout: float = 5.0,
        position_tolerance: float = 0.05,
    ) -> bool:
        """Block until *actuator* reaches its commanded ``move_to`` target."""

    @abstractmethod
    def wait_while_is_moving(
        self,
        actuator: RobotActuators,
        timeout: float | None = 5.0,
        check_interval: float = 0.1,
        position_tolerance: float = 0.0005,
    ) -> bool:
        """Block while *actuator* is still in motion."""

    @abstractmethod
    def is_reached_set_position(
        self,
        actuator: RobotActuators,
        position_tolerance: float = 0.05,
    ) -> bool:
        """Return True when *actuator* is within tolerance of its target."""

    # -- state queries ---------------------------------------------------

    @abstractmethod
    def pull_status(self) -> RobotStatus:
        """Return the latest joint-state snapshot."""

    @abstractmethod
    def get_base_pose(self) -> tuple[float, float, float]:
        """Return ``(x, y, theta)`` in world coordinates."""

    @abstractmethod
    def get_ee_pose(self) -> np.ndarray:
        """Return the 4×4 end-effector pose in world coordinates."""

    @abstractmethod
    def get_link_pose(self, link_name: str) -> np.ndarray:
        """Return the 4×4 pose of *link_name* in world coordinates."""

    @abstractmethod
    def get_site_pose(self, site: RobotBodySites) -> np.ndarray:
        """Return the 4×4 pose of a body site in world coordinates."""

    # -- camera ----------------------------------------------------------

    @abstractmethod
    def pull_camera_data(self) -> RobotCameraData:
        """Return the latest camera snapshot."""

    def get_camera_intrinsics(self, camera: RobotCameras) -> np.ndarray:
        """Return the 3×3 K matrix for *camera*."""
        return self.pull_camera_data().get_intrinsic_matrix(camera)

    # -- sensors ---------------------------------------------------------

    @abstractmethod
    def pull_sensor_data(self) -> RobotSensorData:
        """Return the latest sensor snapshot."""

    def get_sensor_metadata(self, sensor: RobotSensors) -> SensorMetadata:
        """Return metadata for *sensor*."""
        return self.pull_sensor_data().get_metadata(sensor)

    @property
    @abstractmethod
    def available_sensors(self) -> list[RobotSensors]:
        """Sensors that are actually present in the current model."""

    # -- grasping --------------------------------------------------------

    @abstractmethod
    def attach_object_to_gripper(self, object_id: str) -> None:
        """Fix *object_id* to the gripper for a stable grasp."""

    @abstractmethod
    def release_grasped_object(self) -> None:
        """Release a previously attached object."""

    # -- utility ---------------------------------------------------------

    @abstractmethod
    def set_object_visibility(self, object_id: str, visible: bool) -> None:
        """Show or hide a scene object."""

    @abstractmethod
    def pull_joint_limits(self) -> dict[RobotActuators, tuple[float, float]]:
        """Return per-actuator (min, max) joint limits."""

    @abstractmethod
    def add_world_frame(
        self,
        position: tuple[float, float, float],
        rotation: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None:
        """Visualise a coordinate frame in the scene."""

    # -- robot identity (class-level accessors) --------------------------

    @property
    @abstractmethod
    def Actuators(self) -> type[RobotActuators]:
        """The robot's actuator enum type."""

    @property
    @abstractmethod
    def Cameras(self) -> type[RobotCameras]:
        """The robot's camera enum type."""

    @property
    @abstractmethod
    def Sensors(self) -> type[RobotSensors]:
        """The robot's sensor enum type."""

    @property
    @abstractmethod
    def Sites(self) -> type[RobotBodySites]:
        """The robot's body-site enum type."""
