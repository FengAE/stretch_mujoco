"""Shared camera parameter types used across all robot camera enums."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CameraCrop:
    """Region-of-interest crop for a camera image."""

    x_min: int
    x_max: int
    y_min: int
    y_max: int

    @property
    def x_offset(self) -> int:
        return self.x_min

    @property
    def y_offset(self) -> int:
        return self.y_min

    @property
    def width(self) -> int:
        return self.x_max - self.x_min

    @property
    def height(self) -> int:
        return self.y_max - self.y_min


@dataclass
class CameraSettings:
    """Camera intrinsic and rendering parameters.

    This replaces the Stretch-specific CameraSettings previously defined in
    ``enums/stretch_cameras.py`` and is now usable by any robot.
    """

    field_of_view_vertical_in_degrees: float
    """Vertical FOV for the camera in degrees."""

    focal: tuple[float, float]
    """(fx, fy) Focal lengths in pixels."""

    width: int
    """Width of the rendered image (may differ from sensor_resolution)."""

    height: int
    """Height of the rendered image (may differ from sensor_resolution)."""

    sensor_resolution: tuple[float, float] | None = None
    """The maximum resolution of the image sensor."""

    sensor_pixel_size_micrometers: float | None = None
    """The size of a single pixel in µm."""

    sensor_size_millimeters: tuple[float, float] | None = None
    """Physical sensor dimensions.  Computed from sensor_resolution +
    sensor_pixel_size_micrometers when not provided explicitly."""

    crop: CameraCrop | None = None
    """Optional ROI crop applied after rendering."""

    distortion_params: tuple[float, ...] | None = None
    """Distortion coefficients (e.g. plumb_bob k1,k2,t1,t2,k3).
    Zeros are returned by get_distortion_params_d() when not set."""

    principal_point: tuple[float, float] | None = None
    """Optional calibrated ``(cx, cy)`` in rendered-image pixels."""

    # ------------------------------------------------------------------
    # Computed properties
    # ------------------------------------------------------------------

    @property
    def sensor_size(self) -> tuple[float, float] | None:
        """Physical sensor size in mm, computed when possible."""
        if self.sensor_size_millimeters is not None:
            return self.sensor_size_millimeters
        if self.sensor_pixel_size_micrometers is None or self.sensor_resolution is None:
            return None
        return (
            self.sensor_pixel_size_micrometers * self.sensor_resolution[0] / 1000.0,
            self.sensor_pixel_size_micrometers * self.sensor_resolution[1] / 1000.0,
        )

    # ------------------------------------------------------------------
    # Intrinsic / projection helpers
    # ------------------------------------------------------------------

    def get_intrinsic_params_k(self) -> list[float]:
        """3×3 intrinsic camera matrix K (row-major).

        K = [[fx,  0, cx],
             [ 0, fy, cy],
             [ 0,  0,  1]]
        """
        cx, cy = self.principal_point or (self.width / 2.0, self.height / 2.0)
        return [
            self.focal[0],
            0.0,
            cx,
            0.0,
            self.focal[1],
            cy,
            0.0,
            0.0,
            1.0,
        ]

    def get_projection_matrix_p(self) -> list[float]:
        """3×4 projection matrix P (row-major).

        P = [[fx′,  0, cx′, 0],
             [ 0,  fy′, cy′, 0],
             [ 0,   0,   1,  0]]
        """
        cx, cy = self.principal_point or (self.width / 2.0, self.height / 2.0)
        return [
            self.focal[0],
            0.0,
            cx,
            0.0,
            0.0,
            self.focal[1],
            cy,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
        ]

    def get_distortion_params_d(self) -> list[float]:
        """Distortion coefficients.

        For the common "plumb_bob" model: [k1, k2, t1, t2, k3].
        Returns zeros when no distortion parameters are configured.
        """
        if self.distortion_params is not None:
            return list(self.distortion_params)
        return [0.0, 0.0, 0.0, 0.0, 0.0]

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def field_of_view_vertical_from_horizontal(
        fov_horizontal_degrees: float,
        width: int,
        height: int,
    ) -> float:
        """Convert horizontal FOV to vertical FOV using the aspect ratio."""
        horizontal_fov = np.radians(fov_horizontal_degrees)
        aspect_ratio = width / height
        vertical_fov = np.rad2deg(2.0 * np.arctan(np.tan(horizontal_fov / 2.0) * aspect_ratio))
        return float(abs(vertical_fov))
