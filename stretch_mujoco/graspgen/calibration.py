"""Camera calibration and frame conversions for the simulated D435i."""

from __future__ import annotations

import numpy as np

from stretch_mujoco.enums.stretch_cameras import StretchCameras
from stretch_mujoco.utils import compute_K


ROTATED_TO_RAW_OPTICAL = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=float)


def d435i_rotated_camera_intrinsics() -> np.ndarray:
    """Return K matching the clockwise-rotated 240x424 D435i image."""
    settings = StretchCameras.cam_d435i_rgb.initial_camera_settings
    raw_k = compute_K(settings.field_of_view_vertical_in_degrees, settings.width, settings.height)
    return np.array(
        [
            [raw_k[1, 1], 0.0, settings.height / 2.0],
            [0.0, raw_k[0, 0], settings.width / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def rotated_d435i_optical_pose(raw_optical_pose: np.ndarray) -> np.ndarray:
    """Convert a raw optical-frame pose to the frame of the rotated image."""
    pose = np.asarray(raw_optical_pose, dtype=float)
    if pose.shape != (4, 4):
        raise ValueError(f"Expected a 4x4 optical pose, got {pose.shape}")
    transform = np.eye(4)
    transform[:3, :3] = ROTATED_TO_RAW_OPTICAL
    return pose @ transform


def planar_world_from_base(
    base_xyt: tuple[float, float, float] | np.ndarray,
) -> np.ndarray:
    """Return the 4x4 planar world-from-base transform for base pose (x, y, theta).

    The base lives in the z=0 plane; the transform maps a point in the base frame
    into the world frame. Inverse of :func:`world_to_base_pose`.
    """
    x, y, theta = np.asarray(base_xyt, dtype=float)[:3]
    cosine, sine = np.cos(theta), np.sin(theta)
    return np.array(
        [
            [cosine, -sine, 0.0, x],
            [sine, cosine, 0.0, y],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def world_to_base_pose(base_xyt: tuple[float, float, float] | np.ndarray) -> np.ndarray:
    """Return the 4x4 base-from-world transform for base pose (x, y, theta)."""
    return np.linalg.inv(planar_world_from_base(base_xyt))
