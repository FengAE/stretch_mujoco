"""Off-screen RGB and depth image capture from MuJoCo.

Captures first-person views that are sent to the VLM for semantic scoring
and to the value map for spatial projection.
"""

from __future__ import annotations

import copy
import io
import math
import threading
from typing import Optional

import mujoco
import numpy as np
from PIL import Image


class ImageCapture:
    """Render RGB and depth images from a MuJoCo scene.

    Uses MuJoCo's off-screen renderer.

    Parameters
    ----------
    model, data:
        MuJoCo model and data buffers.
    width, height:
        Render resolution.
    camera_name:
        Name of a fixed camera defined in the scene XML.  If *None* a
        free camera is positioned at the robot.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        *,
        width: int = 480,
        height: int = 360,
        camera_name: Optional[str] = None,
    ) -> None:
        if not isinstance(width, int) or isinstance(width, bool) or width <= 0:
            raise ValueError("width must be a positive integer")
        if not isinstance(height, int) or isinstance(height, bool) or height <= 0:
            raise ValueError("height must be a positive integer")
        if camera_name is not None:
            camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
            if camera_id < 0:
                raise ValueError(f"unknown MuJoCo camera: {camera_name}")

        self.model = model
        self.data = data
        self.width = width
        self.height = height
        self.camera_name = camera_name
        self._camera_id = camera_id if camera_name is not None else None
        self._render_lock = threading.RLock()
        self._last_capture_pose: Optional[tuple[tuple[float, float], float]] = None

        # Off-screen renderers (RGB and depth)
        self._renderer = mujoco.Renderer(model, height, width)

    def close(self) -> None:
        self._renderer.close()

    def horizontal_fov_rad(self, fovy: Optional[float] = None) -> float:
        """Return the rendered horizontal FOV in radians.

        MuJoCo camera FOV values are vertical. The value-map projection needs
        horizontal FOV, which also depends on the render aspect ratio.
        """
        if self._camera_id is not None:
            effective_fovy = float(self.model.cam_fovy[self._camera_id])
        elif fovy is None:
            effective_fovy = float(self.model.vis.global_.fovy)
        else:
            effective_fovy = float(fovy)
        if not np.isfinite(effective_fovy) or not 1.0 <= effective_fovy <= 179.0:
            raise ValueError("fovy must be in [1, 179] degrees")
        vertical = math.radians(effective_fovy)
        return 2.0 * math.atan((self.width / self.height) * math.tan(vertical / 2.0))

    # ------------------------------------------------------------------
    # RGB capture
    # ------------------------------------------------------------------
    def capture_rgb_array(
        self,
        position: tuple[float, float, float],
        yaw: float,
        pitch: float = -0.3,
        fovy: float = 70.0,
    ) -> np.ndarray:
        """Render an RGB image without PNG encoding."""
        pos = np.array(position, dtype=float)
        if pos.shape != (3,) or not np.all(np.isfinite(pos)):
            raise ValueError("position must contain three finite coordinates")
        if not np.isfinite(yaw) or not np.isfinite(pitch):
            raise ValueError("yaw and pitch must be finite")
        if not np.isfinite(fovy) or not 1.0 <= fovy <= 179.0:
            raise ValueError("fovy must be in [1, 179] degrees")

        cam = self._make_camera(pos, yaw, pitch)
        old_fovy = float(self.model.vis.global_.fovy)
        try:
            if self.camera_name is None:
                self.model.vis.global_.fovy = float(fovy)
            self._renderer.update_scene(
                self.data,
                camera=self.camera_name if self.camera_name is not None else cam,
                scene_option=mujoco.MjvOption(),
            )
            pixels = self._renderer.render()
        finally:
            self.model.vis.global_.fovy = old_fovy
        return _to_uint8_rgb(pixels)

    def capture_rgb(
        self,
        position: tuple[float, float, float],
        yaw: float,
        pitch: float = -0.3,
        fovy: float = 70.0,
    ) -> bytes:
        """Render an RGB image from a virtual camera at *position*.

        Parameters
        ----------
        position:
            World (x, y, z) of the virtual camera.
        yaw:
            Camera yaw angle (radians, 0 = +X, π/2 = +Y).
        pitch:
            Camera pitch (radians, negative = look down).
        fovy:
            Vertical field-of-view in degrees.

        Returns
        -------
        bytes
            PNG-encoded RGB image.
        """
        rgb = self.capture_rgb_array(position, yaw, pitch, fovy)
        img = Image.fromarray(rgb, mode="RGB")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    # ------------------------------------------------------------------
    # Depth capture
    # ------------------------------------------------------------------
    def capture_depth(
        self,
        position: tuple[float, float, float],
        yaw: float,
        pitch: float = -0.3,
        fovy: float = 70.0,
    ) -> np.ndarray:
        """Render a depth image from a virtual camera at *position*.

        Uses MuJoCo's depth rendering, which returns distances from the
        camera plane in metres.

        Parameters
        ----------
        position:
            World (x, y, z) of the virtual camera.
        yaw:
            Camera yaw angle (radians).
        pitch:
            Camera pitch (radians, negative = look down).
        fovy:
            Vertical field-of-view in degrees.

        Returns
        -------
        np.ndarray
            ``(height, width)`` float32 array of depth values in metres.
        """
        pos = np.array(position, dtype=float)
        if pos.shape != (3,) or not np.all(np.isfinite(pos)):
            raise ValueError("position must contain three finite coordinates")
        if not np.isfinite(yaw) or not np.isfinite(pitch):
            raise ValueError("yaw and pitch must be finite")
        if not np.isfinite(fovy) or not 1.0 <= fovy <= 179.0:
            raise ValueError("fovy must be in [1, 179] degrees")
        cam = self._make_camera(pos, yaw, pitch)

        old_fovy = float(self.model.vis.global_.fovy)
        try:
            if self.camera_name is None:
                self.model.vis.global_.fovy = float(fovy)
            self._renderer.update_scene(
                self.data,
                camera=self.camera_name if self.camera_name is not None else cam,
                scene_option=mujoco.MjvOption(),
            )
            # Enable depth rendering
            self._renderer.enable_depth_rendering()
            try:
                depth_pixels = self._renderer.render()
            finally:
                self._renderer.disable_depth_rendering()
        finally:
            self.model.vis.global_.fovy = old_fovy

        # MuJoCo depth is (height, width) float32 in metres
        if depth_pixels.ndim == 3:
            depth_pixels = depth_pixels[:, :, 0]
        return np.asarray(depth_pixels, dtype=np.float32)

    def capture_rgbd(
        self,
        position: tuple[float, float, float],
        yaw: float,
        pitch: float = -0.3,
        fovy: float = 70.0,
    ) -> tuple[bytes, np.ndarray]:
        """Capture both RGB and depth in one call.

        Returns
        -------
        rgb_bytes:
            PNG-encoded RGB image.
        depth:
            ``(height, width)`` float32 depth array in metres.
        """
        # Render both modalities from one immutable MjData snapshot. This
        # prevents a concurrently stepping simulator from moving the camera or
        # scene between the RGB and depth renders.
        with self._render_lock:
            live_data = self.data
            self.data = copy.copy(live_data)
            try:
                rgb = self.capture_rgb(position, yaw, pitch, fovy)
                depth = self.capture_depth(position, yaw, pitch, fovy)
                self._last_capture_pose = self.camera_pose_xy_yaw(
                    (float(position[0]), float(position[1])), yaw
                )
            finally:
                self.data = live_data
        return rgb, depth

    def camera_pose_xy_yaw(
        self,
        fallback_xy: tuple[float, float],
        fallback_yaw: float,
    ) -> tuple[tuple[float, float], float]:
        """Return the world pose of the camera used for rendering.

        Free-camera captures use the supplied pose. Named MuJoCo cameras use
        their actual world transform from ``MjData`` so value-map projection
        cannot accidentally use the robot base pose.
        """
        fallback = np.asarray(fallback_xy, dtype=float)
        if fallback.shape != (2,) or not np.all(np.isfinite(fallback)):
            raise ValueError("fallback_xy must contain two finite coordinates")
        if not np.isfinite(fallback_yaw):
            raise ValueError("fallback_yaw must be finite")
        if self._camera_id is None:
            return (float(fallback[0]), float(fallback[1])), float(fallback_yaw)

        position = np.asarray(self.data.cam_xpos[self._camera_id], dtype=float)
        rotation = np.asarray(self.data.cam_xmat[self._camera_id], dtype=float).reshape(3, 3)
        forward = -(rotation @ np.array([0.0, 0.0, 1.0]))
        yaw = math.atan2(float(forward[1]), float(forward[0]))
        return (float(position[0]), float(position[1])), yaw

    def capture_robot_observation(
        self,
        robot_xy: tuple[float, float],
        robot_yaw: float,
        camera_height: float = 0.5,
        pitch: float = -0.3,
        fovy: float = 70.0,
    ) -> tuple[bytes, np.ndarray, tuple[float, float], float, float]:
        """Capture RGB-D together with its projection pose and horizontal FOV."""
        rgb, depth = self.capture_at_robot(
            robot_xy,
            robot_yaw,
            camera_height=camera_height,
            pitch=pitch,
            fovy=fovy,
        )
        if self._last_capture_pose is None:
            raise RuntimeError("RGB-D capture did not produce a camera pose")
        camera_xy, camera_yaw = self._last_capture_pose
        return rgb, depth, camera_xy, camera_yaw, self.horizontal_fov_rad(fovy)

    # ------------------------------------------------------------------
    # Convenience: capture at robot pose
    # ------------------------------------------------------------------
    def capture_at_robot(
        self,
        robot_xy: tuple[float, float],
        robot_yaw: float,
        camera_height: float = 0.5,
        pitch: float = -0.3,
        fovy: float = 70.0,
    ) -> tuple[bytes, np.ndarray]:
        """Capture RGB + depth from the robot's current pose.

        The camera is positioned at ``(robot_xy[0], robot_xy[1], camera_height)``
        and oriented along *robot_yaw*.

        Returns
        -------
        rgb_bytes:
            PNG-encoded RGB image.
        depth:
            ``(height, width)`` float32 depth array in metres.
        """
        return self.capture_rgbd(
            position=(robot_xy[0], robot_xy[1], camera_height),
            yaw=robot_yaw,
            pitch=pitch,
            fovy=fovy,
        )

    # ------------------------------------------------------------------
    # Panorama capture (backward-compatible)
    # ------------------------------------------------------------------
    def capture_panorama(
        self,
        robot_xy: tuple[float, float],
        robot_z: float = 0.5,
        num_directions: int = 8,
        heading_yaw: float = 0.0,
        pitch: float = -0.2,
        fovy: float = 75.0,
    ) -> list[bytes]:
        """Capture *num_directions* RGB images around the robot.

        Directions are evenly spaced starting from *heading_yaw*.

        Returns
        -------
        list[bytes]
            PNG images indexed clockwise from the heading.
        """
        if (
            not isinstance(num_directions, int)
            or isinstance(num_directions, bool)
            or num_directions <= 0
        ):
            raise ValueError("num_directions must be a positive integer")
        if not np.isfinite(heading_yaw):
            raise ValueError("heading_yaw must be finite")

        images: list[bytes] = []
        x, y = robot_xy
        for i in range(num_directions):
            yaw = heading_yaw + 2.0 * math.pi * i / num_directions
            img = self.capture_rgb(
                position=(x, y, robot_z),
                yaw=yaw,
                pitch=pitch,
                fovy=fovy,
            )
            images.append(img)
        return images

    def capture_at_pose(
        self,
        position: tuple[float, float, float],
        yaw: float,
        pitch: float = -0.3,
        fovy: float = 70.0,
    ) -> bytes:
        """Backward-compatible instance alias for :meth:`capture_rgb`."""
        return self.capture_rgb(position, yaw, pitch, fovy)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _make_camera(
        position: np.ndarray,
        yaw: float,
        pitch: float,
    ) -> mujoco.MjvCamera:
        """Build a free MuJoCo camera at *position* looking along *yaw*."""
        forward = np.array(
            [
                math.cos(pitch) * math.cos(yaw),
                math.cos(pitch) * math.sin(yaw),
                math.sin(pitch),
            ]
        )
        target = position + forward

        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(cam)
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        direction = target - position
        dist = float(np.linalg.norm(direction))
        if dist < 1e-6:
            direction = np.array([1.0, 0.0, 0.0])
            dist = 1.0

        cam.lookat[:] = target
        cam.distance = dist
        cam.azimuth = float(math.degrees(math.atan2(direction[1], direction[0]))) - 90
        cam.elevation = float(
            math.degrees(
                math.atan2(
                    direction[2],
                    math.sqrt(direction[0] ** 2 + direction[1] ** 2),
                )
            )
        )
        return cam


def _to_uint8_rgb(pixels: np.ndarray) -> np.ndarray:
    """Convert MuJoCo renderer output to uint8 RGB."""
    if np.issubdtype(pixels.dtype, np.floating):
        return np.clip(pixels * 255.0, 0.0, 255.0).astype(np.uint8)
    return np.clip(pixels, 0, 255).astype(np.uint8, copy=False)
