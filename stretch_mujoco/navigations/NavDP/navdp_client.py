"""HTTP client for the NavDP System-1 inference server.

Talks to ``navdp_server.py`` (from the InternRobotics/NavDP repo) over its HTTP
API:

- ``POST /navigator_reset`` — (re)initialise the policy with camera intrinsics.
- ``POST /pointgoal_step`` — feed one RGB-D frame plus a **body-frame** goal
  and receive a **body-frame** waypoint trajectory.

The server keeps a rolling 8-frame memory internally, so the client sends one
frame per call (the caller does not need to stack history).

Only the Python standard library is used (no ``requests``), mirroring
``stretch_mujoco.navigations.VLFM.vlm_client``.
"""

from __future__ import annotations

import io
import json
import uuid
from urllib import error as urllib_error
from urllib import request as urllib_request

import cv2
import numpy as np


class NavDPClient:
    """Minimal HTTP client for a NavDP ``navdp_server.py`` instance."""

    def __init__(self, base_url: str = "http://127.0.0.1:8888", timeout: float = 30.0) -> None:
        if not isinstance(base_url, str) or not base_url.strip():
            raise ValueError("base_url must be a non-empty string")
        if not base_url.startswith(("http://", "https://")):
            raise ValueError("base_url must start with http:// or https://")
        if not np.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be finite and > 0")
        self.base_url = base_url.rstrip("/")
        self.timeout = float(timeout)
        # NavDP normally runs on localhost or a trusted LAN host. Bypass
        # process-wide HTTP proxies so loopback requests cannot be intercepted.
        self._opener = urllib_request.build_opener(urllib_request.ProxyHandler({}))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(
        self,
        intrinsic: np.ndarray | list,
        stop_threshold: float = -0.5,
        batch_size: int = 1,
    ) -> dict:
        """Reset the NavDP policy (called once per navigation episode).

        The server instantiates the model on the *first* reset call using the
        supplied ``intrinsic``, so the camera calibration must be passed here
        before any :meth:`step_pointgoal`.
        """
        intrinsic = np.asarray(intrinsic, dtype=float)
        if intrinsic.shape != (3, 3):
            raise ValueError(f"intrinsic must be a 3x3 matrix, got {intrinsic.shape}")
        payload = {
            "intrinsic": intrinsic.tolist(),
            "stop_threshold": float(stop_threshold),
            "batch_size": int(batch_size),
        }
        return self._request_json("POST", "/navigator_reset", payload=payload)

    def step_pointgoal(
        self,
        goal_xy: np.ndarray,
        rgb_bytes: bytes,
        depth_m: np.ndarray,
    ) -> np.ndarray:
        """Send one observation with a body-frame goal and return a trajectory.

        Parameters
        ----------
        goal_xy:
            Goal position ``(N, 2)`` in the **robot body frame** (x forward).
        rgb_bytes:
            JPEG/PNG-encoded RGB image bytes of the current egocentric view.
        depth_m:
            ``(H, W)`` float32 depth image in metres, aligned to the RGB view.

        Returns
        -------
        np.ndarray
            Selected trajectory ``(T, 3)`` of body-frame ``[x, y, yaw]``
            waypoints.
        """
        goal_xy = np.asarray(goal_xy, dtype=float)
        if goal_xy.ndim == 1:
            goal_xy = goal_xy.reshape(1, -1)
        if goal_xy.ndim != 2 or goal_xy.shape[1] < 2:
            raise ValueError(f"goal_xy must have shape (N, 2), got {goal_xy.shape}")
        goal_data = json.dumps(
            {"goal_x": goal_xy[:, 0].tolist(), "goal_y": goal_xy[:, 1].tolist()}
        )
        return self._step_goal("/pointgoal_step", goal_data, rgb_bytes, depth_m)

    def step_pixelgoal(
        self,
        pixel_xy: np.ndarray,
        rgb_bytes: bytes,
        depth_m: np.ndarray,
    ) -> np.ndarray:
        """Send one observation with a **pixel** goal and return a trajectory.

        The goal is a ``(u, v)`` pixel coordinate in the RGB view (0-based,
        origin top-left) — used by a vision model that points at the target.

        Returns
        -------
        np.ndarray
            Selected trajectory ``(T, 3)`` of body-frame waypoints.
        """
        pixel = np.asarray(pixel_xy).reshape(-1)[:2]
        if pixel.size < 2 or not np.all(np.isfinite(pixel)):
            raise ValueError(f"pixel_xy must contain two finite values, got {pixel}")
        # The NavDP server uses the pixel to index image slices, so it must be
        # sent as integers.
        goal_data = json.dumps(
            {"goal_x": [int(pixel[0])], "goal_y": [int(pixel[1])]}
        )
        return self._step_goal("/pixelgoal_step", goal_data, rgb_bytes, depth_m)

    def _step_goal(
        self,
        endpoint: str,
        goal_data: str,
        rgb_bytes: bytes,
        depth_m: np.ndarray,
    ) -> np.ndarray:
        """Shared multipart POST for pointgoal / pixelgoal style endpoints."""
        if not isinstance(rgb_bytes, (bytes, bytearray)) or not rgb_bytes:
            raise ValueError("rgb_bytes must contain encoded image bytes")

        depth_uint16 = _encode_depth_png(depth_m)
        body = _multipart_body(
            [
                ("image", "image.jpg", "image/jpeg", bytes(rgb_bytes)),
                ("depth", "depth.png", "image/png", depth_uint16),
                # Form field (no filename) — Flask must expose it in request.form.
                ("goal_data", "", "application/json", goal_data.encode("utf-8")),
            ]
        )
        boundary = body[0]
        result = self._request_json(
            "POST",
            endpoint,
            data=body[1],
            content_type=f"multipart/form-data; boundary={boundary}",
        )
        trajectory = result.get("trajectory")
        if trajectory is None:
            raise RuntimeError(f"NavDP server response has no 'trajectory': {str(result)[:200]}")
        return np.asarray(trajectory, dtype=float)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _request_json(
        self,
        method: str,
        path: str,
        *,
        payload: dict | None = None,
        data: bytes | None = None,
        content_type: str = "application/json",
        timeout: float | None = None,
    ) -> dict:
        if data is None:
            data = b"" if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib_request.Request(
            f"{self.base_url}{path}",
            data=data,
            method=method,
            headers={"Content-Type": content_type},
        )
        try:
            with self._opener.open(request, timeout=timeout or self.timeout) as response:
                body = response.read().decode("utf-8")
        except urllib_error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"NavDP server HTTP {exc.code} at {path}: {detail}") from exc
        except (urllib_error.URLError, TimeoutError, OSError) as exc:
            raise RuntimeError(
                f"Cannot reach NavDP server at {self.base_url}: {exc}"
            ) from exc
        try:
            result = json.loads(body)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"NavDP server returned invalid JSON: {body[:200]}") from exc
        if not isinstance(result, dict):
            raise RuntimeError("NavDP server response must be a JSON object")
        return result


# ---------------------------------------------------------------------------
# Encoding helpers
# ---------------------------------------------------------------------------


def _encode_depth_png(depth_m: np.ndarray) -> bytes:
    """Encode a metres depth image to a uint16 PNG (protocol: metres * 10000)."""
    depth_m = np.asarray(depth_m)
    if depth_m.ndim != 2 or depth_m.size == 0:
        raise ValueError(f"depth_m must be a 2-D image, got shape {depth_m.shape}")
    if not np.all(np.isfinite(depth_m)):
        depth_m = np.nan_to_num(depth_m, nan=0.0, posinf=0.0, neginf=0.0)
    depth_uint16 = np.clip(depth_m * 10000.0, 0.0, 65535.0).astype(np.uint16)
    ok, buf = cv2.imencode(".png", depth_uint16)
    if not ok:
        raise RuntimeError("cv2 failed to encode depth PNG")
    return buf.tobytes()


def _multipart_body(
    fields: list[tuple[str, str, str, bytes]],
) -> tuple[str, bytes]:
    """Build a multipart/form-data body.

    Returns ``(boundary, body)``. Each field is
    ``(name, filename, content_type, content)``.
    """
    boundary = "----NavDPClient" + uuid.uuid4().hex
    buf = io.BytesIO()
    crlf = b"\r\n"
    for name, filename, content_type, content in fields:
        buf.write(b"--" + boundary.encode("ascii") + crlf)
        disposition = f'Content-Disposition: form-data; name="{name}"'
        if filename:
            disposition += f'; filename="{filename}"'
        buf.write(disposition.encode("utf-8") + crlf)
        buf.write(f"Content-Type: {content_type}".encode("utf-8") + crlf + crlf)
        buf.write(content)
        buf.write(crlf)
    buf.write(b"--" + boundary.encode("ascii") + b"--" + crlf)
    return boundary, buf.getvalue()
