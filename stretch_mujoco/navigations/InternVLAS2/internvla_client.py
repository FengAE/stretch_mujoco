"""HTTP client for the InternVLA-N1 dual-system server.

Talks to ``http_internvla_server.py`` (port 5801 by default) over
``POST /eval_dual``: send one RGB-D frame (plus an optional ``reset`` flag and
an optional language ``instruction``) and receive either a ``discrete_action``
(S2 output) or a ``trajectory`` (S1 diffusion output) plus an optional
``pixel_goal``.

The trajectory returned by the server is a body-frame ``(N, 2)`` waypoint list
(see ``traj_to_actions``); the caller transforms it to the world frame.

Reuses the HTTP/multipart plumbing of :class:`NavDPClient`.
"""

from __future__ import annotations

import json
from typing import Optional

import numpy as np

from stretch_mujoco.navigations.NavDP.navdp_client import NavDPClient


class InternVLAClient(NavDPClient):
    """Minimal HTTP client for the InternVLA-N1 dual-system ``/eval_dual`` API."""

    def __init__(self, base_url: str = "http://127.0.0.1:5801", timeout: float = 120.0) -> None:
        super().__init__(base_url=base_url, timeout=timeout)

    def step(
        self,
        rgb_bytes: bytes,
        depth_m: np.ndarray,
        *,
        reset: bool = False,
        instruction: Optional[str] = None,
    ) -> dict:
        """Send one observation and return the parsed server response.

        Parameters
        ----------
        rgb_bytes:
            JPEG/PNG-encoded RGB image bytes.
        depth_m:
            ``(H, W)`` float32 depth in metres, aligned to the RGB view.
        reset:
            Reset the server-side policy (first call of an episode).
        instruction:
            Optional natural-language instruction; the server falls back to its
            built-in instruction when omitted.

        Returns
        -------
        dict
            ``{"discrete_action": [...]}`` or ``{"trajectory": [[x, y], ...],
            "pixel_goal": [u, v]?}``.
        """
        if not isinstance(rgb_bytes, (bytes, bytearray)) or not rgb_bytes:
            raise ValueError("rgb_bytes must contain encoded image bytes")

        payload: dict = {"reset": bool(reset)}
        if instruction is not None and str(instruction).strip():
            payload["instruction"] = str(instruction).strip()

        from stretch_mujoco.navigations.NavDP.navdp_client import _encode_depth_png, _multipart_body

        depth_uint16 = _encode_depth_png(depth_m)
        body = _multipart_body(
            [
                ("image", "image.jpg", "image/jpeg", bytes(rgb_bytes)),
                ("depth", "depth.png", "image/png", depth_uint16),
                # Form field (no filename) — Flask exposes it in request.form.
                ("json", "", "application/json", json.dumps(payload).encode("utf-8")),
            ]
        )
        boundary = body[0]
        return self._request_json(
            "POST",
            "/eval_dual",
            data=body[1],
            content_type=f"multipart/form-data; boundary={boundary}",
        )
