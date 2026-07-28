"""Wire protocol for the GraspGen RGB-D ZMQ service."""

from __future__ import annotations

from typing import Any
import uuid
import zlib

import cv2
import msgpack
import numpy as np


PROTOCOL = "graspgen.rgbd.v1"


def control_request(action: str) -> list[bytes]:
    return [
        msgpack.packb(
            {
                "protocol": PROTOCOL,
                "action": action,
                "type": "control",
                "command": action,
            },
            use_bin_type=True,
        )
    ]


def encode_rgbd_request(
    rgb: np.ndarray,
    depth_metres: np.ndarray,
    camera_k: np.ndarray,
    text_prompt: str,
    *,
    output_frame: str = "camera",
    camera_pose: np.ndarray | None = None,
    inference: dict[str, Any] | None = None,
    jpeg_quality: int = 90,
) -> list[bytes]:
    rgb = np.asarray(rgb)
    depth = np.asarray(depth_metres, dtype=np.float32)
    camera_k = np.asarray(camera_k, dtype=np.float32)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 RGB image, got {rgb.shape}")
    if depth.shape != rgb.shape[:2]:
        raise ValueError(f"RGB and depth shapes differ: {rgb.shape[:2]} != {depth.shape}")
    if camera_k.shape != (3, 3):
        raise ValueError(f"Expected a 3x3 camera matrix, got {camera_k.shape}")
    if output_frame not in {"camera", "base"}:
        raise ValueError(f"Unsupported output frame: {output_frame}")
    if output_frame == "base" and camera_pose is None:
        raise ValueError("camera_pose is required for base-frame output")

    bgr = cv2.cvtColor(rgb.astype(np.uint8, copy=False), cv2.COLOR_RGB2BGR)
    encoded, buffer = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
    if not encoded:
        raise ValueError("Failed to JPEG-encode RGB image")
    depth_mm = np.clip(
        np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0) * 1000.0,
        0.0,
        np.iinfo(np.uint16).max,
    ).astype(np.uint16)
    header: dict[str, Any] = {
        "protocol": PROTOCOL,
        "action": "infer_rgbd",
        "request_id": str(uuid.uuid4()),
        "text_prompt": text_prompt,
        "output_frame": output_frame,
        "camera_K": camera_k.tolist(),
        "rgb": {
            "encoding": "jpeg",
            "shape": list(rgb.shape),
            "quality": int(jpeg_quality),
        },
        "depth": {
            "encoding": "zlib_uint16_mm",
            "shape": list(depth_mm.shape),
            "scale_metres": 0.001,
            "invalid_value": 0,
        },
        "inference": inference or {},
    }
    if camera_pose is not None:
        pose = np.asarray(camera_pose, dtype=np.float32)
        if pose.shape != (4, 4):
            raise ValueError(f"Expected a 4x4 camera pose, got {pose.shape}")
        header["camera_pose"] = pose.tolist()
    return [
        msgpack.packb(header, use_bin_type=True),
        buffer.tobytes(),
        zlib.compress(np.ascontiguousarray(depth_mm).tobytes()),
    ]
