"""Save RGBD evidence around robot state-changing tool calls."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from itertools import count
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from strech_codex.world.state import get_robot_type, get_sim, has_sim

STATE_CHANGING_TOOLS = {
    "nav_execute_path",
    "robot_move_to",
    "robot_move_by",
    "robot_set_base_velocity",
    "robot_home",
    "robot_stow",
    "robot_attach_object",
    "robot_release_object",
}
_sequence = count(1)


def capture(tool: str, phase: str, invocation: int) -> list[dict[str, Any]]:
    root = os.environ.get("STRECH_CODEX_EPISODE_DIR")
    if not root or not has_sim():
        return []
    directory = Path(root) / "evidence" / f"{invocation:03d}_{_safe(tool)}" / phase
    directory.mkdir(parents=True, exist_ok=True)
    sim = get_sim()
    snapshot = sim.pull_camera_data()
    artifacts = []
    for camera in sim.Cameras.all():
        try:
            data = snapshot.get_camera_data(camera, auto_correct_rgb=True)
        except ValueError:
            continue
        name = _safe(camera.name)
        if camera.is_depth:
            raw_path = directory / f"{name}.npy"
            preview_path = directory / f"{name}.png"
            np.save(raw_path, data)
            cv2.imwrite(str(preview_path), _depth_preview(data))
            paths = {"raw": str(raw_path), "preview": str(preview_path)}
        else:
            image_path = directory / f"{name}.png"
            cv2.imwrite(str(image_path), data)
            paths = {"image": str(image_path)}
        artifacts.append(
            {
                "camera": camera.name,
                "kind": "depth" if camera.is_depth else "rgb",
                "shape": list(data.shape),
                "dtype": str(data.dtype),
                **paths,
            }
        )
    state: dict[str, Any] = {}
    for key, getter in {
        "base_pose": sim.get_base_pose,
        "ee_pose": sim.get_ee_pose,
        "status": lambda: sim.pull_status().to_dict(),
    }.items():
        try:
            state[key] = getter()
        except Exception as exc:
            state[key] = {"error": str(exc)}
    metadata = directory / "metadata.json"
    metadata.write_text(
        json.dumps(
            {
                "tool": tool,
                "phase": phase,
                "invocation": invocation,
                "captured_at": datetime.now().astimezone().isoformat(timespec="milliseconds"),
                "robot_type": get_robot_type(),
                "state": state,
                "cameras": artifacts,
            },
            ensure_ascii=False,
            indent=2,
            default=lambda value: value.tolist() if hasattr(value, "tolist") else str(value),
        )
        + "\n",
        encoding="utf-8",
    )
    return artifacts


def next_invocation() -> int:
    return next(_sequence)


def _safe(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "item"


def _depth_preview(data: np.ndarray) -> np.ndarray:
    finite = np.isfinite(data)
    if not finite.any():
        return np.zeros((*data.shape[:2], 3), dtype=np.uint8)
    low, high = np.percentile(data[finite], (2, 98))
    scaled = np.clip((data - low) / max(float(high - low), 1e-6), 0, 1)
    return cv2.applyColorMap((scaled * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
