"""Shared Stretch 3 observation/action contract for OpenPI training and inference.

Collect and execute at 10 Hz with the head fixed at ``HEAD_POSE``. Actions are
relative increments except for the gripper, which is an absolute position in
the real Stretch range. Normalization belongs to OpenPI, not this module.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from stretch_mujoco.enums.stretch_cameras import StretchCameras
from stretch_mujoco.robots.stretch3.config import GRIPPER_MIN_MAX


FPS = 10
IMAGE_SIZE = (224, 224)
HEAD_POSE = {"head_pan": -1.57, "head_tilt": -0.65}

STATE_NAMES = (
    "base_linear_velocity_mps",
    "base_angular_velocity_radps",
    "lift_position_m",
    "arm_position_m",
    "wrist_yaw_position_rad",
    "wrist_pitch_position_rad",
    "wrist_roll_position_rad",
    "gripper_position",
)

ACTION_NAMES = (
    "base_translate_delta_m",
    "base_rotate_delta_rad",
    "lift_delta_m",
    "arm_delta_m",
    "wrist_yaw_delta_rad",
    "wrist_pitch_delta_rad",
    "wrist_roll_delta_rad",
    "gripper_position",
)

_KEY_DELTAS = {
    "w": (0, 0.07),
    "s": (0, -0.07),
    "a": (1, 0.15),
    "d": (1, -0.15),
    "i": (2, 0.1),
    "k": (2, -0.1),
    "j": (3, -0.05),
    "l": (3, 0.05),
    "o": (4, 0.2),
    "p": (4, -0.2),
    "c": (5, 0.2),
    "v": (5, -0.2),
    "e": (6, 0.2),
    "r": (6, -0.2),
    "n": (7, 0.07),
    "m": (7, -0.07),
}


def state_from_status(status) -> np.ndarray:
    """Return the canonical 8-D state from simulator or hardware status."""
    return np.asarray(
        [
            status.base.x_vel,
            status.base.theta_vel,
            status.lift.pos,
            status.arm.pos,
            status.wrist_yaw.pos,
            status.wrist_pitch.pos,
            status.wrist_roll.pos,
            status.gripper.pos,
        ],
        dtype=np.float32,
    )


def make_observation(
    *, state: np.ndarray, head_rgb: np.ndarray, wrist_rgb: np.ndarray, prompt: str
) -> dict:
    """Build the exact dictionary consumed by the Stretch OpenPI policy."""
    state = np.asarray(state, dtype=np.float32)
    if state.shape != (len(STATE_NAMES),) or not np.isfinite(state).all():
        raise ValueError(
            f"state must be finite with shape ({len(STATE_NAMES)},), got {state.shape}"
        )
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")

    return {
        "observation/image": _resize_rgb(head_rgb),
        "observation/wrist_image": _resize_rgb(wrist_rgb),
        "observation/state": state,
        "prompt": prompt.strip(),
    }


def observation_from_simulator(sim, prompt: str) -> dict:
    """Read one canonical observation from a running Stretch simulator."""
    cameras = sim.pull_camera_data()
    return make_observation(
        state=state_from_status(sim.pull_status()),
        head_rgb=cameras.get_camera_data(
            StretchCameras.cam_d435i_rgb, auto_rotate=True, auto_correct_rgb=False
        ),
        wrist_rgb=cameras.get_camera_data(
            StretchCameras.cam_d405_rgb, auto_rotate=True, auto_correct_rgb=False
        ),
        prompt=prompt,
    )


def validate_action(action: np.ndarray) -> np.ndarray:
    """Validate and return one canonical 8-D action as float32."""
    action = np.asarray(action, dtype=np.float32)
    if action.shape != (len(ACTION_NAMES),) or not np.isfinite(action).all():
        raise ValueError(
            f"action must be finite with shape ({len(ACTION_NAMES)},), got {action.shape}"
        )
    return action


def action_from_keys(keys, gripper_position: float) -> np.ndarray:
    """Convert currently held teleop keys to one canonical action."""
    action = np.zeros(len(ACTION_NAMES), dtype=np.float64)
    action[-1] = gripper_position
    for key in keys:
        if key in _KEY_DELTAS:
            index, delta = _KEY_DELTAS[key]
            action[index] += delta
    action[-1] = np.clip(action[-1], *GRIPPER_MIN_MAX)
    return action.astype(np.float32)


def base_action_deltas(prev: tuple[float, float], following: tuple[float, float]) -> np.ndarray:
    """Return the 2-D base action delta ``[base_translate_delta, base_rotate_delta]``
    between two planar base poses ``(x, theta)``. The angular delta is wrap-aware."""
    x0, theta0 = prev
    x1, theta1 = following
    delta_theta = (theta1 - theta0 + np.pi) % (2 * np.pi) - np.pi
    return np.asarray([x1 - x0, delta_theta], dtype=np.float32)


def to_lerobot_frame(observation: dict, action: np.ndarray) -> dict:
    """Map the runtime observation keys to one LeRobot training frame."""
    return {
        "image": observation["observation/image"],
        "wrist_image": observation["observation/wrist_image"],
        "state": observation["observation/state"],
        "actions": validate_action(action),
        "task": observation["prompt"],
    }


def grasp_diagnostics(metrics: dict, *, initial_object_position: np.ndarray | None = None) -> dict:
    """Return JSON/NPZ-safe grasp metrics for offline diagnosis, not model input."""
    object_position = np.asarray(metrics["object_position"], dtype=float)
    grasp_center = np.asarray(metrics["grasp_center_position"], dtype=float)
    if object_position.shape != (3,) or grasp_center.shape != (3,):
        raise ValueError("grasp positions must have shape (3,)")
    initial_height = (
        float(object_position[2])
        if initial_object_position is None
        else float(np.asarray(initial_object_position, dtype=float)[2])
    )
    return {
        "object_position": object_position.tolist(),
        "grasp_center_position": grasp_center.tolist(),
        "center_distance_m": float(metrics["center_distance_m"]),
        "left_finger_contacts": int(metrics.get("left_finger_contacts", 0)),
        "right_finger_contacts": int(metrics.get("right_finger_contacts", 0)),
        "bilateral_contact": bool(metrics.get("bilateral_contact", False)),
        "object_height_m": float(object_position[2]),
        "object_lift_m": float(object_position[2] - initial_height),
    }


def save_episode(
    path: str | Path,
    frames: list[dict],
    timestamps: list[float],
    *,
    diagnostics: list[dict] | None = None,
    metadata: dict | None = None,
) -> Path:
    """Atomically save raw frames for later LeRobot conversion."""
    if not frames or len(frames) != len(timestamps):
        raise ValueError("frames and timestamps must have the same non-zero length")
    prompts = {frame["task"] for frame in frames}
    if len(prompts) != 1:
        raise ValueError("all frames in an episode must use the same task prompt")
    actions = np.stack([frame["actions"] for frame in frames])
    if not np.any(np.abs(actions[:, :-1]) > 1e-6) and np.ptp(actions[:, -1]) <= 1e-6:
        raise ValueError("episode contains no commanded motion")
    if diagnostics is not None and len(diagnostics) != len(frames):
        raise ValueError("diagnostics must have one entry per frame")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    payload = {
        "image": np.stack([frame["image"] for frame in frames]),
        "wrist_image": np.stack([frame["wrist_image"] for frame in frames]),
        "state": np.stack([frame["state"] for frame in frames]),
        "actions": actions,
        "timestamps": np.asarray(timestamps, dtype=np.float64),
        "task": np.asarray(prompts.pop()),
        "fps": np.asarray(FPS, dtype=np.int32),
        "state_names": np.asarray(STATE_NAMES),
        "action_names": np.asarray(ACTION_NAMES),
        "format_version": np.asarray(1, dtype=np.int32),
    }
    if diagnostics:
        keys = diagnostics[0].keys()
        if any(item.keys() != keys for item in diagnostics):
            raise ValueError("all diagnostic frames must have the same fields")
        payload.update(
            {f"diagnostics/{key}": np.asarray([item[key] for item in diagnostics]) for key in keys}
        )
    if metadata is not None:
        payload["episode_metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True))
    try:
        with temporary.open("wb") as output:
            np.savez_compressed(output, **payload)
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return path


def _resize_rgb(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"RGB image must have shape (height, width, 3), got {image.shape}")
    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    elif image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    target_h, target_w = IMAGE_SIZE
    height, width = image.shape[:2]
    scale = min(target_h / height, target_w / width)
    resized_h, resized_w = max(1, round(height * scale)), max(1, round(width * scale))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (resized_w, resized_h), interpolation=interpolation)
    output = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    y = (target_h - resized_h) // 2
    x = (target_w - resized_w) // 2
    output[y : y + resized_h, x : x + resized_w] = resized
    return output
