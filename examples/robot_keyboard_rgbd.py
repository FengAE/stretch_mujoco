#!/usr/bin/env python3
"""Keyboard teleoperation with live RGB-D display for Google Robot and TidyBot.

Examples
--------
  python examples/robot_keyboard_rgbd.py --robot google_robot
  python examples/robot_keyboard_rgbd.py --robot tidybot
  MUJOCO_GL=egl python examples/robot_keyboard_rgbd.py --robot tidybot --no-viewer

The depth streams are simulated from the same optical frames as their RGB
partners. TidyBot's wrist camera is also simulation-only.
"""

from __future__ import annotations

import argparse
import threading
import time
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from stretch_mujoco.robots import RobotType, create_simulator


@dataclass(frozen=True)
class ControlProfile:
    arm_keys: dict[str, tuple[Any, float]]
    gripper_open: tuple[tuple[Any, float], ...]
    gripper_close: tuple[tuple[Any, float], ...]


HELP = """
Controls
========
  W / S       forward / backward
  A / D       rotate left / right
  Q / E       strafe left / right

  1..7        arm joints positive
  Z..M        arm joints negative
  O / P       gripper open / close
  R           reset robot pose
  /           print actuator positions
  ESC         quit

Google Robot only:
  8 / ,       head pan positive / negative
  9 / .       head tilt positive / negative
"""


class KeyState:
    def __init__(self, keyboard_module: Any) -> None:
        self._keyboard = keyboard_module
        self._pressed: set[str] = set()
        self._lock = threading.Lock()
        self.quit_requested = threading.Event()

    def press(self, key: Any) -> None:
        if key == self._keyboard.Key.esc:
            self.quit_requested.set()
        elif isinstance(key, self._keyboard.KeyCode) and key.char:
            with self._lock:
                self._pressed.add(key.char.lower())

    def release(self, key: Any) -> None:
        if isinstance(key, self._keyboard.KeyCode) and key.char:
            with self._lock:
                self._pressed.discard(key.char.lower())

    def snapshot(self) -> set[str]:
        with self._lock:
            return set(self._pressed)


def build_profile(simulator: Any, robot_type: RobotType) -> ControlProfile:
    actuators = simulator.Actuators
    negative_keys = "zxcvbnm"
    if robot_type == RobotType.GOOGLE_ROBOT:
        arm = [
            actuators.joint_torso,
            actuators.joint_shoulder,
            actuators.joint_bicep,
            actuators.joint_elbow,
            actuators.joint_forearm,
            actuators.joint_wrist,
            actuators.joint_gripper,
        ]
        arm_keys = {
            key: (actuator, delta)
            for index, actuator in enumerate(arm)
            for key, delta in ((str(index + 1), 0.08), (negative_keys[index], -0.08))
        }
        arm_keys.update(
            {
                "8": (actuators.joint_head_pan, 0.08),
                ",": (actuators.joint_head_pan, -0.08),
                "9": (actuators.joint_head_tilt, 0.06),
                ".": (actuators.joint_head_tilt, -0.06),
            }
        )
        return ControlProfile(
            arm_keys=arm_keys,
            gripper_open=(
                (actuators.joint_finger_right, -0.06),
                (actuators.joint_finger_left, -0.06),
            ),
            gripper_close=(
                (actuators.joint_finger_right, 0.06),
                (actuators.joint_finger_left, 0.06),
            ),
        )

    arm = [getattr(actuators, f"joint_{index}") for index in range(1, 8)]
    arm_keys = {
        key: (actuator, delta)
        for index, actuator in enumerate(arm)
        for key, delta in ((str(index + 1), 0.08), (negative_keys[index], -0.08))
    }
    return ControlProfile(
        arm_keys=arm_keys,
        gripper_open=((actuators.fingers_actuator, -20.0),),
        gripper_close=((actuators.fingers_actuator, 20.0),),
    )


def reset_robot(simulator: Any, robot_type: RobotType) -> None:
    simulator.set_base_velocity(0.0, 0.0, 0.0)
    if robot_type == RobotType.TIDYBOT:
        simulator.home()
        return

    targets = {
        simulator.Actuators.joint_torso: 0.0,
        simulator.Actuators.joint_shoulder: 0.0,
        simulator.Actuators.joint_bicep: 0.0,
        simulator.Actuators.joint_elbow: 0.0,
        simulator.Actuators.joint_forearm: 0.0,
        simulator.Actuators.joint_wrist: 0.0,
        simulator.Actuators.joint_gripper: 0.0,
        simulator.Actuators.joint_finger_right: 0.01,
        simulator.Actuators.joint_finger_left: 0.01,
        simulator.Actuators.joint_head_pan: -0.00285961,
        simulator.Actuators.joint_head_tilt: 0.9351361,
    }
    for actuator, target in targets.items():
        simulator.move_to(actuator, target)


def print_status(simulator: Any) -> None:
    status = simulator.pull_status()
    x, y, heading = simulator.get_base_pose()
    print(f"\nbase: x={x:.3f}, y={y:.3f}, heading={heading:.3f}")
    for actuator in simulator.Actuators.all():
        if actuator.is_base_actuator():
            continue
        print(f"  {actuator.name:24s} {actuator.get_position(status): .4f}")


def apply_controls(
    simulator: Any,
    profile: ControlProfile,
    pressed: set[str],
    *,
    linear_speed: float,
    angular_speed: float,
) -> None:
    forward = linear_speed * (float("w" in pressed) - float("s" in pressed))
    lateral = linear_speed * (float("q" in pressed) - float("e" in pressed))
    angular = angular_speed * (float("a" in pressed) - float("d" in pressed))
    simulator.set_base_velocity(forward, angular, lateral)

    for key, (actuator, delta) in profile.arm_keys.items():
        if key in pressed:
            simulator.move_by(actuator, delta)
    if "o" in pressed:
        for actuator, delta in profile.gripper_open:
            simulator.move_by(actuator, delta)
    if "p" in pressed:
        for actuator, delta in profile.gripper_close:
            simulator.move_by(actuator, delta)


def colorize_depth(depth: np.ndarray, max_depth: float) -> np.ndarray:
    depth = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0.0)
    normalized = np.zeros(depth.shape, dtype=np.uint8)
    normalized[valid] = np.asarray(
        255.0 * (1.0 - np.clip(depth[valid], 0.0, max_depth) / max_depth),
        dtype=np.uint8,
    )
    colored = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    colored[~valid] = 0
    if np.any(valid):
        label = f"depth {depth[valid].min():.2f}-{depth[valid].max():.2f} m"
        cv2.putText(colored, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return colored


def rgbd_panels(simulator: Any, max_depth: float) -> dict[str, np.ndarray]:
    camera_data = simulator.pull_camera_data()
    available = camera_data.get_all(auto_correct_rgb=True, use_depth_color_map=False)
    panels: dict[str, np.ndarray] = {}
    depth_by_frame = {
        camera.camera_name_in_mjcf: available[camera]
        for camera in simulator.Cameras.depth()
        if camera in available
    }
    for camera in simulator.Cameras.rgb():
        if camera not in available:
            continue
        rgb = available[camera]
        depth = depth_by_frame.get(camera.camera_name_in_mjcf)
        if depth is None:
            panel = rgb
        else:
            depth_color = colorize_depth(depth, max_depth)
            if depth_color.shape[:2] != rgb.shape[:2]:
                depth_color = cv2.resize(depth_color, (rgb.shape[1], rgb.shape[0]))
            panel = np.concatenate((rgb, depth_color), axis=1)
        panels[camera.camera_name_in_mjcf] = panel
    return panels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--robot",
        choices=(RobotType.GOOGLE_ROBOT.value, RobotType.TIDYBOT.value),
        default=RobotType.GOOGLE_ROBOT.value,
    )
    parser.add_argument(
        "--no-viewer", action="store_true", help="Do not open the MuJoCo 3-D viewer"
    )
    parser.add_argument("--linear-speed", type=float, default=0.35, help="Base speed in m/s")
    parser.add_argument("--angular-speed", type=float, default=0.8, help="Turn speed in rad/s")
    parser.add_argument(
        "--depth-max", type=float, default=3.0, help="Depth color-map maximum in metres"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from pynput import keyboard

    robot_type = RobotType(args.robot)
    probe = create_simulator(robot_type)
    cameras = probe.Cameras.all()
    simulator = create_simulator(robot_type, cameras_to_use=cameras, camera_hz=20)
    profile = build_profile(simulator, robot_type)
    keys = KeyState(keyboard)
    listener = keyboard.Listener(on_press=keys.press, on_release=keys.release)

    print(HELP)
    print(f"Starting {robot_type.value} with cameras: {[camera.name for camera in cameras]}")
    simulator.start(
        headless=args.no_viewer,
        show_viewer_ui=True,
        use_passive_viewer=True,
    )
    listener.start()
    previous_pressed: set[str] = set()
    next_control_time = time.monotonic()

    try:
        while simulator.is_running() and not keys.quit_requested.is_set():
            pressed = keys.snapshot()
            newly_pressed = pressed - previous_pressed
            previous_pressed = pressed

            if "r" in newly_pressed:
                reset_robot(simulator, robot_type)
            if "/" in newly_pressed:
                print_status(simulator)

            now = time.monotonic()
            if now >= next_control_time:
                apply_controls(
                    simulator,
                    profile,
                    pressed,
                    linear_speed=args.linear_speed,
                    angular_speed=args.angular_speed,
                )
                next_control_time = now + 0.08

            for frame_name, panel in rgbd_panels(simulator, args.depth_max).items():
                window_name = f"{robot_type.value} | {frame_name} | RGB + depth"
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                cv2.imshow(window_name, panel)
            if cv2.waitKey(1) & 0xFF == 27:
                keys.quit_requested.set()
            time.sleep(0.005)
    finally:
        simulator.set_base_velocity(0.0, 0.0, 0.0)
        listener.stop()
        cv2.destroyAllWindows()
        simulator.stop()


if __name__ == "__main__":
    main()
