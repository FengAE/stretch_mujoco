"""Keyboard-driven test for the Google Robot model from MuJoCo Menagerie.

Verify that the model loads and its mobile-base, arm, and gripper actuators respond.
"""

from __future__ import annotations

import time
from pathlib import Path
from threading import Event, Lock

import mujoco
import mujoco.viewer

from stretch_mujoco.keyboard_control import (
    change_joint_target,
    drive_planar_base,
    reset_to_home,
    step_for_period,
)

MODELS_DIR = Path(__file__).resolve().parents[1] / "stretch_mujoco" / "models"
ROBOT_XML = MODELS_DIR / "assets" / "google_robot" / "scene.xml"

KEY_MAP: dict[str, tuple[str, float]] = {
    # Arm joints: number row = +, bottom row = -
    "1": ("joint_torso", 0.15),
    "z": ("joint_torso", -0.15),
    "2": ("joint_shoulder", 0.15),
    "x": ("joint_shoulder", -0.15),
    "3": ("joint_bicep", 0.15),
    "c": ("joint_bicep", -0.15),
    "4": ("joint_elbow", 0.15),
    "v": ("joint_elbow", -0.15),
    "5": ("joint_forearm", 0.15),
    "b": ("joint_forearm", -0.15),
    "6": ("joint_wrist", 0.15),
    "n": ("joint_wrist", -0.15),
    "7": ("joint_gripper", 0.15),
    "m": ("joint_gripper", -0.15),
    # Fingers
    "8": ("joint_finger_right", 0.08),
    ",": ("joint_finger_right", -0.08),
    "9": ("joint_finger_left", 0.08),
    ".": ("joint_finger_left", -0.08),
}

HELP = """
╔══════════════════════════════════════════════════════════════════════╗
║              Google Robot — Keyboard Controls                       ║
╠══════════════════════════════════════════════════════════════════════╣
║  W / S  : Move forward / backward    A / D : Turn left / right     ║
║                                                                      ║
║  ARM JOINTS                          FINGERS                        ║
║    1 / Z  : Torso         (±)        8 / ,  : Right finger  (±)    ║
║    2 / X  : Shoulder      (±)        9 / .  : Left finger   (±)    ║
║    3 / C  : Bicep         (±)                                       ║
║    4 / V  : Elbow         (±)                                       ║
║    5 / B  : Forearm       (±)                                       ║
║    6 / N  : Wrist         (±)                                       ║
║    7 / M  : Gripper Rot   (±)                                       ║
╠══════════════════════════════════════════════════════════════════════╣
║  P       : Print current joint positions                           ║
║  R       : Reset all joints to neutral                             ║
║  ESC     : Quit                                                    ║
╚══════════════════════════════════════════════════════════════════════╝
"""


def print_help() -> None:
    print(HELP)


def load_model() -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Load Google Robot from the test scene XML file."""
    model = mujoco.MjModel.from_xml_path(str(ROBOT_XML))
    data = mujoco.MjData(model)
    reset_to_home(model, data)
    return model, data


def main() -> None:
    from pynput import keyboard as pynput_keyboard

    model, data = load_model()

    print(f"Loaded: {model.names}")
    print(f"  Bodies: {model.nbody}")
    print(f"  Joints: {model.njnt}")
    print(f"  Actuators: {model.nu}")
    print(f"  Actuator names: {[model.actuator(i).name for i in range(model.nu)]}")
    print_help()

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.azimuth = -120
        viewer.cam.elevation = -25
        viewer.cam.distance = 3.0

        key_buffer: set[str] = set()
        key_lock = Lock()
        quit_requested = Event()

        def on_press(key: pynput_keyboard.Key | pynput_keyboard.KeyCode) -> None:
            if key == pynput_keyboard.Key.esc:
                quit_requested.set()
                return
            if isinstance(key, pynput_keyboard.KeyCode) and key.char is not None:
                with key_lock:
                    key_buffer.add(key.char.lower())

        def on_release(key: pynput_keyboard.Key | pynput_keyboard.KeyCode) -> None:
            if isinstance(key, pynput_keyboard.KeyCode) and key.char is not None:
                with key_lock:
                    key_buffer.discard(key.char.lower())

        listener = pynput_keyboard.Listener(on_press=on_press, on_release=on_release)
        listener.start()

        prev_time = time.perf_counter()

        try:
            while viewer.is_running() and not quit_requested.is_set():
                # Process keys at ~50 Hz
                now = time.perf_counter()
                if now - prev_time < 0.02:
                    time.sleep(0.001)
                    continue
                prev_time = now

                with key_lock:
                    active_keys = tuple(key_buffer)
                for k in active_keys:
                    if k == "p":
                        print("\n--- Joint Positions ---")
                        for joint_name in (
                            "base_x",
                            "base_y",
                            "base_theta",
                            *[v[0] for v in KEY_MAP.values()],
                        ):
                            print(f"  {joint_name}: {float(data.joint(joint_name).qpos[0]):.4f}")
                        with key_lock:
                            key_buffer.discard(k)

                    elif k == "r":
                        print("\n[RESET] Restoring neutral joint positions")
                        reset_to_home(model, data)
                        with key_lock:
                            key_buffer.discard(k)

                    elif k in {"w", "a", "s", "d"}:
                        drive_planar_base(
                            model,
                            data,
                            k,
                            x_joint="base_x",
                            y_joint="base_y",
                            heading_joint="base_theta",
                        )
                    elif k in KEY_MAP:
                        joint_name, delta = KEY_MAP[k]
                        change_joint_target(model, data, joint_name, delta)

                # Step the simulation
                step_for_period(model, data, 0.02)
                viewer.sync()

        finally:
            listener.stop()


if __name__ == "__main__":
    main()
