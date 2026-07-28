"""Keyboard-driven test for the Stanford TidyBot model from MuJoCo Menagerie.

Verify that the model loads and all 11 actuators respond correctly.
The TidyBot is composed of a custom mobile base, Kinova Gen3 7-DoF arm,
and Robotiq 2F-85 gripper.
"""

from __future__ import annotations

import time
from pathlib import Path
from threading import Event, Lock

import mujoco
import mujoco.viewer
import numpy as np

from stretch_mujoco.keyboard_control import (
    change_joint_target,
    drive_planar_base,
    reset_to_home,
    step_for_period,
)

MODELS_DIR = Path(__file__).resolve().parents[1] / "stretch_mujoco" / "models"
ROBOT_XML = MODELS_DIR / "assets" / "stanford_tidybot" / "scene.xml"

KEY_MAP: dict[str, tuple[str, float]] = {
    # --- Gen3 Arm: number keys = +, lower-row = - ---
    "1": ("joint_1", 0.15),
    "z": ("joint_1", -0.15),
    "2": ("joint_2", 0.15),
    "x": ("joint_2", -0.15),
    "3": ("joint_3", 0.15),
    "c": ("joint_3", -0.15),
    "4": ("joint_4", 0.15),
    "v": ("joint_4", -0.15),
    "5": ("joint_5", 0.15),
    "b": ("joint_5", -0.15),
    "6": ("joint_6", 0.15),
    "n": ("joint_6", -0.15),
    "7": ("joint_7", 0.15),
    "m": ("joint_7", -0.15),
    # --- 2F-85 Gripper (0=open, 255=closed) ---
}

HELP = """
╔══════════════════════════════════════════════════════════════════════╗
║           Stanford TidyBot — Keyboard Controls                      ║
╠══════════════════════════════════════════════════════════════════════╣
║  MOBILE BASE                       ARM (Gen3 7-DoF)                 ║
║    W / S  : Forward / backward      1…7  : joint +                  ║
║    A / D  : Turn left / right       Z…M  : joint -                  ║
║                                                                      ║
║  GRIPPER (2F-85)                  SYSTEM                             ║
║    O / P  : Open / Close           /    : Print joint positions     ║
║                                    R    : Reset to neutral          ║
║                                    ESC  : Quit                      ║
╚══════════════════════════════════════════════════════════════════════╝
"""


def print_help() -> None:
    print(HELP)


def load_model() -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Load TidyBot from the test scene XML file."""
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
    for i in range(model.nu):
        name = model.actuator(i).name
        ctrl_range = model.actuator_ctrlrange[i]
        print(f"  [{i}] {name}: ctrlrange=({ctrl_range[0]:.3f}, {ctrl_range[1]:.3f})")
    print_help()

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.azimuth = -120
        viewer.cam.elevation = -30
        viewer.cam.distance = 4.0

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
                now = time.perf_counter()
                if now - prev_time < 0.02:
                    time.sleep(0.001)
                    continue
                prev_time = now

                with key_lock:
                    active_keys = tuple(key_buffer)
                for k in active_keys:
                    if k == "/":
                        print("\n--- Joint Positions ---")
                        for i in range(model.nu):
                            name = model.actuator(i).name
                            joint_id = model.actuator_jntid(i)
                            if joint_id >= 0:
                                pos = data.joint(joint_id).qpos[0]
                                print(f"  {name}: {pos:.4f}")
                            else:
                                # tendon actuator
                                print(f"  {name}: ctrl={data.ctrl[i]:.1f}")
                        with key_lock:
                            key_buffer.discard(k)

                    elif k == "r":
                        print("\n[RESET] Restoring neutral positions")
                        reset_to_home(model, data)
                        with key_lock:
                            key_buffer.discard(k)

                    elif k in {"w", "a", "s", "d"}:
                        drive_planar_base(
                            model,
                            data,
                            k,
                            x_joint="joint_x",
                            y_joint="joint_y",
                            heading_joint="joint_th",
                        )
                    elif k in {"o", "p"}:
                        actuator_id = mujoco.mj_name2id(
                            model, mujoco.mjtObj.mjOBJ_ACTUATOR, "fingers_actuator"
                        )
                        current = float(data.ctrl[actuator_id])
                        delta = -25.0 if k == "o" else 25.0
                        target = float(
                            np.clip(
                                current + delta,
                                *model.actuator_ctrlrange[actuator_id],
                            )
                        )
                        data.ctrl[actuator_id] = target

                    elif k in KEY_MAP:
                        joint_name, delta = KEY_MAP[k]
                        change_joint_target(model, data, joint_name, delta)

                step_for_period(model, data, 0.02)
                viewer.sync()

        finally:
            listener.stop()


if __name__ == "__main__":
    main()
