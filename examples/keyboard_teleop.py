from datetime import datetime
from pathlib import Path
from pprint import pprint
from time import monotonic, sleep

import click
from pynput import keyboard

from examples.camera_feeds import show_camera_feeds_sync
from examples.laser_scan import show_laser_scan
from stretch_mujoco import StretchMujocoSimulator
from stretch_mujoco.enums.actuators import Actuators
from stretch_mujoco.enums.stretch_cameras import StretchCameras
from stretch_mujoco.enums.stretch_sensors import StretchSensors
from stretch_mujoco.openpi_contract import (
    FPS,
    HEAD_POSE,
    action_from_keys,
    observation_from_simulator,
    save_episode,
    to_lerobot_frame,
)


def print_keyboard_options(recording: bool = False):
    click.secho("\n       Keyboard Controls:", fg="yellow")
    click.secho("=====================================", fg="yellow")
    print("W / A / S / D: Move BASE")
    print("T / F / G / H: Move HEAD" if not recording else "HEAD: Fixed while recording")
    print("I / J / K / L: Move LIFT & ARM")
    print("O / P: Move WRIST YAW")
    print("C / V: Move WRIST PITCH")
    print("E / R: Move WRIST ROLL")
    print("N / M: Open & Close GRIPPER")
    print("ctrl + shift + ): Enable keyboard input")
    print("ctrl + shift + (: Disable keyboard input")
    print("Z : Print status")
    print("Q : Save episode and stop" if recording else "Q : Stop")
    click.secho("=====================================", fg="yellow")


def keyboard_control(key: str | None, sim: StretchMujocoSimulator):
    # mobile base
    if key == "w":
        sim.move_by(Actuators.base_translate, 0.07)
    elif key == "s":
        sim.move_by(Actuators.base_translate, -0.07)
    elif key == "a":
        sim.move_by(Actuators.base_rotate, 0.15)
    elif key == "d":
        sim.move_by(Actuators.base_rotate, -0.15)

    # head
    elif key == "t":
        sim.move_by(Actuators.head_tilt, 0.2)
    elif key == "f":
        sim.move_by(Actuators.head_pan, 0.2)
    elif key == "g":
        sim.move_by(Actuators.head_tilt, -0.2)
    elif key == "h":
        sim.move_by(Actuators.head_pan, -0.2)

    # arm
    elif key == "i":
        sim.move_by(Actuators.lift, 0.1)
    elif key == "k":
        sim.move_by(Actuators.lift, -0.1)
    elif key == "j":
        sim.move_by(Actuators.arm, -0.05)
    elif key == "l":
        sim.move_by(Actuators.arm, 0.05)

    # wrist
    elif key == "o":
        sim.move_by(Actuators.wrist_yaw, 0.2)
    elif key == "p":
        sim.move_by(Actuators.wrist_yaw, -0.2)
    elif key == "c":
        sim.move_by(Actuators.wrist_pitch, 0.2)
    elif key == "v":
        sim.move_by(Actuators.wrist_pitch, -0.2)
    elif key == "e":
        sim.move_by(Actuators.wrist_roll, 0.2)
    elif key == "r":
        sim.move_by(Actuators.wrist_roll, -0.2)

    # gripper
    elif key == "n":
        sim.move_by(Actuators.gripper, 0.07)
    elif key == "m":
        sim.move_by(Actuators.gripper, -0.07)

    # other
    elif key == "z":
        pprint(sim.pull_status())
    elif key == "q":
        sim.stop()


def keyboard_control_release(key: str | None, sim: StretchMujocoSimulator):
    if key in ("w", "s", "a", "d"):
        sim.set_base_velocity(0, 0)


# Allow multiple key-presses, references https://stackoverflow.com/a/74910695
key_buffer = []
active = True  # when false, disable kbd commands
ctrl_pressed = False


def on_press(key):
    global key_buffer, active, ctrl_pressed
    if key == keyboard.Key.ctrl:
        ctrl_pressed = True
    if ctrl_pressed and str(key) == "'('":
        active = False
        print("Disabling keyboard")
    if ctrl_pressed and str(key) == "')'":
        active = True
        print("Enabling keyboard")
    if not active:
        return
    if isinstance(key, keyboard.KeyCode) and key.char is not None:
        key = keyboard.KeyCode.from_char(key.char.lower())
    if key not in key_buffer and len(key_buffer) < 3:
        key_buffer.append(key)


def on_release(key, sim: StretchMujocoSimulator):
    global key_buffer, active, ctrl_pressed
    if key == keyboard.Key.ctrl:
        ctrl_pressed = False
    if isinstance(key, keyboard.KeyCode) and key.char is not None:
        key = keyboard.KeyCode.from_char(key.char.lower())
    if key in key_buffer:
        key_buffer.remove(key)
    if not active:
        return
    if isinstance(key, keyboard.KeyCode):
        keyboard_control_release(key.char, sim)


@click.command()
@click.option("--scene-xml-path", type=str, default=None, help="Path to the scene xml file")
@click.option("--robocasa-env", is_flag=True, help="Use robocasa environment")
@click.option("--imagery-nav", is_flag=True, help="Show only the Navigation camera")
@click.option("--imagery", is_flag=True, help="Show all the cameras' imagery")
@click.option("--lidar", is_flag=True, help="Show the lidar scan in Matplotlib")
@click.option("--print-ratio", is_flag=True, help="Print the sim-to-real time ratio to the cli.")
@click.option(
    "--record-dir",
    type=click.Path(path_type=Path, file_okay=False),
    default=None,
    help="Save one OpenPI episode in this directory.",
)
@click.option("--prompt", type=str, default=None, help="Language task for the recorded episode.")
def main(
    scene_xml_path: str | None,
    robocasa_env: bool,
    imagery_nav: bool,
    imagery: bool,
    lidar: bool,
    print_ratio: bool,
    record_dir: Path | None,
    prompt: str | None,
):
    recording = record_dir is not None
    if recording and (prompt is None or not prompt.strip()):
        raise click.UsageError("--prompt is required with --record-dir")

    cameras_to_use = StretchCameras.all() if imagery else []
    if imagery_nav:
        cameras_to_use = [StretchCameras.cam_nav_rgb]
        imagery = True
    use_imagery = imagery or imagery_nav
    if recording:
        for camera in (StretchCameras.cam_d435i_rgb, StretchCameras.cam_d405_rgb):
            if camera not in cameras_to_use:
                cameras_to_use.append(camera)

    model = None

    if robocasa_env:
        from stretch_mujoco.robocasa_gen import model_generation_wizard

        model, xml, objects_info = model_generation_wizard()

    sim = StretchMujocoSimulator(
        model=model,
        scene_xml_path=scene_xml_path,
        camera_hz=FPS if recording else 30,
        cameras_to_use=cameras_to_use,
    )

    listener = None
    # ponytail: keep one short demonstration in RAM; stream shards if episodes grow past a few minutes.
    frames = []
    timestamps = []
    try:
        sim.start()

        if recording:
            sim.move_to(Actuators.head_pan, HEAD_POSE["head_pan"])
            sim.wait_until_at_setpoint(Actuators.head_pan, position_tolerance=0.01)
            sim.move_to(Actuators.head_tilt, HEAD_POSE["head_tilt"])
            sim.wait_until_at_setpoint(Actuators.head_tilt, position_tolerance=0.01)
            camera_time = sim.pull_camera_data().time
            deadline = monotonic() + 2.0
            while sim.pull_camera_data().time <= camera_time:
                if monotonic() >= deadline:
                    raise TimeoutError("cameras did not refresh after entering manipulation mode")
                sleep(0.01)

        print_keyboard_options(recording)

        listener = keyboard.Listener(on_press=on_press, on_release=lambda key: on_release(key, sim))

        listener.start()
        key_buffer.clear()
        started_at = monotonic()
        next_sample = started_at
        gripper_target = float(sim.pull_status().gripper.pos)

        while sim.is_running():
            if recording:
                now = monotonic()
                if now < next_sample:
                    sleep(min(next_sample - now, 0.01))
                    continue

                keys = {
                    key.char
                    for key in tuple(key_buffer)
                    if isinstance(key, keyboard.KeyCode) and key.char is not None
                }
                if "q" in keys:
                    break
                if "z" in keys:
                    pprint(sim.pull_status())

                action = action_from_keys(keys, gripper_target)
                gripper_target = float(action[-1])
                observation = observation_from_simulator(sim, prompt or "")
                frames.append(to_lerobot_frame(observation, action))
                timestamps.append(now - started_at)
                _execute_recorded_action(sim, action)

                next_sample += 1.0 / FPS
                if next_sample < now:
                    next_sample = now + 1.0 / FPS
                continue

            for key in key_buffer:
                if isinstance(key, keyboard.KeyCode):
                    keyboard_control(key.char, sim)

            if not lidar and not use_imagery:
                sleep(0.05)

            if print_ratio:
                print(f"{sim.pull_status().sim_to_real_time_ratio_msg}")

            if use_imagery:
                show_camera_feeds_sync(sim, False)

            if lidar:
                sensor_data = sim.pull_sensor_data()

                try:
                    show_laser_scan(scan_data=sensor_data.get_data(StretchSensors.base_lidar))
                except:
                    ...

    except KeyboardInterrupt:
        pass
    finally:
        if listener is not None:
            listener.stop()
        if sim.is_running():
            sim.stop()
        if recording and frames:
            episode = record_dir / f"episode_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.npz"
            save_episode(episode, frames, timestamps)
            click.secho(f"Saved {len(frames)} frames to {episode}", fg="green")


def _execute_recorded_action(sim: StretchMujocoSimulator, action) -> None:
    actuators = (
        Actuators.base_translate,
        Actuators.base_rotate,
        Actuators.lift,
        Actuators.arm,
        Actuators.wrist_yaw,
        Actuators.wrist_pitch,
        Actuators.wrist_roll,
    )
    for actuator, delta in zip(actuators, action[:-1], strict=True):
        if delta:
            sim.move_by(actuator, float(delta))
    sim.move_to(Actuators.gripper, float(action[-1]))


if __name__ == "__main__":
    main()
