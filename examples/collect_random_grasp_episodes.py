"""Collect randomized, physically validated Stretch block-grasp episodes."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import random
import time

import click
import mujoco
import numpy as np

from examples.openpi_auto_grasp import EpisodeCapture, _enter_manipulation_mode
from stretch_mujoco import StretchMujocoSimulator
from stretch_mujoco.enums.actuators import Actuators
from stretch_mujoco.enums.stretch_cameras import StretchCameras
from stretch_mujoco.graspgen.ik import StretchGraspIK
from stretch_mujoco.grasp_task import (
    GraspScenario,
    apply_scenario,
    fixed_blue_scenario,
    random_training_scenario,
    validation_scenarios,
)
from stretch_mujoco.openpi_contract import FPS


SCENE = Path(__file__).resolve().parents[1] / "stretch_mujoco" / "models" / "scene.xml"
OBJECTS = {"blue": "object1", "red": "object2"}
RED_GRASP_HEIGHT_OFFSET = 0.025
GRIPPER_CLOSE_STEP = 0.04
SEEDS = {
    "blue": np.array([0.0, 0.656, 0.383, 0.057, -1.4, 0.056]),
    "red": np.array([0.0, 0.410, 0.155, 0.385, 0.0, 0.0]),
}


def _randomized_model(
    target: str, rng: random.Random, scenario: GraspScenario | None = None
) -> tuple[mujoco.MjModel, str, str]:
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    scenario = scenario or random_training_scenario(target, rng, expanded=False)
    target_id = apply_scenario(model, target, scenario)
    other = "red" if target == "blue" else "blue"
    other_id = OBJECTS[other]
    return model, target_id, other_id


def _move_base_slow(
    recorder: EpisodeCapture, dx: float, speed: float = 0.08, timeout: float = 8.0
) -> None:
    if abs(dx) < 0.005:
        return
    recorder.base_was_commanded = True
    start_x, _, _ = recorder.sim.get_base_pose()
    recorder.sim.set_base_velocity(np.copysign(speed, dx), 0.0)
    deadline = time.monotonic() + timeout
    while abs(recorder.sim.get_base_pose()[0] - start_x) < abs(dx) - 0.005:
        if time.monotonic() >= deadline:
            recorder.sim.set_base_velocity(0.0, 0.0)
            raise RuntimeError(f"Timed out moving base by {dx:.3f} m")
        recorder.capture()

    recorder.sim.set_base_velocity(0.0, 0.0)
    stable_frames = 0
    while stable_frames < 2:
        recorder.capture()
        base = recorder.sim.pull_status().base
        stable_frames = stable_frames + 1 if abs(base.x_vel) < 0.02 else 0
        if time.monotonic() >= deadline:
            raise RuntimeError("Base did not settle after moving")


def _move_target(recorder: EpisodeCapture, joints: np.ndarray, target: str) -> None:
    if target == "blue":
        recorder.move({Actuators.gripper: 0.56})
        recorder.move({Actuators.arm: 0.0}, max_step=0.02)
        recorder.move({Actuators.lift: 0.85}, max_step=0.02)
        recorder.move(
            {
                Actuators.wrist_yaw: joints[3],
                Actuators.wrist_pitch: joints[4],
                Actuators.wrist_roll: joints[5],
            },
            max_step=0.04,
        )
        recorder.move({Actuators.arm: joints[2]}, max_step=0.04)
        recorder.move({Actuators.lift: min(joints[1] + 0.10, 1.1)}, max_step=0.02)
        recorder.move({Actuators.lift: joints[1]}, timeout=4.0, required=False, max_step=0.02)
    else:
        # Configure the horizontal grasp at a safe height, then approach diagonally.
        recorder.move({Actuators.gripper: 0.56})
        recorder.move({Actuators.arm: 0.0}, max_step=0.02)
        recorder.move({Actuators.lift: 0.85}, max_step=0.02)
        recorder.move(
            {
                Actuators.wrist_yaw: joints[3],
                Actuators.wrist_pitch: joints[4],
                Actuators.wrist_roll: joints[5],
            },
            max_step=0.04,
        )
        recorder.move(
            {
                Actuators.arm: max(joints[2] - 0.08, 0.0),
                Actuators.lift: min(joints[1] + 0.10, 1.1),
            },
            max_step=0.04,
        )
        recorder.move(
            {Actuators.arm: joints[2], Actuators.lift: joints[1]},
            max_step=0.02,
        )
        recorder.move({Actuators.wrist_yaw: joints[3] + 0.10}, max_step=0.03)
        recorder.move({Actuators.arm: joints[2] + 0.02}, max_step=0.02)
        recorder.move({Actuators.wrist_yaw: joints[3] + 0.125}, max_step=0.03)


def _close_gripper_until_contact(
    recorder: EpisodeCapture, *, required_frames: int = 3, max_frames: int = 30
) -> None:
    target = float(recorder.sim.pull_status().gripper.pos)
    stable_frames = 0
    metrics = {}
    for _ in range(max_frames):
        if stable_frames == 0:
            target = max(-0.15, target - GRIPPER_CLOSE_STEP)
            recorder.sim.move_to(Actuators.gripper, target)
        recorder.capture()
        metrics = recorder.sim.pull_grasp_metrics()
        stable_frames = stable_frames + 1 if metrics.get("bilateral_contact", False) else 0
        if stable_frames >= required_frames:
            recorder.sim.move_to(Actuators.gripper, -0.15)
            for _ in range(required_frames):
                recorder.capture()
                metrics = recorder.sim.pull_grasp_metrics()
            if metrics.get("bilateral_contact", False):
                return
            stable_frames = 0
    raise RuntimeError(
        f"no stable bilateral contact (left={metrics.get('left_finger_contacts', 0)}, "
        f"right={metrics.get('right_finger_contacts', 0)})"
    )


def _run_episode(
    target: str,
    record_dir: Path,
    *,
    headless: bool,
    rng: random.Random,
    scenario: GraspScenario | None = None,
    split: str = "train",
) -> Path:
    scenario = scenario or random_training_scenario(target, rng, expanded=False)
    model, object_id, _ = _randomized_model(target, rng, scenario)
    sim = StretchMujocoSimulator(
        model=model,
        camera_hz=FPS,
        cameras_to_use=[StretchCameras.cam_d435i_rgb, StretchCameras.cam_d405_rgb],
    )
    recorder = None
    try:
        sim.start(headless=headless)
        sim.set_robot_motion_speed(3.0)
        time.sleep(1.5)
        for actuator, value in (
            (Actuators.lift, scenario.initial_lift),
            (Actuators.arm, scenario.initial_arm),
            (Actuators.wrist_yaw, scenario.initial_wrist_yaw),
            (Actuators.wrist_pitch, scenario.initial_wrist_pitch),
            (Actuators.wrist_roll, scenario.initial_wrist_roll),
        ):
            sim.move_to(actuator, value)
            sim.wait_until_at_setpoint(actuator, position_tolerance=0.01)
        _enter_manipulation_mode(
            sim, {"head_pan": scenario.head_pan, "head_tilt": scenario.head_tilt}
        )
        sim.request_grasp_metrics(object_id)
        time.sleep(0.3)
        initial = sim.pull_grasp_metrics()
        initial_object = np.asarray(initial["object_position"], dtype=float)

        ik = StretchGraspIK()
        seed = SEEDS[target].copy()
        target_pose = ik.forward(seed)
        target_pose[:3, 3] = initial_object
        if target == "red":
            target_pose[2, 3] += RED_GRASP_HEIGHT_OFFSET
        joints, position_error, _ = ik.solve(target_pose, initial=seed)
        if position_error > 0.01:
            raise RuntimeError(f"IK error={position_error:.3f} m")

        recorder = EpisodeCapture(
            sim,
            f"pick up the {target} block",
            initial_object_position=initial_object,
            episode_metadata={"scenario": scenario.__dict__, "split": split},
        )
        _move_base_slow(recorder, float(joints[0]))
        _move_target(recorder, joints, target)
        recorder.hold(0.3)
        _close_gripper_until_contact(recorder, max_frames=30)

        # sim.attach_object_to_gripper(object_id)  # Disabled to collect physical grasps.
        recorder.move(
            {Actuators.lift: min(joints[1] + 0.20, 1.1)},
            max_step=0.01 if target == "red" else 0.02,
            position_tolerance=0.01,
        )
        recorder.hold(0.5)
        final_object = np.asarray(sim.pull_grasp_metrics()["object_position"], dtype=float)
        lift_delta = float(final_object[2] - initial_object[2])
        if lift_delta < 0.15:
            raise RuntimeError(f"lift={lift_delta:.3f} m")
        recorder.move({Actuators.arm: 0.0}, max_step=0.02)
        recorder.hold(0.3)
        retracted = sim.pull_grasp_metrics()
        retracted_lift = float(retracted["object_position"][2] - initial_object[2])
        if not retracted.get("bilateral_contact", False) or retracted_lift < 0.15:
            raise RuntimeError(f"object slipped during retraction (lift={retracted_lift:.3f} m)")

        path = (
            record_dir / f"episode_auto_{target}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.npz"
        )
        recorder.save(path)
        return path
    finally:
        if sim.is_running():
            sim.stop()


@click.command()
@click.option("--episodes", type=click.IntRange(min=1), default=10, show_default=True)
@click.option(
    "--object", "object_mode", type=click.Choice(["blue", "red", "random"]), default="random"
)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option(
    "--scene-set",
    type=click.Choice(["random", "expanded", "fixed-blue", "validation"]),
    default="random",
    show_default=True,
)
@click.option(
    "--record-dir",
    type=click.Path(path_type=Path, file_okay=False),
    default=Path("datasets/stretch_pick"),
)
@click.option("--headless", is_flag=True)
def main(
    episodes: int,
    object_mode: str,
    seed: int,
    scene_set: str,
    record_dir: Path,
    headless: bool,
) -> None:
    record_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    if scene_set == "fixed-blue" and object_mode != "blue":
        raise click.UsageError("--scene-set=fixed-blue requires --object=blue")
    successes = 0
    for index in range(episodes):
        target = rng.choice(["blue", "red"]) if object_mode == "random" else object_mode
        if scene_set == "fixed-blue":
            scenario = fixed_blue_scenario()
        elif scene_set == "validation":
            scenario = validation_scenarios(target)[index % 20]
        else:
            scenario = random_training_scenario(target, rng, expanded=scene_set == "expanded")
        try:
            path = _run_episode(
                target,
                record_dir,
                headless=headless,
                rng=rng,
                scenario=scenario,
                split="validation" if scene_set == "validation" else "train",
            )
            successes += 1
            click.secho(f"[{index + 1}/{episodes}] OK {path}", fg="green")
        except Exception as exc:
            click.secho(f"[{index + 1}/{episodes}] SKIP {target}: {exc}", fg="yellow")
    click.echo(f"Finished: {successes}/{episodes} successful episodes in {record_dir}")


if __name__ == "__main__":
    main()
