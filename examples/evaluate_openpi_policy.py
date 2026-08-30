"""Evaluate a served Stretch OpenPI policy on randomized block grasps."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import random
import time

import click
import cv2
import mujoco
import numpy as np
from openpi_client import websocket_client_policy

from stretch_mujoco import StretchMujocoSimulator
from stretch_mujoco.enums.actuators import Actuators
from stretch_mujoco.enums.stretch_cameras import StretchCameras
from stretch_mujoco.grasp_task import (
    GRASP_FRICTION,
    GraspScenario,
    apply_scenario,
    fixed_blue_scenario,
    random_training_scenario,
    validation_scenarios,
)
from stretch_mujoco.openpi_contract import (
    FPS,
    HEAD_POSE,
    grasp_diagnostics,
    observation_from_simulator,
    state_from_status,
    validate_action,
)
from stretch_mujoco.robots.stretch3.config import GRIPPER_MIN_MAX


SCENE = Path(__file__).resolve().parents[1] / "stretch_mujoco" / "models" / "scene.xml"
OBJECTS = {"blue": "object1", "red": "object2"}
CAMERAS = [
    StretchCameras.cam_d435i_rgb,
    StretchCameras.cam_d405_rgb,
    StretchCameras.office_overview_rgb,
]
ACTUATORS = (
    Actuators.base_translate,
    Actuators.base_rotate,
    Actuators.lift,
    Actuators.arm,
    Actuators.wrist_yaw,
    Actuators.wrist_pitch,
    Actuators.wrist_roll,
)
# ponytail: conservative per-100-ms limits; replace with measured Stretch 3 limits before hardware use.
ACTION_LIMITS = np.array([0.01, 0.01, 0.10, 0.05, 0.20, 0.20, 0.20], dtype=np.float32)
ACTION_DEADBANDS = np.array([3e-3, 5e-4, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5], dtype=np.float32)


def _randomized_model(
    target: str, rng: random.Random, scenario: GraspScenario | None = None
) -> tuple[mujoco.MjModel, str]:
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    scenario = scenario or random_training_scenario(target, rng, expanded=False)
    target_id = apply_scenario(model, target, scenario)
    return model, target_id


def _enter_manipulation_mode(
    sim: StretchMujocoSimulator, head_pose: dict[str, float] = HEAD_POSE
) -> None:
    sim.move_to(Actuators.head_pan, head_pose["head_pan"])
    sim.wait_until_at_setpoint(Actuators.head_pan, position_tolerance=0.01)
    sim.move_to(Actuators.head_tilt, head_pose["head_tilt"])
    sim.wait_until_at_setpoint(Actuators.head_tilt, position_tolerance=0.01)
    previous = sim.pull_camera_data().time
    deadline = time.monotonic() + 2.0
    while sim.pull_camera_data().time <= previous:
        if time.monotonic() >= deadline:
            raise TimeoutError("Cameras did not refresh in manipulation mode")
        time.sleep(0.01)


def _safe_action(action: np.ndarray) -> np.ndarray:
    action = validate_action(np.asarray(action)[:8]).copy()
    action[:7] = np.clip(action[:7], -ACTION_LIMITS, ACTION_LIMITS)
    action[7] = np.clip(action[7], *GRIPPER_MIN_MAX)
    return action


def _applied_action(action: np.ndarray) -> np.ndarray:
    """Return the clipped action after executor deadbands are applied."""
    action = _safe_action(action)
    action[:7] = np.where(np.abs(action[:7]) > ACTION_DEADBANDS, action[:7], 0.0)
    return action


def _execute_action(
    sim: StretchMujocoSimulator, action: np.ndarray, joint_targets: np.ndarray
) -> None:
    action = _applied_action(action)
    sim.set_base_velocity(float(action[0] * FPS), float(action[1] * FPS))
    joint_targets += action[2:7]
    for actuator, target in zip(ACTUATORS[2:], joint_targets, strict=True):
        sim.move_to(actuator, float(target))
    sim.move_to(Actuators.gripper, float(action[7]))


def _label(frame: np.ndarray, text: str) -> np.ndarray:
    frame = frame.copy()
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 34), (20, 24, 28), -1)
    cv2.putText(frame, text, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    return frame


def _pad_to(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    """Center *frame* on a black canvas of *width* x *height* without rescaling.

    Keeps the camera at its native resolution; only adds black borders.
    """
    h, w = frame.shape[:2]
    if w > width or h > height:
        raise ValueError(f"Frame {frame.shape} is larger than tile ({width}, {height})")
    top = (height - h) // 2
    bottom = height - h - top
    left = (width - w) // 2
    right = width - w - left
    return cv2.copyMakeBorder(frame, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))


def _video_frame(sim: StretchMujocoSimulator, detail: str) -> np.ndarray:
    cameras = sim.pull_camera_data()
    overview = cameras.get_camera_data(StretchCameras.office_overview_rgb)
    head = cameras.get_camera_data(StretchCameras.cam_d435i_rgb)
    wrist = cameras.get_camera_data(StretchCameras.cam_d405_rgb)
    overview = _label(overview, f"THIRD PERSON | {detail}")
    head = _label(head, "HEAD D435i")
    wrist = _label(wrist, "WRIST D405")
    # Keep every camera at its native resolution; pad to a common tile instead of rescaling.
    # Note: the D435i head feed is portrait (H=424, W=240) after optical-frame correction,
    # so it cannot be shown in the old landscape (480, 270) tile without distortion.
    right_width = max(head.shape[1], wrist.shape[1])
    wrist_tile = _pad_to(wrist, right_width, wrist.shape[0])
    head_tile = _pad_to(head, right_width, head.shape[0])
    right = np.vstack((wrist_tile, head_tile))
    overview_tile = _pad_to(overview, overview.shape[1], right.shape[0])
    return np.hstack((overview_tile, right))


def _open_writer(path: Path, width: int, height: int) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open MP4 writer: {path}")
    return writer


def _wait_for_metrics(sim: StretchMujocoSimulator, object_id: str) -> dict:
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        metrics = sim.pull_grasp_metrics()
        if metrics.get("object_id") == object_id:
            return metrics
        time.sleep(0.02)
    raise TimeoutError(f"No grasp metrics for {object_id}")


def _run_episode(
    policy,
    index: int,
    target: str,
    rng: random.Random,
    video_path: Path,
    trace_path: Path,
    scenario: GraspScenario | None = None,
    *,
    max_steps: int,
    open_loop_horizon: int,
) -> dict:
    scenario = scenario or random_training_scenario(target, rng, expanded=False)
    model, object_id = _randomized_model(target, rng, scenario)
    sim = StretchMujocoSimulator(model=model, camera_hz=FPS, cameras_to_use=CAMERAS)
    writer = None
    started = time.monotonic()
    result = {
        "episode": index,
        "target": target,
        "video": str(video_path),
        "scenario": scenario.__dict__,
    }
    try:
        sim.start(headless=True)
        sim.set_robot_motion_speed(3.0)
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
        initial = _wait_for_metrics(sim, object_id)
        result["initial_position"] = initial["object_position"]
        policy.reset()

        chunk = None
        chunk_index = open_loop_horizon
        joint_targets = None
        ever_contact = False
        stable_success = 0
        max_lift = 0.0
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        result["trace"] = str(trace_path)
        with trace_path.open("w") as trace:
            for step in range(max_steps):
                if chunk is None or chunk_index >= open_loop_horizon:
                    sim.set_base_velocity(0.0, 0.0)
                    observation = observation_from_simulator(sim, f"pick up the {target} block")
                    chunk = np.asarray(policy.infer(observation)["actions"])
                    if chunk.ndim != 2 or chunk.shape[0] < open_loop_horizon or chunk.shape[1] < 8:
                        raise ValueError(f"Policy returned invalid action chunk {chunk.shape}")
                    chunk_index = 0
                    joint_targets = np.asarray(
                        observation["observation/state"][2:7], dtype=np.float64
                    ).copy()

                action_index = chunk_index
                raw_action = np.asarray(chunk[action_index])[:8]
                clipped_action = _safe_action(raw_action)
                executed_action = _applied_action(raw_action)
                tick = time.monotonic()
                _execute_action(sim, executed_action, joint_targets)
                chunk_index += 1
                remaining = 1.0 / FPS - (time.monotonic() - tick)
                if remaining > 0:
                    time.sleep(remaining)

                metrics = sim.pull_grasp_metrics()
                diagnostic = grasp_diagnostics(
                    metrics, initial_object_position=initial["object_position"]
                )
                trace.write(
                    json.dumps(
                        {
                            "step": step,
                            "elapsed_s": time.monotonic() - started,
                            "action_chunk_index": action_index,
                            "raw_action": raw_action.tolist(),
                            "clipped_action": clipped_action.tolist(),
                            "executed_action": executed_action.tolist(),
                            "state": state_from_status(sim.pull_status()).tolist(),
                            **diagnostic,
                        }
                    )
                    + "\n"
                )
                trace.flush()
                # No welding: success requires the object to stay pinched (bilateral
                # finger contact) while lifted, i.e. the grasp must hold under physics.
                held = diagnostic["bilateral_contact"]
                ever_contact = ever_contact or held
                lift = diagnostic["object_lift_m"]
                max_lift = max(max_lift, lift)
                stable_success = stable_success + 1 if held and lift >= 0.15 else 0
                frame = _video_frame(sim, f"episode {index:03d} | {target} | step {step:03d}")
                if writer is None:
                    writer = _open_writer(video_path, frame.shape[1], frame.shape[0])
                writer.write(frame)
                if stable_success >= 3:
                    result.update(success=True, steps=step + 1)
                    break
            else:
                result.update(success=False, steps=max_steps)

        result.update(max_lift_m=max_lift, bilateral_contact=ever_contact)
    finally:
        if writer is not None:
            writer.release()
        if sim.is_running():
            sim.set_base_velocity(0.0, 0.0)
            sim.stop()
        result["duration_s"] = time.monotonic() - started
    return result


def _write_results(path: Path, episodes: list[dict], requested: int) -> None:
    successes = sum(bool(item.get("success")) for item in episodes)
    payload = {
        "requested_episodes": requested,
        "completed_episodes": len(episodes),
        "successes": successes,
        "success_rate": successes / len(episodes) if episodes else 0.0,
        "episodes": episodes,
    }
    temporary = path.with_suffix(".json.partial")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


@click.command()
@click.option("--host", default="localhost", show_default=True)
@click.option("--port", type=int, default=8000, show_default=True)
@click.option("--episodes", type=click.IntRange(min=1), default=100, show_default=True)
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
@click.option("--max-steps", type=click.IntRange(min=1), default=200, show_default=True)
@click.option("--open-loop-horizon", type=click.IntRange(min=1), default=4, show_default=True)
@click.option("--output-dir", type=click.Path(path_type=Path, file_okay=False), default=None)
def main(
    host: str,
    port: int,
    episodes: int,
    object_mode: str,
    seed: int,
    scene_set: str,
    max_steps: int,
    open_loop_horizon: int,
    output_dir: Path | None,
) -> None:
    """Run independent randomized grasp evaluations against an OpenPI policy server."""
    output_dir = output_dir or Path("evaluations") / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=False)
    policy = websocket_client_policy.WebsocketClientPolicy(host, port)
    rng = random.Random(seed)
    if scene_set == "fixed-blue" and object_mode != "blue":
        raise click.UsageError("--scene-set=fixed-blue requires --object=blue")
    records = []
    for index in range(episodes):
        target = rng.choice(list(OBJECTS)) if object_mode == "random" else object_mode
        if scene_set == "fixed-blue":
            scenario = fixed_blue_scenario()
        elif scene_set == "validation":
            scenario = validation_scenarios(target)[index % 20]
        else:
            scenario = random_training_scenario(target, rng, expanded=scene_set == "expanded")
        video = output_dir / f"episode_{index:03d}_{target}.mp4"
        trace = output_dir / f"episode_{index:03d}_{target}.jsonl"
        try:
            record = _run_episode(
                policy,
                index,
                target,
                rng,
                video,
                trace,
                scenario,
                max_steps=max_steps,
                open_loop_horizon=open_loop_horizon,
            )
        except Exception as exc:
            record = {
                "episode": index,
                "target": target,
                "success": False,
                "video": str(video),
                "trace": str(trace),
                "scenario": scenario.__dict__,
                "error": f"{type(exc).__name__}: {exc}",
            }
        records.append(record)
        _write_results(output_dir / "results.json", records, episodes)
        successes = sum(item["success"] for item in records)
        click.echo(
            f"[{index + 1}/{episodes}] {'SUCCESS' if record['success'] else 'FAIL'} "
            f"{target}; total={successes}/{index + 1}"
        )

    successes = sum(item["success"] for item in records)
    click.secho(f"Finished: {successes}/{episodes} successes; results in {output_dir}", fg="green")


if __name__ == "__main__":
    main()
