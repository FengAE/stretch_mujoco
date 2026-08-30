"""Replay recorded Stretch actions and measure execution-contract error."""

from __future__ import annotations

import json
from pathlib import Path
import time

import click
import cv2
import mujoco
import numpy as np

from examples.evaluate_openpi_policy import (
    ACTION_DEADBANDS,
    ACTUATORS,
    CAMERAS,
    SCENE,
    _enter_manipulation_mode,
    _execute_action,
    _safe_action,
)
from examples.openpi_auto_grasp import _blue_grasp_model
from stretch_mujoco import StretchMujocoSimulator
from stretch_mujoco.enums.actuators import Actuators
from stretch_mujoco.openpi_contract import (
    FPS,
    STATE_NAMES,
    observation_from_simulator,
    state_from_status,
    validate_action,
)


def _load_episode(path: Path) -> dict[str, np.ndarray | str]:
    required = {"image", "wrist_image", "state", "actions", "task", "fps"}
    with np.load(path, allow_pickle=False) as data:
        missing = required - set(data.files)
        if missing:
            raise ValueError(f"Episode is missing fields: {sorted(missing)}")
        episode = {key: np.array(data[key]) for key in required - {"task"}}
        episode["task"] = str(data["task"].item())

    state = episode["state"]
    actions = episode["actions"]
    if state.ndim != 2 or state.shape[1] != 8 or actions.shape != state.shape:
        raise ValueError(f"Expected matching (frames, 8) state/actions, got {state.shape}/{actions.shape}")
    if episode["image"].shape[0] != len(state) or episode["wrist_image"].shape[0] != len(state):
        raise ValueError("Image and state frame counts do not match")
    if int(episode["fps"].item()) != FPS:
        raise ValueError(f"Expected {FPS} fps")
    return episode


def _execute_velocity_action(sim, action: np.ndarray) -> None:
    action = _safe_action(action)
    base = np.where(np.abs(action[:2]) > ACTION_DEADBANDS[:2], action[:2], 0.0)
    sim.set_base_velocity(float(base[0] * FPS), float(base[1] * FPS))
    for actuator, delta, deadband in zip(
        ACTUATORS[2:], action[2:7], ACTION_DEADBANDS[2:], strict=True
    ):
        if abs(delta) > deadband:
            sim.move_by(actuator, float(delta))
    sim.move_to(Actuators.gripper, float(action[7]))


def _execute_target_action(sim, action: np.ndarray, joint_targets: np.ndarray) -> None:
    """Accumulate recorded deltas into the absolute targets used during collection."""
    action = validate_action(action)
    sim.set_base_velocity(float(action[0] * FPS), float(action[1] * FPS))
    joint_targets += action[2:7]
    for actuator, target in zip(ACTUATORS[2:], joint_targets, strict=True):
        sim.move_to(actuator, float(target))
    sim.move_to(Actuators.gripper, float(action[7]))


def _label(image: np.ndarray, text: str) -> np.ndarray:
    image = np.asarray(image).copy()
    cv2.rectangle(image, (0, 0), (image.shape[1], 24), (20, 24, 28), -1)
    cv2.putText(image, text, (7, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    return image


def _comparison_frame(
    recorded_head: np.ndarray,
    replay_head: np.ndarray,
    recorded_wrist: np.ndarray,
    replay_wrist: np.ndarray,
    step: int,
) -> np.ndarray:
    return np.vstack(
        [
            np.hstack(
                [_label(recorded_head, f"RECORDED HEAD | {step}"), _label(replay_head, "REPLAY HEAD")]
            ),
            np.hstack(
                [
                    _label(recorded_wrist, "RECORDED WRIST"),
                    _label(replay_wrist, "REPLAY WRIST"),
                ]
            ),
        ]
    )


def _initialize_robot(sim: StretchMujocoSimulator, initial_state: np.ndarray) -> None:
    targets = {
        actuator: float(value)
        for actuator, value in zip(ACTUATORS[2:], initial_state[2:7], strict=True)
    }
    targets[Actuators.gripper] = float(initial_state[7])
    for actuator, target in targets.items():
        sim.move_to(actuator, target)
    for actuator in targets:
        sim.wait_until_at_setpoint(actuator, timeout=5.0, position_tolerance=0.01)


def replay_episode(
    episode: Path,
    video_path: Path,
    *,
    executor: str = "target",
    scene_mode: str = "default",
    motion_speed: float = 3.0,
    max_steps: int | None = None,
) -> dict:
    """Replay one NPZ episode and write the RECORDED|REPLAY comparison video.

    ``executor`` controls how recorded actions are applied: "target" accumulates
    deltas into absolute joint targets (matching collection), "velocity" drives
    joints directly, and "evaluation" reuses the policy-eval executor.

    Returns the execution report dict (state RMSE, base deltas, video path).
    """
    data = _load_episode(episode)
    states = np.asarray(data["state"], dtype=np.float32)
    actions = np.asarray(data["actions"], dtype=np.float32)
    frames = min(len(states), max_steps or len(states))

    model = (
        _blue_grasp_model()
        if scene_mode == "fixed-blue"
        else mujoco.MjModel.from_xml_path(str(SCENE))
    )
    sim = StretchMujocoSimulator(model=model, camera_hz=FPS, cameras_to_use=CAMERAS[:2])
    writer = None
    replay_states = []
    try:
        sim.start(headless=True)
        sim.set_robot_motion_speed(motion_speed)
        time.sleep(1.5)
        _enter_manipulation_mode(sim)
        _initialize_robot(sim, states[0])
        time.sleep(0.3)
        joint_targets = states[0, 2:7].astype(np.float64).copy()
        start_base = np.asarray(sim.get_base_pose())
        next_step = time.monotonic()

        for step in range(frames):
            delay = next_step - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            observation = observation_from_simulator(sim, str(data["task"]))
            replay_states.append(state_from_status(sim.pull_status()))
            frame = _comparison_frame(
                np.asarray(data["image"])[step],
                observation["observation/image"],
                np.asarray(data["wrist_image"])[step],
                observation["observation/wrist_image"],
                step,
            )
            if writer is None:
                writer = cv2.VideoWriter(
                    str(video_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    FPS,
                    (frame.shape[1], frame.shape[0]),
                )
                if not writer.isOpened():
                    raise RuntimeError(f"Cannot open video writer: {video_path}")
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

            if executor == "evaluation":
                _execute_action(sim, actions[step], joint_targets)
            elif executor == "velocity":
                _execute_velocity_action(sim, actions[step])
            else:
                _execute_target_action(sim, actions[step], joint_targets)
            next_step = max(next_step + 1.0 / FPS, time.monotonic())

        delay = next_step - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        if executor in {"velocity", "target"}:
            sim.set_base_velocity(0.0, 0.0)
        end_base = np.asarray(sim.get_base_pose())
    finally:
        if writer is not None:
            writer.release()
        if sim.is_running():
            sim.stop()

    replay_states = np.asarray(replay_states)
    error = replay_states - states[:frames]
    expected_base_delta = actions[:frames, :2].sum(axis=0)
    actual_base_delta = np.array(
        [
            end_base[0] - start_base[0],
            (end_base[2] - start_base[2] + np.pi) % (2 * np.pi) - np.pi,
        ]
    )
    return {
        "episode": str(episode),
        "task": data["task"],
        "frames": frames,
        "executor": executor,
        "scene": scene_mode,
        "motion_speed": motion_speed,
        "state_rmse": dict(
            zip(STATE_NAMES, np.sqrt(np.mean(error**2, axis=0)).tolist(), strict=True)
        ),
        "expected_base_delta_x_theta": expected_base_delta.tolist(),
        "actual_base_delta_x_theta": actual_base_delta.tolist(),
        "video": str(video_path),
        "scene_note": (
            "fixed-blue restores only the legacy deterministic blue scene"
            if scene_mode == "fixed-blue"
            else "the NPZ format does not store randomized object positions; visual replay is not exact"
        ),
    }


@click.command()
@click.option("--episode", type=click.Path(path_type=Path, exists=True, dir_okay=False), required=True)
@click.option("--output-dir", type=click.Path(path_type=Path, file_okay=False), required=True)
@click.option(
    "--executor",
    type=click.Choice(["evaluation", "velocity", "target"]),
    default="target",
    show_default=True,
)
@click.option("--scene", "scene_mode", type=click.Choice(["default", "fixed-blue"]), default="default")
@click.option("--motion-speed", type=click.FloatRange(min=0.1), default=3.0, show_default=True)
@click.option("--max-steps", type=click.IntRange(min=1), default=None)
def main(
    episode: Path,
    output_dir: Path,
    executor: str,
    scene_mode: str,
    motion_speed: float,
    max_steps: int | None,
) -> None:
    """Replay one NPZ episode without a policy and report state tracking error."""
    output_dir.mkdir(parents=True, exist_ok=False)
    video_path = output_dir / "comparison.mp4"
    report = replay_episode(
        episode,
        video_path,
        executor=executor,
        scene_mode=scene_mode,
        motion_speed=motion_speed,
        max_steps=max_steps,
    )
    (output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    click.echo(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
