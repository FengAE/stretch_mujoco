"""Collect successful multi-object grasp episodes from randomized pick scenes.

Each episode contains 3-4 objects. The requested target is sampled in an
IK-verified reachable region; distractors are placed without overlap. Keyboard
and calculator are excluded from both targets and distractors.

Example:
  MUJOCO_GL=egl uv run python examples/collect_random_object_grasp_episodes.py \
    --scene dark --episodes-per-object 50 --headless
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
import random
import time

import click
import mujoco
import numpy as np

from examples.collect_random_grasp_episodes import (
    _close_gripper_until_contact,
    _move_base_slow,
)
from examples.openpi_auto_grasp import EpisodeCapture, _enter_manipulation_mode
from stretch_mujoco import StretchMujocoSimulator
from stretch_mujoco.enums.actuators import Actuators
from stretch_mujoco.enums.stretch_cameras import StretchCameras
from stretch_mujoco.graspgen.ik import StretchGraspIK
from stretch_mujoco.graspgen.objects import (
    TABLE_TOP_Z,
    available_objects,
    build_grasp_scene,
    load_metadata,
    place_on_table,
    table_textures,
)
from stretch_mujoco.grasp_task import GRASP_FRICTION
from stretch_mujoco.openpi_contract import FPS


EXCLUDED_OBJECTS = {"017_calculator", "116_keyboard"}
HORIZONTAL_SEED = np.array([0.0, 0.49, 0.42, 0.385, 0.0, 0.0])
# At the horizontal seed, the gripper's finger-closing axis (TCP local Y) is
# rotated from world X. Remove that offset before applying the object's yaw so
# asymmetric objects are pinched across their local Y dimension.
GRASP_YAW_ALIGNMENT = np.pi / 2 - HORIZONTAL_SEED[3]
TARGET_X_RANGE = (-0.05, 0.16)
TARGET_Y_RANGE = (-0.88, -0.76)
TABLE_X_RANGE = (-0.42, 0.42)
TABLE_Y_RANGE = (-1.32, -0.70)


@dataclass(frozen=True)
class ObjectPlacement:
    x: float
    y: float
    yaw: float


@dataclass(frozen=True)
class EpisodeScene:
    target: str
    objects: tuple[str, ...]
    placements: dict[str, ObjectPlacement]
    grasp_point: tuple[float, float, float]
    grasp_roll: float
    target_joints: tuple[float, ...]


def eligible_objects() -> list[str]:
    return [name for name in available_objects() if name not in EXCLUDED_OBJECTS]


def task_prompt(object_name: str) -> str:
    prefix, separator, suffix = object_name.partition("_")
    label = (suffix if separator and prefix.isdigit() else object_name).replace("-", " ")
    label = label.replace("_", " ")
    return f"pick up the {label}"


def _local_center(object_name: str) -> np.ndarray:
    extents = np.asarray(load_metadata(object_name)["extents"], dtype=float)
    return extents.mean(axis=1)


def _footprint_radius(object_name: str) -> float:
    extents = np.asarray(load_metadata(object_name)["extents"], dtype=float)
    return float(np.max(np.ptp(extents[:2], axis=1)) / 2)


def _grasp_roll(object_name: str) -> float:
    return 0.0


def _world_grasp_point(object_name: str, placement: ObjectPlacement) -> np.ndarray:
    center = _local_center(object_name)
    cosine, sine = np.cos(placement.yaw), np.sin(placement.yaw)
    rotated_xy = np.array(
        [cosine * center[0] - sine * center[1], sine * center[0] + cosine * center[1]]
    )
    body_z = TABLE_TOP_Z - float(load_metadata(object_name)["min_z"])
    return np.array([placement.x + rotated_xy[0], placement.y + rotated_xy[1], body_z + center[2]])


def _solve_grasp(
    ik: StretchGraspIK,
    grasp_point: np.ndarray,
    target_yaw: float,
    grasp_roll: float,
) -> np.ndarray | None:
    target_pose = ik.forward(HORIZONTAL_SEED)
    aligned_yaw = target_yaw + GRASP_YAW_ALIGNMENT
    cosine, sine = np.cos(aligned_yaw), np.sin(aligned_yaw)
    yaw_rotation = np.array([[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]])
    roll_cosine, roll_sine = np.cos(grasp_roll), np.sin(grasp_roll)
    roll_rotation = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, roll_cosine, -roll_sine],
            [0.0, roll_sine, roll_cosine],
        ]
    )
    target_pose[:3, :3] = yaw_rotation @ target_pose[:3, :3] @ roll_rotation
    target_pose[:3, 3] = grasp_point
    initial = HORIZONTAL_SEED.copy()
    initial[5] = grasp_roll
    joints, position_error, orientation_error = ik.solve(target_pose, initial=initial)
    if position_error > 0.01 or orientation_error > 0.05 or joints[2] > 0.50:
        return None
    return joints


def sample_episode_scene(
    target: str,
    rng: random.Random,
    *,
    min_objects: int = 3,
    max_objects: int = 4,
    ik: StretchGraspIK | None = None,
) -> EpisodeScene:
    candidates = eligible_objects()
    if target not in candidates:
        raise ValueError(f"target must be one of {candidates}, got {target!r}")
    count = rng.randint(min_objects, max_objects)
    distractors = rng.sample([name for name in candidates if name != target], count - 1)
    names = (target, *distractors)
    ik = ik or StretchGraspIK()
    grasp_roll = _grasp_roll(target)

    for _ in range(500):
        target_placement = ObjectPlacement(
            rng.uniform(*TARGET_X_RANGE),
            rng.uniform(*TARGET_Y_RANGE),
            rng.uniform(-np.pi, np.pi),
        )
        grasp_point = _world_grasp_point(target, target_placement)
        joints = _solve_grasp(ik, grasp_point, target_placement.yaw, grasp_roll)
        if joints is not None:
            break
    else:
        raise RuntimeError(f"could not sample a reachable pose for {target}")

    placements = {target: target_placement}
    for name in distractors:
        radius = _footprint_radius(name)
        for _ in range(500):
            placement = ObjectPlacement(
                rng.uniform(*TABLE_X_RANGE), rng.uniform(*TABLE_Y_RANGE), rng.uniform(-np.pi, np.pi)
            )
            if all(
                np.hypot(placement.x - other.x, placement.y - other.y)
                > radius + _footprint_radius(other_name) + 0.08
                for other_name, other in placements.items()
            ):
                placements[name] = placement
                break
        else:
            raise RuntimeError(f"could not place distractor {name} without overlap")
    return EpisodeScene(
        target=target,
        objects=names,
        placements=placements,
        grasp_point=tuple(float(value) for value in grasp_point),
        grasp_roll=grasp_roll,
        target_joints=tuple(float(value) for value in joints),
    )


def build_episode_model(scene: EpisodeScene, texture: str) -> mujoco.MjModel:
    model = build_grasp_scene(list(scene.objects), table_texture=texture)
    place_on_table(
        model, {name: (placement.x, placement.y) for name, placement in scene.placements.items()}
    )
    for name, placement in scene.placements.items():
        body = model.body(name)
        qpos = int(model.jnt_qposadr[int(body.jntadr[0])])
        model.qpos0[qpos + 3 : qpos + 7] = (
            np.cos(placement.yaw / 2),
            0.0,
            0.0,
            np.sin(placement.yaw / 2),
        )
    for body_name in ("rubber_tip_left", "rubber_tip_right", *scene.objects):
        body = model.body(body_name)
        first = int(body.geomadr[0])
        for geom_id in range(first, first + int(body.geomnum[0])):
            model.geom_friction[geom_id, 0] = GRASP_FRICTION
    return model


def _prepare_gripper(recorder: EpisodeCapture, joints: np.ndarray) -> None:
    safe_lift = min(max(float(joints[1]) + 0.15, 0.70), 0.90)
    recorder.move({Actuators.gripper: 0.56})
    recorder.move({Actuators.arm: 0.0}, timeout=30.0, max_step=0.03)
    recorder.move({Actuators.lift: safe_lift}, timeout=30.0)


def _move_to_horizontal_grasp(recorder: EpisodeCapture, joints: np.ndarray) -> None:
    recorder.move(
        {
            Actuators.wrist_yaw: joints[3],
            Actuators.wrist_pitch: joints[4],
            Actuators.wrist_roll: joints[5],
        },
        timeout=20.0,
        max_step=0.04,
    )
    recorder.move(
        {
            Actuators.arm: max(joints[2] - 0.08, 0.0),
            Actuators.lift: min(joints[1] + 0.06, 1.1),
        },
        timeout=15.0,
        max_step=0.03,
    )
    recorder.move(
        {Actuators.arm: joints[2], Actuators.lift: joints[1]},
        timeout=12.0,
        required=False,
        max_step=0.015,
        position_tolerance=0.03,
    )


def _align_grasp_center(
    recorder: EpisodeCapture, local_center_offset: np.ndarray, target_yaw: float
) -> None:
    cosine, sine = np.cos(target_yaw), np.sin(target_yaw)
    rotated_offset = np.array(
        [
            cosine * local_center_offset[0] - sine * local_center_offset[1],
            sine * local_center_offset[0] + cosine * local_center_offset[1],
            local_center_offset[2],
        ]
    )
    for _ in range(3):
        metrics = recorder.sim.pull_grasp_metrics()
        target_point = np.asarray(metrics["object_position"], dtype=float) + rotated_offset
        error = target_point - np.asarray(metrics["grasp_center_position"], dtype=float)
        status = recorder.sim.pull_status()
        recorder.move(
            {
                Actuators.lift: np.clip(float(status.lift.pos) + error[2], 0.0, 1.1),
                Actuators.arm: np.clip(float(status.arm.pos) - error[1], 0.0, 0.52),
                Actuators.wrist_yaw: np.clip(
                    float(status.wrist_yaw.pos) + np.clip(error[0] / 0.26, -0.04, 0.04),
                    -1.39,
                    4.42,
                ),
            },
            timeout=2.0,
            required=False,
            max_step=0.02,
            position_tolerance=0.02,
        )


def collect_episode(
    scene: EpisodeScene,
    texture: str,
    record_dir: Path,
    *,
    headless: bool,
) -> Path:
    model = build_episode_model(scene, texture)
    sim = StretchMujocoSimulator(
        model=model,
        camera_hz=FPS,
        cameras_to_use=[StretchCameras.cam_d435i_rgb, StretchCameras.cam_d405_rgb],
    )
    try:
        sim.start(headless=headless)
        sim.set_robot_motion_speed(3.0)
        time.sleep(1.5)
        _enter_manipulation_mode(sim)
        sim.request_grasp_metrics(scene.target)
        time.sleep(0.3)
        initial = sim.pull_grasp_metrics()
        initial_object = np.asarray(initial["object_position"], dtype=float)
        recorder = EpisodeCapture(
            sim,
            task_prompt(scene.target),
            initial_object_position=initial_object,
            episode_metadata={
                "target_object": scene.target,
                "scene_texture": texture,
                "scene_objects": list(scene.objects),
                "placements": {
                    name: asdict(placement) for name, placement in scene.placements.items()
                },
                "grasp_point": list(scene.grasp_point),
                "grasp_roll": scene.grasp_roll,
            },
        )
        joints = np.asarray(scene.target_joints)
        _prepare_gripper(recorder, joints)
        _move_base_slow(recorder, float(joints[0]), timeout=15.0)
        _move_to_horizontal_grasp(recorder, joints)
        _align_grasp_center(
            recorder,
            _local_center(scene.target),
            scene.placements[scene.target].yaw,
        )
        recorder.hold(0.2)
        _close_gripper_until_contact(recorder, max_frames=24)
        recorder.move(
            {Actuators.lift: min(float(recorder.sim.pull_status().lift.pos) + 0.20, 1.1)},
            max_step=0.01,
            position_tolerance=0.01,
        )
        recorder.hold(0.5)
        lifted = sim.pull_grasp_metrics()
        lift_delta = float(lifted["object_position"][2] - initial_object[2])
        if not lifted.get("bilateral_contact", False) or lift_delta < 0.15:
            raise RuntimeError(
                f"unstable lift: contact={lifted.get('bilateral_contact')} lift={lift_delta:.3f}"
            )
        recorder.move({Actuators.arm: 0.06}, max_step=0.015)
        recorder.hold(0.3)
        final = sim.pull_grasp_metrics()
        final_lift = float(final["object_position"][2] - initial_object[2])
        if not final.get("bilateral_contact", False) or final_lift < 0.15:
            raise RuntimeError(f"object slipped during retraction: lift={final_lift:.3f}")

        path = record_dir / (
            f"episode_{scene.target}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.npz"
        )
        recorder.save(path)
        return path
    finally:
        if sim.is_running():
            sim.stop()


@click.command()
@click.option("--scene", "texture", type=click.Choice(sorted(table_textures())), default="dark")
@click.option("--episodes-per-object", type=click.IntRange(min=1), default=50, show_default=True)
@click.option(
    "--objects", default=None, help="comma-separated target objects; default: all eligible"
)
@click.option("--min-scene-objects", type=click.IntRange(min=2), default=3, show_default=True)
@click.option("--max-scene-objects", type=click.IntRange(min=2), default=4, show_default=True)
@click.option("--max-attempts-per-object", type=click.IntRange(min=1), default=250)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option(
    "--record-dir",
    type=click.Path(path_type=Path, file_okay=False),
    default=Path("datasets/random_object_grasps"),
    show_default=True,
)
@click.option("--headless", is_flag=True)
def main(
    texture: str,
    episodes_per_object: int,
    objects: str | None,
    min_scene_objects: int,
    max_scene_objects: int,
    max_attempts_per_object: int,
    seed: int,
    record_dir: Path,
    headless: bool,
) -> None:
    """Collect a balanced successful dataset for every eligible object."""
    targets = (
        [name.strip() for name in objects.split(",") if name.strip()]
        if objects
        else eligible_objects()
    )
    invalid = sorted(set(targets) - set(eligible_objects()))
    if invalid:
        raise click.UsageError(f"unsupported/excluded targets: {invalid}")
    if min_scene_objects > max_scene_objects:
        raise click.UsageError("--min-scene-objects cannot exceed --max-scene-objects")
    if max_scene_objects > len(eligible_objects()):
        raise click.UsageError(f"--max-scene-objects cannot exceed {len(eligible_objects())}")
    record_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    ik = StretchGraspIK()
    total = len(targets) * episodes_per_object
    collected = 0
    for target in targets:
        successes = len(list(record_dir.glob(f"episode_{target}_*.npz")))
        if successes > episodes_per_object:
            successes = episodes_per_object
        collected += successes
        attempts = 0
        while successes < episodes_per_object and attempts < max_attempts_per_object:
            attempts += 1
            try:
                scene = sample_episode_scene(
                    target,
                    rng,
                    min_objects=min_scene_objects,
                    max_objects=max_scene_objects,
                    ik=ik,
                )
                path = collect_episode(scene, texture, record_dir, headless=headless)
            except Exception as exc:
                click.secho(
                    f"[{target} {successes}/{episodes_per_object}] retry {attempts}: {exc}",
                    fg="yellow",
                )
                continue
            successes += 1
            collected += 1
            click.secho(
                f"[{collected}/{total}] {target} {successes}/{episodes_per_object}: {path}",
                fg="green",
            )
        if successes < episodes_per_object:
            raise click.ClickException(
                f"{target}: collected {successes}/{episodes_per_object} after {attempts} attempts"
            )
    click.echo(f"Finished: {collected}/{total} successful episodes in {record_dir}")


if __name__ == "__main__":
    main()
