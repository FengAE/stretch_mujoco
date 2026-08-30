"""Shared deterministic and randomized settings for the Stretch block-grasp task."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random

import mujoco


GRASP_FRICTION = 0.9
OBJECTS = {"blue": "object1", "red": "object2"}
VALIDATION_EXCLUSION_RADIUS = 0.008


@dataclass(frozen=True)
class GraspScenario:
    target_x: float
    target_y: float
    target_yaw: float = 0.0
    initial_lift: float = 0.597
    initial_arm: float = 0.10
    initial_wrist_yaw: float = 0.0
    initial_wrist_pitch: float = -0.005
    initial_wrist_roll: float = 0.0
    head_pan: float = -1.57
    head_tilt: float = -0.65


def validation_scenarios(target: str) -> tuple[GraspScenario, ...]:
    """Return 20 fixed held-out scenarios for repeatable checkpoint evaluation."""
    center_x = -0.02 if target == "blue" else 0.08
    x_offsets = (-0.045, -0.0225, 0.0, 0.0225, 0.045)
    y_values = (-0.842, -0.814, -0.786, -0.758)
    return tuple(
        GraspScenario(
            target_x=center_x + dx,
            target_y=y,
            target_yaw=(-0.18 + 0.12 * ((row * 5 + col) % 4)),
            initial_lift=0.597 + (-0.02 + 0.01 * (col % 5)),
            initial_arm=0.10 + (-0.02 + 0.01 * (row % 4)),
            initial_wrist_yaw=(-0.06 + 0.03 * ((row + col) % 5)),
            initial_wrist_pitch=-0.005 + (-0.03 + 0.02 * ((2 * row + col) % 4)),
            initial_wrist_roll=-0.04 + 0.02 * ((row + 2 * col) % 5),
            head_pan=-1.57 + (-0.04 + 0.02 * (col % 5)),
            head_tilt=-0.65 + (-0.03 + 0.02 * (row % 4)),
        )
        for row, y in enumerate(y_values)
        for col, dx in enumerate(x_offsets)
    )


def fixed_blue_scenario() -> GraspScenario:
    return GraspScenario(target_x=-0.02, target_y=-0.80)


def random_training_scenario(target: str, rng: random.Random, *, expanded: bool) -> GraspScenario:
    """Sample training randomization while excluding neighborhoods around validation poses."""
    center_x = -0.02 if target == "blue" else 0.08
    held_out = validation_scenarios(target)
    for _ in range(1000):
        x = center_x + rng.uniform(-0.05, 0.05)
        y = -0.80 + rng.uniform(-0.05, 0.05)
        if all(
            math.hypot(x - item.target_x, y - item.target_y) >= VALIDATION_EXCLUSION_RADIUS
            for item in held_out
        ):
            break
    else:
        raise RuntimeError("could not sample outside validation positions")

    if not expanded:
        return GraspScenario(target_x=x, target_y=y)
    return GraspScenario(
        target_x=x,
        target_y=y,
        target_yaw=rng.uniform(-0.25, 0.25),
        initial_lift=0.597 + rng.uniform(-0.03, 0.03),
        initial_arm=0.10 + rng.uniform(-0.03, 0.03),
        initial_wrist_yaw=rng.uniform(-0.08, 0.08),
        initial_wrist_pitch=-0.005 + rng.uniform(-0.05, 0.05),
        initial_wrist_roll=rng.uniform(-0.06, 0.06),
        head_pan=-1.57 + rng.uniform(-0.05, 0.05),
        head_tilt=-0.65 + rng.uniform(-0.04, 0.04),
    )


def apply_scenario(model: mujoco.MjModel, target: str, scenario: GraspScenario) -> str:
    """Apply object pose and shared friction to a freshly loaded model."""
    target_id = OBJECTS[target]
    other_id = OBJECTS["red" if target == "blue" else "blue"]
    other_x = scenario.target_x + (0.30 if target == "blue" else -0.30)
    for body_name, position, yaw in (
        (target_id, (scenario.target_x, scenario.target_y, 0.60), scenario.target_yaw),
        (other_id, (other_x, scenario.target_y, 0.60), 0.0),
    ):
        body = model.body(body_name)
        qpos = int(model.jnt_qposadr[int(body.jntadr[0])])
        model.qpos0[qpos : qpos + 3] = position
        model.qpos0[qpos + 3 : qpos + 7] = (
            math.cos(yaw / 2),
            0.0,
            0.0,
            math.sin(yaw / 2),
        )

    for body_name in ("rubber_tip_left", "rubber_tip_right", "object1", "object2"):
        body = model.body(body_name)
        first = int(body.geomadr[0])
        for geom_id in range(first, first + int(body.geomnum[0])):
            model.geom_friction[geom_id, 0] = GRASP_FRICTION
    return target_id
