"""Small MuJoCo helpers shared by the interactive robot keyboard demos."""

from __future__ import annotations

import math

import mujoco
import numpy as np


def actuator_for_joint(model: mujoco.MjModel, joint_name: str) -> int:
    """Return the actuator driving *joint_name*, independent of actuator order."""
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if joint_id < 0:
        raise KeyError(f"Joint does not exist: {joint_name}")
    matches = np.flatnonzero(model.actuator_trnid[:, 0] == joint_id)
    if not len(matches):
        raise KeyError(f"Joint has no actuator: {joint_name}")
    return int(matches[0])


def reset_to_home(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Reset to the model's home keyframe and synchronize position targets."""
    home_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if home_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, home_id)
    else:
        mujoco.mj_resetData(model, data)

    for actuator_id in range(model.nu):
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        if joint_id < 0:
            continue
        joint_type = model.jnt_type[joint_id]
        if joint_type not in {mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE}:
            continue
        target = float(data.qpos[model.jnt_qposadr[joint_id]])
        if model.actuator_ctrllimited[actuator_id]:
            target = float(np.clip(target, *model.actuator_ctrlrange[actuator_id]))
        data.ctrl[actuator_id] = target
    mujoco.mj_forward(model, data)


def change_joint_target(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    joint_name: str,
    delta: float,
) -> float:
    """Increment a position target and clamp only genuinely limited actuators."""
    actuator_id = actuator_for_joint(model, joint_name)
    target = float(data.ctrl[actuator_id]) + delta
    # For unlimited actuators MuJoCo stores ctrlrange=[0, 0]. Clamping to that
    # placeholder was the reason TidyBot printed changing commands but did not move.
    if model.actuator_ctrllimited[actuator_id]:
        target = float(np.clip(target, *model.actuator_ctrlrange[actuator_id]))
    data.ctrl[actuator_id] = target
    return target


def drive_planar_base(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    key: str,
    *,
    x_joint: str,
    y_joint: str,
    heading_joint: str,
    distance_step: float = 0.04,
    angle_step: float = 0.10,
) -> tuple[float, float, float]:
    """Apply Stretch-style W/S translation and A/D rotation to a planar base."""
    x_actuator = actuator_for_joint(model, x_joint)
    y_actuator = actuator_for_joint(model, y_joint)
    heading_actuator = actuator_for_joint(model, heading_joint)
    heading = float(data.ctrl[heading_actuator])

    if key in {"w", "s"}:
        signed_distance = distance_step if key == "w" else -distance_step
        change_joint_target(model, data, x_joint, signed_distance * math.cos(heading))
        change_joint_target(model, data, y_joint, signed_distance * math.sin(heading))
    elif key in {"a", "d"}:
        change_joint_target(
            model,
            data,
            heading_joint,
            angle_step if key == "a" else -angle_step,
        )
    else:
        raise ValueError(f"Not a planar-base key: {key}")

    return (
        float(data.ctrl[x_actuator]),
        float(data.ctrl[y_actuator]),
        float(data.ctrl[heading_actuator]),
    )


def step_for_period(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    period_seconds: float,
) -> None:
    """Advance approximately one real-time control period, not just one physics tick."""
    steps = max(1, int(round(period_seconds / float(model.opt.timestep))))
    mujoco.mj_step(model, data, nstep=steps)
