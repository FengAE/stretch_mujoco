"""Record one verified automatic blue-block grasp in the default Stretch scene."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import time

import click
import mujoco
import numpy as np

from stretch_mujoco import StretchMujocoSimulator
from stretch_mujoco.enums.actuators import Actuators
from stretch_mujoco.enums.stretch_cameras import StretchCameras
from stretch_mujoco.graspgen.ik import StretchGraspIK
from stretch_mujoco.openpi_contract import (
    FPS,
    HEAD_POSE,
    base_action_deltas,
    grasp_diagnostics,
    observation_from_simulator,
    save_episode,
    to_lerobot_frame,
)


OBJECT_ID = "object1"
SCENE = Path(__file__).resolve().parents[1] / "stretch_mujoco" / "models" / "scene.xml"
# 6-DOF seed: [base_x, lift, arm, wrist_yaw, wrist_pitch, wrist_roll].
BLUE_BLOCK_SEED = np.array([0.0, 0.656, 0.383, 0.057, -1.4, 0.056])


def _blue_grasp_model() -> mujoco.MjModel:
    """Keep the red distractor visible without blocking the blue grasp corridor."""
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    body = model.body("object2")
    qpos = int(model.jnt_qposadr[int(body.jntadr[0])])
    model.qpos0[qpos : qpos + 3] = (0.30, -0.80, 0.60)
    return model


class EpisodeCapture:
    def __init__(
        self,
        sim: StretchMujocoSimulator,
        prompt: str,
        *,
        initial_object_position: np.ndarray | None = None,
        episode_metadata: dict | None = None,
    ) -> None:
        self.sim = sim
        self.prompt = prompt
        self.started_at = time.monotonic()
        self.next_sample = self.started_at
        self.observations: list[dict] = []
        self.timestamps: list[float] = []
        self.base_poses: list[tuple[float, float]] = []
        self.base_was_commanded = False
        self.initial_object_position = initial_object_position
        self.episode_metadata = episode_metadata or {}
        self.diagnostics: list[dict] = []

    def capture(self) -> None:
        delay = self.next_sample - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        self.observations.append(observation_from_simulator(self.sim, self.prompt))
        self.timestamps.append(time.monotonic() - self.started_at)
        base = self.sim.pull_status().base
        self.base_poses.append((float(base.x), float(base.theta)))
        if self.initial_object_position is not None:
            self.diagnostics.append(
                grasp_diagnostics(
                    self.sim.pull_grasp_metrics(),
                    initial_object_position=self.initial_object_position,
                )
            )
        self.next_sample = max(self.next_sample + 1.0 / FPS, time.monotonic())

    def hold(self, seconds: float) -> None:
        for _ in range(max(1, round(seconds * FPS))):
            self.capture()

    def wait_for_grasp_alignment(self, *, timeout: float = 3.0, max_distance: float = 0.08) -> None:
        """Wait until the gripper center is both close to the object and stationary."""
        deadline = time.monotonic() + timeout
        previous_center = None
        stable_frames = 0
        while time.monotonic() < deadline:
            self.capture()
            metrics = self.sim.pull_grasp_metrics()
            center = np.asarray(metrics["grasp_center_position"], dtype=float)
            distance = float(metrics["center_distance_m"])
            stable_frames = (
                stable_frames + 1
                if previous_center is not None and np.linalg.norm(center - previous_center) <= 0.005
                else 0
            )
            if distance <= max_distance and stable_frames >= 2:
                return
            previous_center = center
        raise RuntimeError("Gripper did not settle near the object")

    def move(
        self,
        targets: dict[Actuators, float],
        *,
        timeout: float = 8.0,
        required=True,
        max_step: float | None = None,
        position_tolerance: float = 0.025,
    ) -> None:
        """Move the joints to ``targets`` while sampling 10 Hz frames.

        If ``max_step`` is given, large joint moves are interpolated into increments of
        at most ``max_step`` meters/radians, sampling a frame at each waypoint. This
        avoids recording single-step teleports (which made the wrist jump ~1.6 rad in
        one frame). With ``max_step=None`` the move is a single command (used for the
        absolute-position gripper, whose controller creeps too slowly to settle on
        small intermediate waypoints).
        """
        if max_step is None:
            for actuator, target in targets.items():
                self.sim.move_to(actuator, float(target))
            deadline = time.monotonic() + timeout
            settled_frames = 0
            while True:
                self.capture()
                status = self.sim.pull_status()
                at_target = all(
                    abs(getattr(status, actuator.name).pos - target) <= position_tolerance
                    for actuator, target in targets.items()
                )
                settled_frames = settled_frames + 1 if at_target else 0
                if settled_frames >= 2:
                    return
                if time.monotonic() >= deadline:
                    if required:
                        raise RuntimeError(f"Timed out moving {', '.join(a.name for a in targets)}")
                    return
            return

        if max_step <= 0:
            raise ValueError("max_step must be positive")
        deadline = time.monotonic() + timeout
        status = self.sim.pull_status()
        current = {a: float(getattr(status, a.name).pos) for a in targets}
        max_range = max((abs(targets[a] - current[a]) for a in targets), default=0.0)
        n_steps = max(1, int(np.ceil(max_range / max_step)))
        for step in range(1, n_steps + 1):
            frac = step / n_steps
            waypoint = {a: current[a] + frac * (targets[a] - current[a]) for a in targets}
            for actuator, target in waypoint.items():
                self.sim.move_to(actuator, float(target))
            self.capture()

        settled_frames = 0
        while True:
            status = self.sim.pull_status()
            at_target = all(
                abs(getattr(status, actuator.name).pos - target) <= position_tolerance
                for actuator, target in targets.items()
            )
            settled_frames = settled_frames + 1 if at_target else 0
            if settled_frames >= 2:
                return
            if time.monotonic() >= deadline:
                if required:
                    raise RuntimeError(f"Timed out moving {', '.join(a.name for a in targets)}")
                return
            for actuator, target in targets.items():
                self.sim.move_to(actuator, float(target))
            self.capture()

    def move_base_by(self, dx: float, *, timeout: float = 8.0, required: bool = True) -> None:
        """Drive the base forward/backward by ``dx`` meters, sampling frames at 10 Hz."""
        # The continuous base controller overshoots sub-millimeter IK corrections.
        if abs(dx) < 0.005:
            return
        self.base_was_commanded = True
        start_x, _, _ = self.sim.get_base_pose()
        self.sim.move_by(Actuators.base_translate, float(dx))
        deadline = time.monotonic() + timeout
        while True:
            self.capture()
            x, _, _ = self.sim.get_base_pose()
            if abs(x - start_x) >= abs(dx) - 0.005:
                return
            if time.monotonic() >= deadline:
                if required:
                    raise RuntimeError(f"Timed out moving base by {dx:.3f} m")
                return

    def save(self, path: Path) -> Path:
        self.capture()
        frames = []
        for i, (current, following) in enumerate(
            zip(self.observations, self.observations[1:], strict=False)
        ):
            state = current["observation/state"]
            next_state = following["observation/state"]
            action = np.zeros(8, dtype=np.float32)
            # Base deltas come from the tracked planar base pose (x, theta).
            if self.base_was_commanded:
                action[:2] = base_action_deltas(self.base_poses[i], self.base_poses[i + 1])
            action[2:7] = next_state[2:7] - state[2:7]
            action[7] = next_state[7]
            frames.append(to_lerobot_frame(current, action))
        diagnostics = self.diagnostics[:-1] if self.initial_object_position is not None else None
        metadata = self.episode_metadata.copy()
        if diagnostics:
            metadata.update(
                {
                    "initial_object_position": np.asarray(self.initial_object_position).tolist(),
                    "min_center_distance_m": min(item["center_distance_m"] for item in diagnostics),
                    "ever_bilateral_contact": any(
                        item["bilateral_contact"] for item in diagnostics
                    ),
                    "max_object_lift_m": max(item["object_lift_m"] for item in diagnostics),
                    "success": any(
                        item["bilateral_contact"] and item["object_lift_m"] >= 0.15
                        for item in diagnostics
                    ),
                }
            )
        return save_episode(
            path,
            frames,
            self.timestamps[:-1],
            diagnostics=diagnostics,
            metadata=metadata or None,
        )


def _enter_manipulation_mode(
    sim: StretchMujocoSimulator, head_pose: dict[str, float] = HEAD_POSE
) -> None:
    sim.move_to(Actuators.head_pan, head_pose["head_pan"])
    sim.wait_until_at_setpoint(Actuators.head_pan, position_tolerance=0.01)
    sim.move_to(Actuators.head_tilt, head_pose["head_tilt"])
    sim.wait_until_at_setpoint(Actuators.head_tilt, position_tolerance=0.01)
    camera_time = sim.pull_camera_data().time
    deadline = time.monotonic() + 2.0
    while sim.pull_camera_data().time <= camera_time:
        if time.monotonic() >= deadline:
            raise TimeoutError("cameras did not refresh in manipulation mode")
        time.sleep(0.01)


@click.command()
@click.option(
    "--record-dir",
    type=click.Path(path_type=Path, file_okay=False),
    default=Path("datasets/stretch_pick"),
    show_default=True,
)
@click.option("--prompt", default="pick up the blue block", show_default=True)
@click.option("--headless", is_flag=True, help="Run without the MuJoCo viewer.")
@click.option("--motion-speed", type=click.FloatRange(min=0.1), default=3.0, show_default=True)
def main(record_dir: Path, prompt: str, headless: bool, motion_speed: float) -> None:
    sim = StretchMujocoSimulator(
        model=_blue_grasp_model(),
        camera_hz=FPS,
        cameras_to_use=[StretchCameras.cam_d435i_rgb, StretchCameras.cam_d405_rgb],
    )
    recorder = None
    try:
        sim.start(headless=headless)
        # Waypoints limit trajectory speed; stronger servos keep the actual joints from
        # lagging behind them and continuing to descend after grasp closure.
        sim.set_robot_motion_speed(motion_speed)
        # Let the free base and home keyframe settle before recording manipulation.
        time.sleep(1.5)
        _enter_manipulation_mode(sim)
        sim.request_grasp_metrics(OBJECT_ID)
        time.sleep(0.3)

        initial_metrics = sim.pull_grasp_metrics()
        initial_object = np.asarray(initial_metrics["object_position"], dtype=float)
        ik = StretchGraspIK()
        # Solve for [base_x, lift, arm, wrist_yaw, wrist_pitch, wrist_roll]; base_x is
        # the forward base displacement that positions the gripper laterally.
        joints = ik.solve_ik(initial_object, initial=BLUE_BLOCK_SEED)
        if joints is None:
            raise RuntimeError("Blue block is not reachable")

        recorder = EpisodeCapture(sim, prompt, initial_object_position=initial_object)
        recorder.move_base_by(joints[0])
        recorder.move({Actuators.gripper: 0.56})
        # Rate-limit every large joint move to produce executable 10 Hz trajectories.
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
        recorder.move({Actuators.lift: joints[1]}, timeout=4.0, max_step=0.02)

        for _ in range(2):
            metrics = sim.pull_grasp_metrics()
            error = np.asarray(metrics["object_position"]) - np.asarray(
                metrics["grasp_center_position"]
            )
            joints[1] = np.clip(joints[1] + error[2], 0.0, 1.1)
            joints[2] = np.clip(joints[2] - error[1], 0.0, 0.52)
            joints[3] = np.clip(joints[3] + np.clip(error[0] / 0.26, -0.03, 0.03), -1.39, 4.42)
            recorder.move(
                {
                    Actuators.lift: joints[1],
                    Actuators.arm: joints[2],
                    Actuators.wrist_yaw: joints[3],
                    Actuators.wrist_pitch: joints[4],
                    Actuators.wrist_roll: joints[5],
                },
                timeout=2.0,
                required=False,
                max_step=0.03,
                position_tolerance=0.035,
            )

        recorder.wait_for_grasp_alignment()
        # Close the gripper. It reaches ~0.04 (tight on the block) within ~1 s; the
        # remaining creep to -0.15 is slow, so a short timeout avoids a ~4 s stall
        # before the grasp.
        recorder.move({Actuators.gripper: -0.15}, timeout=1.5, required=False)
        recorder.hold(0.3)
        closed = sim.pull_grasp_metrics()
        if not closed.get("bilateral_contact", False):
            raise RuntimeError(
                "Grasp failed: "
                f"left={closed.get('left_finger_contacts', 0)}, "
                f"right={closed.get('right_finger_contacts', 0)}"
            )

        sim.attach_object_to_gripper(OBJECT_ID)
        recorder.hold(0.3)
        recorder.move(
            {Actuators.lift: min(joints[1] + 0.20, 1.1)},
            max_step=0.02,
            position_tolerance=0.01,
        )
        recorder.hold(0.5)
        final_object = np.asarray(sim.pull_grasp_metrics()["object_position"], dtype=float)
        lift_delta = float(final_object[2] - initial_object[2])
        if lift_delta < 0.15:
            raise RuntimeError(f"Grasp did not lift the block: delta={lift_delta:.3f} m")

        path = record_dir / f"episode_auto_blue_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.npz"
        recorder.save(path)
        click.secho(f"Saved successful grasp ({lift_delta:.3f} m lift) to {path}", fg="green")
    finally:
        if sim.is_running():
            sim.stop()


if __name__ == "__main__":
    main()
