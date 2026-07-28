"""In-process MuJoCo backend for position-controlled mobile robots."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from mujoco._structs import MjData, MjModel

from stretch_mujoco.robots.base import (
    ActuatorType,
    BodySiteRef,
    RobotActuators,
    RobotCameraData,
    RobotCameras,
    RobotSensorData,
    RobotSensors,
    RobotStatus,
)


class GenericRobotBackend:
    """Small runtime shared by Google Robot and TidyBot.

    These models use named position actuators for their planar bases and do
    not fit the Stretch-specific multiprocessing server. MuJoCo access is
    serialized with a lock while a background thread advances physics.
    """

    def __init__(
        self,
        *,
        scene_xml_path: str,
        model: MjModel | None,
        actuators: type[RobotActuators],
        status_type: type[RobotStatus],
        camera_data_type: type[RobotCameraData],
        sensor_data_type: type[RobotSensorData],
        sensors: type[RobotSensors],
        cameras_to_use: list[RobotCameras],
        camera_hz: float,
        base_actuator_names: tuple[str, str, str],
        grasp_ref: BodySiteRef,
        start_translation: list[float] | None,
        start_rotation_quat: list[float] | None,
        max_linear_velocity: float,
        max_angular_velocity: float,
        initial_joint_positions: dict[str, float] | None = None,
    ) -> None:
        if camera_hz <= 0:
            raise ValueError("camera_hz must be positive")
        self.scene_xml_path = str(Path(scene_xml_path).resolve())
        self.model = model
        self.Actuators = actuators
        self._status_type = status_type
        self._camera_data_type = camera_data_type
        self._sensor_data_type = sensor_data_type
        self.Sensors = sensors
        self._cameras_to_use = list(cameras_to_use)
        self._camera_period = 1.0 / camera_hz
        self._base_names = base_actuator_names
        self._grasp_ref = grasp_ref
        self._start_translation = start_translation
        self._start_rotation_quat = start_rotation_quat
        self._max_linear_velocity = max_linear_velocity
        self._max_angular_velocity = max_angular_velocity
        self._initial_joint_positions = initial_joint_positions or {}

        self.mjmodel: MjModel | None = None
        self.mjdata: MjData | None = None
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._viewer: Any = None
        self._renderers: dict[RobotCameras, mujoco.Renderer] = {}
        self._error: BaseException | None = None
        self._status = status_type.default()
        self._camera_data = camera_data_type.default()
        self._sensor_data = sensor_data_type.default()
        self._base_velocity = np.zeros(3, dtype=float)
        self._last_targets: dict[RobotActuators, float] = {}
        self._visibility: dict[str, list[tuple[int, float, int, int]]] = {}
        self._grasped_object = ""
        self._grasp_transform: np.ndarray | None = None

    def start(
        self,
        *,
        show_viewer_ui: bool,
        headless: bool,
        use_passive_viewer: bool,
    ) -> None:
        if self.is_running():
            return
        if not headless and not use_passive_viewer:
            raise NotImplementedError(
                "The generic robot backend currently supports the passive viewer only"
            )

        self.mjmodel = self.model or MjModel.from_xml_path(self.scene_xml_path)
        self._apply_start_pose()
        self.mjdata = MjData(self.mjmodel)
        self._apply_initial_joint_positions()
        self._initialize_controls()
        mujoco.mj_forward(self.mjmodel, self.mjdata)
        self._update_snapshots(0.0)

        if not headless:
            from mujoco import viewer

            self._viewer = viewer.launch_passive(
                self.mjmodel, self.mjdata, show_left_ui=show_viewer_ui
            )

        self._error = None
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name=f"{self.Actuators.__name__}Physics",
            daemon=True,
        )
        self._thread.start()

        deadline = time.monotonic() + 5.0
        while self.pull_status().time == 0.0:
            self._raise_background_error()
            if time.monotonic() >= deadline:
                self.stop()
                raise TimeoutError("Timed out while starting MuJoCo physics")
            time.sleep(0.01)

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=5.0)
        self._thread = None
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        for renderer in self._renderers.values():
            renderer.close()
        self._renderers.clear()

    def is_running(self) -> bool:
        return (
            self._thread is not None and self._thread.is_alive() and not self._stop_event.is_set()
        )

    def move_to(self, actuator: RobotActuators, position: float) -> None:
        self._check_actuator(actuator)
        with self._lock:
            data = self._require_data()
            model = self._require_model()
            actuator_id = model.actuator(actuator.name).id
            if model.actuator_ctrllimited[actuator_id]:
                lo, hi = model.actuator_ctrlrange[actuator_id]
                position = float(np.clip(position, lo, hi))
            data.actuator(actuator.name).ctrl = position
            self._last_targets[actuator] = float(position)

    def move_by(self, actuator: RobotActuators, increment: float) -> None:
        self._check_actuator(actuator)
        with self._lock:
            current = self._actuator_position(actuator)
        self.move_to(actuator, current + increment)

    def set_base_velocity(self, v_linear: float, omega: float, v_lateral: float = 0.0) -> None:
        linear_norm = float(np.hypot(v_linear, v_lateral))
        if linear_norm > self._max_linear_velocity:
            scale = self._max_linear_velocity / linear_norm
            v_linear *= scale
            v_lateral *= scale
        with self._lock:
            self._base_velocity[:] = (
                float(v_linear),
                float(v_lateral),
                float(
                    np.clip(
                        omega,
                        -self._max_angular_velocity,
                        self._max_angular_velocity,
                    )
                ),
            )

    def apply_keyframe(self, name: str) -> None:
        with self._lock:
            model = self._require_model()
            data = self._require_data()
            key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, name)
            if key_id < 0:
                raise NotImplementedError(f"Keyframe '{name}' is not defined")
            mujoco.mj_resetDataKeyframe(model, data, key_id)
            mujoco.mj_forward(model, data)
            self._last_targets.clear()
            self._base_velocity[:] = 0.0
            self._update_snapshots(0.0)

    def pull_status(self) -> RobotStatus:
        self._raise_background_error()
        with self._lock:
            return self._status.copy()  # type: ignore[attr-defined,no-any-return]

    def pull_camera_data(self) -> RobotCameraData:
        self._raise_background_error()
        with self._lock:
            return self._camera_data.copy()

    def pull_sensor_data(self) -> RobotSensorData:
        self._raise_background_error()
        with self._lock:
            return self._sensor_data.copy()

    def get_base_pose(self) -> tuple[float, float, float]:
        with self._lock:
            return tuple(
                float(self._require_data().actuator(name).length[0]) for name in self._base_names
            )  # type: ignore[return-value]

    def get_pose(self, name: str, kind: str | None = None) -> np.ndarray:
        with self._lock:
            model = self._require_model()
            data = self._require_data()
            if kind in (None, "body"):
                body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
                if body_id >= 0:
                    return self._pose(data.xpos[body_id], data.xmat[body_id])
            if kind in (None, "site"):
                site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
                if site_id >= 0:
                    return self._pose(data.site_xpos[site_id], data.site_xmat[site_id])
        raise KeyError(f"MuJoCo {kind or 'body/site'} '{name}' does not exist")

    def pull_joint_limits(self) -> dict[RobotActuators, tuple[float, float]]:
        model = self._require_model()
        result: dict[RobotActuators, tuple[float, float]] = {}
        for actuator in self.Actuators.all():
            actuator_id = model.actuator(actuator.name).id
            if model.actuator_ctrllimited[actuator_id]:
                result[actuator] = tuple(float(v) for v in model.actuator_ctrlrange[actuator_id])
                continue
            joint_names = actuator.get_joint_names_in_mjcf()
            if joint_names:
                joint_id = model.joint(joint_names[0]).id
                if model.jnt_limited[joint_id]:
                    result[actuator] = tuple(float(v) for v in model.jnt_range[joint_id])
        return result

    def is_reached_set_position(self, actuator: RobotActuators, position_tolerance: float) -> bool:
        self._check_actuator(actuator)
        target = self._last_targets.get(actuator)
        if target is None:
            return True
        with self._lock:
            return bool(
                np.isclose(
                    self._actuator_position(actuator),
                    target,
                    atol=position_tolerance,
                )
            )

    def wait_until_at_setpoint(
        self,
        actuator: RobotActuators,
        timeout: float,
        position_tolerance: float,
    ) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and self.is_running():
            if self.is_reached_set_position(actuator, position_tolerance):
                return True
            time.sleep(0.01)
        return self.is_reached_set_position(actuator, position_tolerance)

    def wait_while_is_moving(
        self,
        actuator: RobotActuators,
        timeout: float | None,
        check_interval: float,
        position_tolerance: float,
    ) -> bool:
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._lock:
            previous = self._actuator_position(actuator)
        while self.is_running():
            time.sleep(check_interval)
            with self._lock:
                current = self._actuator_position(actuator)
            if np.isclose(current, previous, atol=position_tolerance):
                return True
            previous = current
            if deadline is not None and time.monotonic() >= deadline:
                return False
        return False

    def set_object_visibility(self, object_id: str, visible: bool) -> None:
        with self._lock:
            model = self._require_model()
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, object_id)
            if body_id < 0:
                raise KeyError(f"Body '{object_id}' does not exist")
            defaults = self._visibility.setdefault(object_id, [])
            if not defaults:
                descendants = {
                    i for i in range(model.nbody) if self._is_descendant(model, i, body_id)
                }
                for geom_id in range(model.ngeom):
                    if int(model.geom_bodyid[geom_id]) in descendants:
                        defaults.append(
                            (
                                geom_id,
                                float(model.geom_rgba[geom_id, 3]),
                                int(model.geom_contype[geom_id]),
                                int(model.geom_conaffinity[geom_id]),
                            )
                        )
            for geom_id, alpha, contype, conaffinity in defaults:
                model.geom_rgba[geom_id, 3] = alpha if visible else 0.0
                model.geom_contype[geom_id] = contype if visible else 0
                model.geom_conaffinity[geom_id] = conaffinity if visible else 0

    def attach_object_to_gripper(self, object_id: str) -> None:
        with self._lock:
            object_pose = self.get_pose(object_id, "body")
            grasp_pose = self.get_pose(self._grasp_ref.name, self._grasp_ref.mjcf_kind)
            self._grasped_object = object_id
            self._grasp_transform = np.linalg.inv(grasp_pose) @ object_pose

    def release_grasped_object(self) -> None:
        with self._lock:
            self._grasped_object = ""
            self._grasp_transform = None

    def add_world_frame(
        self,
        position: tuple[float, float, float],
        rotation: tuple[float, float, float],
    ) -> None:
        if self._viewer is None:
            return
        rx, ry, rz = rotation
        cx, sx = np.cos(rx), np.sin(rx)
        cy, sy = np.cos(ry), np.sin(ry)
        cz, sz = np.cos(rz), np.sin(rz)
        rotation_matrix = np.array(
            [
                [cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx],
                [sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx],
                [-sy, cy * sx, cy * cx],
            ]
        )
        start = np.asarray(position, dtype=float)
        colors = (
            np.array([1.0, 0.0, 0.0, 1.0]),
            np.array([0.0, 1.0, 0.0, 1.0]),
            np.array([0.0, 0.0, 1.0, 1.0]),
        )
        with self._viewer.lock():
            scene = self._viewer.user_scn
            if scene.ngeom + 3 > scene.maxgeom:
                raise RuntimeError("Viewer user scene has no room for another frame")
            for axis, color in enumerate(colors):
                geom = scene.geoms[scene.ngeom]
                mujoco.mjv_initGeom(
                    geom,
                    mujoco.mjtGeom.mjGEOM_ARROW,
                    np.zeros(3),
                    np.zeros(3),
                    np.eye(3).reshape(9),
                    color,
                )
                mujoco.mjv_connector(
                    geom,
                    mujoco.mjtGeom.mjGEOM_ARROW,
                    0.008,
                    start,
                    start + rotation_matrix[:, axis] * 0.15,
                )
                scene.ngeom += 1

    def _run(self) -> None:
        next_camera_time = 0.0
        wall_start = time.perf_counter()
        sim_start = self._require_data().time
        try:
            self._create_renderers()
            while not self._stop_event.is_set():
                step_start = time.perf_counter()
                with self._lock:
                    model = self._require_model()
                    data = self._require_data()
                    self._update_base_targets(model.opt.timestep)
                    mujoco.mj_step(model, data)
                    self._apply_grasp()
                    elapsed = max(time.perf_counter() - wall_start, 1e-9)
                    fps = (data.time - sim_start) / (model.opt.timestep * elapsed)
                    self._update_snapshots(fps)
                    if data.time >= next_camera_time:
                        self._render_cameras(fps)
                        next_camera_time = data.time + self._camera_period
                    if self._viewer is not None:
                        self._viewer.sync()
                remaining = self._require_model().opt.timestep - (time.perf_counter() - step_start)
                if remaining > 0:
                    time.sleep(remaining)
        except BaseException as error:
            self._error = error
            self._stop_event.set()

    def _apply_start_pose(self) -> None:
        model = self._require_model()
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        if body_id < 0:
            raise ValueError("Body 'base_link' not found in the MjModel")
        joint_ids = [i for i in range(model.njnt) if model.jnt_bodyid[i] == body_id]
        free_joint = next(
            (i for i in joint_ids if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE),
            None,
        )
        if free_joint is not None:
            qadr = model.jnt_qposadr[free_joint]
            if self._start_translation is not None:
                model.qpos0[qadr : qadr + 3] = self._start_translation
            if self._start_rotation_quat is not None:
                model.qpos0[qadr + 3 : qadr + 7] = self._start_rotation_quat
            return

        if self._start_translation is not None:
            for name, value in zip(self._base_names[:2], self._start_translation[:2]):
                model.qpos0[model.jnt_qposadr[model.joint(name).id]] = value
        if self._start_rotation_quat is not None:
            w, x, y, z = self._start_rotation_quat
            yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
            heading = model.joint(self._base_names[2]).id
            model.qpos0[model.jnt_qposadr[heading]] = yaw

    def _initialize_controls(self) -> None:
        data = self._require_data()
        model = self._require_model()
        for actuator in self.Actuators.all():
            actuator_id = model.actuator(actuator.name).id
            joint_names = actuator.get_joint_names_in_mjcf()
            if joint_names and actuator.actuator_type() == ActuatorType.POSITION:
                joint_id = model.joint(joint_names[0]).id
                data.ctrl[actuator_id] = data.qpos[model.jnt_qposadr[joint_id]]

    def _apply_initial_joint_positions(self) -> None:
        model = self._require_model()
        data = self._require_data()
        for joint_name, position in self._initial_joint_positions.items():
            joint_id = model.joint(joint_name).id
            data.qpos[model.jnt_qposadr[joint_id]] = position

    def _create_renderers(self) -> None:
        model = self._require_model()
        if self._cameras_to_use:
            model.vis.global_.offwidth = max(
                model.vis.global_.offwidth,
                *(camera.initial_camera_settings.width for camera in self._cameras_to_use),
            )
            model.vis.global_.offheight = max(
                model.vis.global_.offheight,
                *(camera.initial_camera_settings.height for camera in self._cameras_to_use),
            )
        for camera in self._cameras_to_use:
            camera_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_CAMERA, camera.camera_name_in_mjcf
            )
            if camera_id < 0:
                raise ValueError(
                    f"Camera '{camera.camera_name_in_mjcf}' is not present in the model"
                )
            settings = camera.initial_camera_settings
            model.cam_fovy[camera_id] = settings.field_of_view_vertical_in_degrees
            renderer = mujoco.Renderer(model, width=settings.width, height=settings.height)
            if camera.is_depth:
                renderer.enable_depth_rendering()
            self._renderers[camera] = renderer

    def _render_cameras(self, fps: float) -> None:
        data = self._require_data()
        camera_data = self._camera_data_type.default()
        camera_data.time = float(data.time)
        camera_data.fps = fps
        for camera, renderer in self._renderers.items():
            renderer.update_scene(data, camera=camera.camera_name_in_mjcf)
            pixels = renderer.render().copy()
            callback = camera.post_processing_callback
            camera_data.set_camera_data(camera, callback(pixels) if callback else pixels)
        self._camera_data = camera_data

    def _update_snapshots(self, fps: float) -> None:
        data = self._require_data()
        status = self._status_type.default()
        status.time = float(data.time)
        status.fps = fps
        status.sim_to_real_time_ratio_msg = f"Sim is running {fps:.3f}x as fast as realtime"
        for actuator in self.Actuators.all():
            value = getattr(status, actuator.name)
            value.pos = self._actuator_position(actuator)
            value.vel = float(data.actuator(actuator.name).velocity[0])
        self._status = status
        sensor_data = self._sensor_data_type.default()
        sensor_data.time = float(data.time)
        sensor_data.fps = fps
        for sensor in self.Sensors.all():
            if sensor.is_replicated:
                values = [
                    np.asarray(data.sensor(name).data).reshape(-1)
                    for name in sensor.get_replicated_names(0)
                ]
                value = np.concatenate(values) if values else np.empty(0)
            else:
                value = np.asarray(data.sensor(sensor.sensor_name_in_mjcf).data).copy()
            sensor_data.set_data(sensor, value)
        self._sensor_data = sensor_data

    def _update_base_targets(self, timestep: float) -> None:
        forward, lateral, omega = self._base_velocity
        if forward == 0.0 and lateral == 0.0 and omega == 0.0:
            return
        data = self._require_data()
        x_name, y_name, heading_name = self._base_names
        heading = float(data.actuator(heading_name).length[0])
        dx = (forward * np.cos(heading) - lateral * np.sin(heading)) * timestep
        dy = (forward * np.sin(heading) + lateral * np.cos(heading)) * timestep
        data.actuator(x_name).ctrl = float(data.actuator(x_name).ctrl[0]) + dx
        data.actuator(y_name).ctrl = float(data.actuator(y_name).ctrl[0]) + dy
        data.actuator(heading_name).ctrl = (
            float(data.actuator(heading_name).ctrl[0]) + omega * timestep
        )

    def _actuator_position(self, actuator: RobotActuators) -> float:
        data = self._require_data()
        if actuator.actuator_type() == ActuatorType.GENERAL:
            return float(data.actuator(actuator.name).ctrl[0])
        return float(data.actuator(actuator.name).length[0])

    def _apply_grasp(self) -> None:
        if not self._grasped_object or self._grasp_transform is None:
            return
        model = self._require_model()
        data = self._require_data()
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, self._grasped_object)
        if body_id < 0:
            return
        joint_id = int(model.body_jntadr[body_id])
        if joint_id < 0 or model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_FREE:
            return
        target = (
            self.get_pose(self._grasp_ref.name, self._grasp_ref.mjcf_kind) @ self._grasp_transform
        )
        qadr = int(model.jnt_qposadr[joint_id])
        dadr = int(model.jnt_dofadr[joint_id])
        quaternion = np.empty(4)
        mujoco.mju_mat2Quat(quaternion, target[:3, :3].reshape(9))
        data.qpos[qadr : qadr + 3] = target[:3, 3]
        data.qpos[qadr + 3 : qadr + 7] = quaternion
        data.qvel[dadr : dadr + 6] = 0.0
        mujoco.mj_forward(model, data)

    def _check_actuator(self, actuator: RobotActuators) -> None:
        if not isinstance(actuator, self.Actuators):
            raise TypeError(f"Expected {self.Actuators.__name__}, got {type(actuator)}")

    def _require_model(self) -> MjModel:
        if self.mjmodel is None:
            raise RuntimeError("Simulator not started")
        return self.mjmodel

    def _require_data(self) -> MjData:
        if self.mjdata is None:
            raise RuntimeError("Simulator not started")
        return self.mjdata

    def _raise_background_error(self) -> None:
        if self._error is not None:
            raise RuntimeError("MuJoCo physics thread failed") from self._error

    @staticmethod
    def _pose(position: np.ndarray, rotation: np.ndarray) -> np.ndarray:
        result = np.eye(4)
        result[:3, :3] = rotation.reshape(3, 3)
        result[:3, 3] = position
        return result

    @staticmethod
    def _is_descendant(model: MjModel, body_id: int, ancestor_id: int) -> bool:
        while body_id > 0:
            if body_id == ancestor_id:
                return True
            body_id = int(model.body_parentid[body_id])
        return body_id == ancestor_id
