"""Constrained numerical IK for GraspGen candidates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from stretch_mujoco.utils import URDFmodel


JOINT_LOWER = np.array([0.0, 0.0, -1.39, -1.57, -3.14])
JOINT_UPPER = np.array([1.1, 0.52, 4.42, 0.56, 3.14])


@dataclass(frozen=True)
class GraspIKCandidate:
    index: int
    joints: np.ndarray
    target_pose: np.ndarray
    position_error: float
    orientation_error: float
    confidence: float

    @property
    def score(self) -> float:
        task_error = (
            self.position_error * 100.0 + self.orientation_error * 0.2 - self.confidence * 0.05
        )
        wrist_posture = abs(float(self.joints[3])) * 0.02 + abs(float(self.joints[4])) * 0.01
        return task_error + wrist_posture


class StretchGraspIK:
    def __init__(self, urdf_model: URDFmodel | None = None) -> None:
        self.urdf_model = urdf_model or URDFmodel()

    @staticmethod
    def _configuration(joints: np.ndarray) -> dict[str, float]:
        return {
            "lift": float(joints[0]),
            "arm": float(joints[1]),
            "wrist_yaw": float(joints[2]),
            "wrist_pitch": float(joints[3]),
            "wrist_roll": float(joints[4]),
            "head_pan": 0.0,
            "head_tilt": 0.0,
        }

    def forward(self, joints: np.ndarray) -> np.ndarray:
        return self.urdf_model.get_transform(
            self._configuration(np.asarray(joints, dtype=float)), "link_grasp_center"
        )

    def solve(
        self,
        target_pose: np.ndarray,
        *,
        initial: np.ndarray | None = None,
    ) -> tuple[np.ndarray, float, float]:
        target = np.asarray(target_pose, dtype=float)
        if target.shape != (4, 4):
            raise ValueError(f"Expected a 4x4 target pose, got {target.shape}")
        initial = (
            np.array([0.7, 0.4, 0.0, -0.5, 0.0])
            if initial is None
            else np.asarray(initial, dtype=float)
        )

        def residual(joints: np.ndarray) -> np.ndarray:
            actual = self.forward(joints)
            position = (actual[:3, 3] - target[:3, 3]) * 20.0
            orientation = Rotation.from_matrix(target[:3, :3] @ actual[:3, :3].T).as_rotvec()
            return np.r_[position, orientation * 0.3]

        result = least_squares(
            residual,
            x0=np.clip(initial, JOINT_LOWER, JOINT_UPPER),
            bounds=(JOINT_LOWER, JOINT_UPPER),
            max_nfev=500,
        )
        actual = self.forward(result.x)
        position_error = float(np.linalg.norm(actual[:3, 3] - target[:3, 3]))
        orientation_error = float(
            Rotation.from_matrix(target[:3, :3] @ actual[:3, :3].T).magnitude()
        )
        return result.x, position_error, orientation_error

    def rank_candidates(
        self,
        target_poses: np.ndarray,
        confidences: np.ndarray,
        *,
        max_position_error: float = 0.015,
    ) -> list[GraspIKCandidate]:
        ranked: list[GraspIKCandidate] = []
        for index, (pose, confidence) in enumerate(zip(target_poses, confidences)):
            joints, position_error, orientation_error = self.solve(pose)
            if position_error <= max_position_error:
                ranked.append(
                    GraspIKCandidate(
                        index=index,
                        joints=joints,
                        target_pose=np.asarray(pose),
                        position_error=position_error,
                        orientation_error=orientation_error,
                        confidence=float(confidence),
                    )
                )
        return sorted(ranked, key=lambda candidate: candidate.score)
