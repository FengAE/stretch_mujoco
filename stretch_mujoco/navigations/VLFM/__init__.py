"""VLFM — Vision-Language Frontier Maps for Zero-Shot Semantic Navigation.

Aligned with `rai-opensource/vlfm <https://github.com/rai-opensource/vlfm>`_
(ICRA 2024).

Core components
---------------
- :class:`VLFMPlanner` — main exploration planner that combines FBE with VLM scoring.
- :class:`ValueMap` — spatial value map that projects VLM scores through FOV.
- :class:`VLMClient` — abstract interface for vision-language backends.
- :class:`AcyclicEnforcer` — prevents frontier oscillation.
- :class:`ImageCapture` — off-screen RGB/D capture from MuJoCo.

Quick start
-----------
.. code-block:: python

    from stretch_mujoco.navigations.VLFM import (
        VLFMPlanner, CLIPVLMClient, ImageCapture,
    )

    vlm = CLIPVLMClient(device="cpu")
    planner = VLFMPlanner(
        god_grid, vlm,
        instruction="Seems like there is a chair ahead.",
    )
    capture = ImageCapture(model, data)

    while not planner.is_finished():
        camera_fovy = 70.0
        rgb, depth = capture.capture_at_robot(
            robot_xy, robot_yaw, fovy=camera_fovy,
        )
        planner.inject_observation(
            rgb, depth, robot_xy, robot_yaw,
            capture.horizontal_fov_rad(camera_fovy),
        )
        planner.step(robot_xy, robot_yaw)
"""

# --- Value map ---
from stretch_mujoco.navigations.VLFM.value_map import (
    ValueMap,
    LanguageValueMap,
)

# --- VLM clients ---
from stretch_mujoco.navigations.VLFM.vlm_client import (
    VLMClient,
    BLIP2ITMVLMClient,
    OpenAIVLMClient,
    CLIPVLMClient,
    SigLIPVLMClient,
    VLMDirectionResult,
)

# --- Acyclic enforcer ---
from stretch_mujoco.navigations.VLFM.acyclic_enforcer import AcyclicEnforcer

# --- Image capture ---
from stretch_mujoco.navigations.VLFM.image_capture import ImageCapture

# --- Planner ---
from stretch_mujoco.navigations.VLFM.planner import (
    VLFMPlanner,
    VLFMDiagnostics,
)

__all__ = [
    # Planner
    "VLFMPlanner",
    "VLFMDiagnostics",
    # Value map
    "ValueMap",
    "LanguageValueMap",
    # VLM
    "VLMClient",
    "BLIP2ITMVLMClient",
    "OpenAIVLMClient",
    "CLIPVLMClient",
    "SigLIPVLMClient",
    "VLMDirectionResult",
    # Support
    "AcyclicEnforcer",
    "ImageCapture",
]
