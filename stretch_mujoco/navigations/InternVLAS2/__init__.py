"""InternVLAS2 — instruction-driven VLN with the InternVLA-N1 dual system.

The heavy S2 (Qwen2.5-VL) + S1 (diffusion policy) run remotely as
``http_internvla_server.py`` (GPU workstation); these classes only speak its
``/eval_dual`` HTTP API and execute the returned trajectory / discrete actions.
"""

from stretch_mujoco.navigations.InternVLAS2.internvla_client import InternVLAClient
from stretch_mujoco.navigations.InternVLAS2.planner import (
    InternVLAS2Diagnostics,
    InternVLAS2Planner,
)

__all__ = [
    "InternVLAClient",
    "InternVLAS2Planner",
    "InternVLAS2Diagnostics",
]
