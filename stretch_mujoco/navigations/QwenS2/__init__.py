"""QwenS2 — instruction-driven goal navigation.

An online Qwen2.5-VL (cloud API) points at a goal pixel from the current view +
a natural-language instruction; NavDP's pixel-goal policy executes the trajectory.
"""

from stretch_mujoco.navigations.QwenS2.planner import (
    QwenS2Diagnostics,
    QwenS2Planner,
)
from stretch_mujoco.navigations.QwenS2.qwen_client import QwenPixelGoalSelector

__all__ = [
    "QwenPixelGoalSelector",
    "QwenS2Planner",
    "QwenS2Diagnostics",
]
