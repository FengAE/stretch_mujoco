"""NavDP — closed-loop goal navigation with the NavDP diffusion policy.

The heavy model runs remotely as ``navdp_server.py`` (GPU workstation); these
classes only speak its HTTP API and follow the returned trajectory locally.
"""

from stretch_mujoco.navigations.NavDP.navdp_client import NavDPClient
from stretch_mujoco.navigations.NavDP.planner import (
    NavDPDiagnostics,
    NavDPPlanner,
)
from stretch_mujoco.navigations.NavDP.trajectory_tracker import TrajectoryTracker

__all__ = [
    "NavDPClient",
    "NavDPPlanner",
    "NavDPDiagnostics",
    "TrajectoryTracker",
]
