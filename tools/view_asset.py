#!/usr/bin/env python3
"""Quickly preview an office asset XML in the MuJoCo viewer.

Usage:
  .venv/bin/python tools/view_asset.py meeting_table/teaming_table
  .venv/bin/python tools/view_asset.py chairs/office_chair
  .venv/bin/python tools/view_asset.py desks/cb_desk_2400
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

ASSETS_DIR = Path(__file__).resolve().parent.parent / "stretch_mujoco" / "models" / "assets" / "office_assets"


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python tools/view_asset.py <category/name>")
        print(f"Available assets:")
        for xml in sorted(ASSETS_DIR.rglob("*.xml")):
            print(f"  {xml.relative_to(ASSETS_DIR).with_suffix('')}")
        return

    asset_rel = sys.argv[1]
    xml_path = ASSETS_DIR / f"{asset_rel}.xml"
    if not xml_path.exists():
        print(f"Asset not found: {xml_path}")
        print(f"Available:")
        for xml in sorted(ASSETS_DIR.rglob("*.xml")):
            print(f"  {xml.relative_to(ASSETS_DIR).with_suffix('')}")
        return

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    # Position the camera to look at the object
    bounds = _compute_bounds(model)
    center = bounds.mean(axis=0)
    extent = bounds[1] - bounds[0]
    distance = max(float(extent[0]), float(extent[1]), 0.5) * 2.5

    print(f"Asset: {asset_rel}")
    print(f"  bodies: {model.nbody}, geoms: {model.ngeom}")
    print(f"  bounds: {bounds[0]} → {bounds[1]}")
    print("  Controls: drag=rotate/pan  scroll=zoom  Esc=quit")

    with mujoco.viewer.launch_passive(
        model, data, show_left_ui=False, show_right_ui=False,
    ) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.cam.lookat[:] = center
        viewer.cam.distance = distance
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -25

        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.01)


def _compute_bounds(model: mujoco.MjModel) -> np.ndarray:
    """Estimate axis-aligned scene bounds from geom positions."""
    mins = np.full(3, np.inf)
    maxs = np.full(3, -np.inf)
    for gid in range(model.ngeom):
        pos = model.geom_pos[gid]
        size = model.geom_size[gid]
        rbound = model.geom_rbound[gid]
        half = np.full(3, float(rbound)) if rbound > 0 else np.abs(size)
        mins = np.minimum(mins, pos - half)
        maxs = np.maximum(maxs, pos + half)
    return np.array([mins, maxs])


if __name__ == "__main__":
    main()
