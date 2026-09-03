"""Render our own generated office_01..10 scenes with habitat-sim's renderer.

Unlike render_habitat_gt.py (which shows the original, unrelated HSSD
apartment those assets came from), this rebuilds *our* layout: it reads an
office_XX.json manifest, maps each placed asset's asset_id back to its
original HSSD object template id (via tools/extract_office_assets.py's
TARGETS table - legacy_sofas already store the template id directly), and
adds those objects into an empty habitat-sim world at the same position/yaw
our generator used. The camera orbits the scene like
examples/generated_office_scene.py --auto-orbit and is saved as a GIF, so you
can eyeball the same layout rendered by MuJoCo vs. by habitat's own renderer.

Procedural props (rugs, plants, snack items, the snack counter itself) have
no HSSD template and are skipped - this is a furniture-layout/scale/material
check, not a full re-render.

Usage:
  .venv/bin/python tools/render_habitat_scene_gt.py --scene office_01_linear_bench
  .venv/bin/python tools/render_habitat_scene_gt.py --scene office_04_central_meeting --frames 48 --out /tmp/habitat_gt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import habitat_sim
import magnum as mn
import numpy as np
from habitat_sim.utils.common import quat_from_angle_axis, quat_from_two_vectors

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_office_assets import TARGETS  # noqa: E402

DEFAULT_HSSD_ROOT = Path("/home/yjw/data/hssd-hab")
SCENES_DIR = Path(__file__).resolve().parents[1] / "stretch_mujoco" / "models" / "assets" / "office_scenes"


def _out_name_to_template_id() -> dict[str, str]:
    return {out_name: template_id for template_id, out_name, _category in TARGETS.values()}


def _mj_to_habitat_pos(x: float, y: float, z: float) -> np.ndarray:
    # HABITAT_TO_MUJOCO (stretch_mujoco/gltf_material_converter.py) maps
    # habitat (x,y,z) -> mujoco (x,-z,y); inverting: mujoco (x,y,z) -> habitat (x,z,-y).
    return np.array([x, z, -y])


def _mj_yaw_to_habitat_quat(yaw: float) -> mn.Quaternion:
    # A rotation about mujoco's Z axis maps to a rotation about habitat's Y
    # axis under the same conversion (mujoco Z is the image of habitat Y).
    q = quat_from_angle_axis(yaw, np.array([0.0, 1.0, 0.0]))
    return mn.Quaternion(mn.Vector3(q.x, q.y, q.z), q.w)


def resolve_template_id(asset_id: str, category: str, out_name_to_tid: dict[str, str]) -> str | None:
    if category == "legacy_sofas":
        return asset_id
    if asset_id.startswith("new_"):
        return out_name_to_tid.get(asset_id[len("new_") :])
    return None


def make_sim(hssd_root: Path, width: int, height: int) -> habitat_sim.Simulator:
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_dataset_config_file = str(hssd_root / "hssd-hab.scene_dataset_config.json")
    sim_cfg.scene_id = "NONE"
    sim_cfg.enable_physics = False
    sim_cfg.override_scene_light_defaults = True
    sim_cfg.scene_light_setup = habitat_sim.gfx.DEFAULT_LIGHTING_KEY
    sensor_spec = habitat_sim.CameraSensorSpec()
    sensor_spec.uuid = "color"
    sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    sensor_spec.resolution = [height, width]
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [sensor_spec]
    sim = habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))
    lights = [
        habitat_sim.gfx.LightInfo(
            vector=mn.Vector4(direction[0], direction[1], direction[2], 0.0),
            color=mn.Vector3(0.9, 0.9, 0.9),
            model=habitat_sim.gfx.LightPositionModel.Global,
        )
        for direction in ((0.5, -1.0, 0.3), (-0.6, -0.4, -0.7), (0.0, 1.0, 0.0))
    ]
    sim.set_light_setup(lights, habitat_sim.gfx.DEFAULT_LIGHTING_KEY)
    return sim


def populate_scene(
    sim: habitat_sim.Simulator, manifest: dict, out_name_to_tid: dict[str, str]
) -> tuple[list[str], list[str]]:
    template_mgr = sim.get_object_template_manager()
    rigid_mgr = sim.get_rigid_object_manager()
    placed, skipped = [], []
    for item in manifest["assets"]:
        asset_id = item["asset_id"]
        if asset_id.startswith("procedural_"):
            skipped.append(item["name"])
            continue
        template_id = resolve_template_id(asset_id, item["category"], out_name_to_tid)
        if template_id is None:
            skipped.append(item["name"])
            continue
        handles = [
            h
            for h in template_mgr.get_template_handles(template_id)
            if Path(h).name == f"{template_id}.object_config.json"
        ]
        if not handles:
            skipped.append(item["name"])
            continue
        obj = rigid_mgr.add_object_by_template_handle(handles[0])
        x, y, z = item["position"]
        obj.translation = _mj_to_habitat_pos(x, y, z)
        obj.rotation = _mj_yaw_to_habitat_quat(item["yaw"])
        placed.append(item["name"])
    return placed, skipped


def render_orbit_gif(
    sim: habitat_sim.Simulator,
    manifest: dict,
    out_path: Path,
    *,
    frames: int,
    orbit_speed_deg: float,
    elevation_deg: float,
) -> None:
    width_m, depth_m = manifest["dimensions_m"]
    radius = max(width_m, depth_m) * 0.55
    center = np.array([0.0, 1.1, 0.0])
    eye_height = radius * np.tan(np.radians(elevation_deg))
    agent = sim.get_agent(0)
    frames_out = []
    for index in range(frames):
        azimuth = np.radians(index * orbit_speed_deg)
        offset = np.array([radius * np.sin(azimuth), eye_height, radius * np.cos(azimuth)])
        position = center + offset
        direction = center - position
        direction /= np.linalg.norm(direction)
        state = habitat_sim.AgentState()
        state.position = position
        state.rotation = quat_from_two_vectors(np.array([0.0, 0.0, -1.0]), direction)
        agent.set_state(state)
        obs = sim.get_sensor_observations()["color"]
        frames_out.append(obs[:, :, :3].copy())
    import imageio

    imageio.mimsave(out_path, frames_out, fps=15)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True, help="Scene id, e.g. office_01_linear_bench")
    parser.add_argument("--hssd-root", type=Path, default=DEFAULT_HSSD_ROOT)
    parser.add_argument("--out", type=Path, default=Path("/tmp/habitat_gt"))
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--orbit-speed", type=float, default=6.0, help="Degrees per frame")
    parser.add_argument("--elevation", type=float, default=28.0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    args = parser.parse_args()

    manifest = json.loads((SCENES_DIR / f"{args.scene}.json").read_text(encoding="utf-8"))
    sim = make_sim(args.hssd_root, args.width, args.height)
    try:
        placed, skipped = populate_scene(sim, manifest, _out_name_to_template_id())
        print(f"placed {len(placed)} objects, skipped {len(skipped)} (procedural/unmapped): {skipped}")
        args.out.mkdir(parents=True, exist_ok=True)
        out_path = args.out / f"{args.scene}_habitat_orbit.gif"
        render_orbit_gif(
            sim,
            manifest,
            out_path,
            frames=args.frames,
            orbit_speed_deg=args.orbit_speed,
            elevation_deg=args.elevation,
        )
        print(f"wrote {out_path}")
    finally:
        sim.close()


if __name__ == "__main__":
    main()
