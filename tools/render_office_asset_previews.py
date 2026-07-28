"""Render a geometry preview shortlist from the HSSD office candidate catalog."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/stretch_mujoco_matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import trimesh
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


DEFAULT_CATALOG = Path("stretch_mujoco/models/assets/office_assets/catalog/hssd_candidates.json")
DEFAULT_OUTPUT = Path("stretch_mujoco/models/assets/office_assets/previews")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--per-category", type=int, default=3)
    parser.add_argument("--resolution", type=int, default=384)
    parser.add_argument("--max-faces", type=int, default=20_000)
    return parser.parse_args()


def load_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, force="scene", process=False)
    scene = loaded if isinstance(loaded, trimesh.Scene) else trimesh.Scene(loaded)
    mesh = scene.to_geometry()
    if mesh.is_empty:
        raise ValueError("asset contains no renderable triangles")
    return mesh


def render_mesh(mesh: trimesh.Trimesh, output: Path, resolution: int, max_faces: int) -> None:
    faces = mesh.faces
    if len(faces) > max_faces:
        indices = np.linspace(0, len(faces) - 1, max_faces, dtype=np.int64)
        faces = faces[indices]

    # HSSD is Y-up. Plot X/Z on the ground plane and Y vertically.
    vertices = mesh.vertices[:, [0, 2, 1]]
    triangles = vertices[faces]
    lower = vertices.min(axis=0)
    upper = vertices.max(axis=0)
    center = (lower + upper) / 2
    radius = max(float((upper - lower).max()) / 2, 1e-3) * 1.08

    dpi = 100
    figure = plt.figure(figsize=(resolution / dpi, resolution / dpi), dpi=dpi)
    axis = figure.add_subplot(111, projection="3d")
    figure.patch.set_facecolor("#111820")
    axis.set_facecolor("#111820")
    collection = Poly3DCollection(
        triangles,
        facecolor="#88a6b8",
        edgecolor="#263640",
        linewidth=0.04,
        alpha=1.0,
    )
    collection.set_rasterized(True)
    axis.add_collection3d(collection)
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)
    axis.set_box_aspect((1, 1, 1))
    axis.view_init(elev=22, azim=-48)
    axis.set_axis_off()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output, dpi=dpi, facecolor=figure.get_facecolor(), bbox_inches="tight", pad_inches=0
    )
    plt.close(figure)


def main() -> None:
    args = parse_args()
    catalog = json.loads(args.catalog.read_text(encoding="utf-8"))
    selected = []
    category_counts: dict[str, int] = {}
    for candidate in catalog["candidates"]:
        category = candidate["office_asset_category"]
        count = category_counts.get(category, 0)
        if count >= args.per_category:
            continue
        category_counts[category] = count + 1
        selected.append(candidate)

    manifest = []
    for index, candidate in enumerate(selected, start=1):
        category = candidate["office_asset_category"]
        output = args.output / category / f"{candidate['asset_id']}.png"
        print(f"[{index}/{len(selected)}] {candidate['display_name']}")
        try:
            mesh = load_mesh(Path(candidate["source"]["render_asset"]))
            render_mesh(mesh, output, args.resolution, args.max_faces)
            status = "rendered"
            error = None
        except Exception as exception:  # Asset decoders raise several library-specific errors.
            status = "failed"
            error = str(exception)
            print(f"Failed: {error}")
        manifest.append(
            {
                "asset_id": candidate["asset_id"],
                "display_name": candidate["display_name"],
                "category": category,
                "preview": str(output),
                "status": status,
                "error": error,
            }
        )

    manifest_path = args.output / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    rendered = sum(item["status"] == "rendered" for item in manifest)
    print(f"Rendered {rendered}/{len(manifest)} previews; manifest: {manifest_path}")


if __name__ == "__main__":
    main()
