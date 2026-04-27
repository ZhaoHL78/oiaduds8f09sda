from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyvista as pv


def iter_polydata(dataset):
    if isinstance(dataset, pv.MultiBlock):
        for block in dataset:
            yield from iter_polydata(block)
    elif isinstance(dataset, pv.PolyData) and dataset.n_points > 0:
        yield dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Preview exported Kikuchi sphere glTF files with a native PyVista window.")
    parser.add_argument(
        "gltf",
        nargs="?",
        default="outputs/2_matched_to_master_sphere.gltf",
        help="Path to the exported .gltf file.",
    )
    parser.add_argument("--width", type=int, default=1100, help="Preview window width.")
    parser.add_argument("--height", type=int, default=850, help="Preview window height.")
    parser.add_argument(
        "--keep-overlay-radius",
        action="store_true",
        help="Keep overlay mesh radius exactly as stored in the glTF file.",
    )
    args = parser.parse_args()

    gltf_path = Path(args.gltf)
    if not gltf_path.exists():
        raise FileNotFoundError(f"glTF file not found: {gltf_path}")

    scene = pv.read(gltf_path)
    meshes = list(iter_polydata(scene))
    if not meshes:
        raise RuntimeError(f"No mesh data found in: {gltf_path}")

    plotter = pv.Plotter(window_size=(args.width, args.height))
    plotter.set_background("white")

    for index, mesh in enumerate(meshes):
        preview_mesh = mesh.copy(deep=True)
        if index > 0 and not args.keep_overlay_radius:
            radius = np.linalg.norm(preview_mesh.points, axis=1)
            valid = radius > 1e-8
            preview_mesh.points[valid] = preview_mesh.points[valid] / radius[valid, None]

        opacity = 1.0
        if "COLOR_0" in preview_mesh.array_names:
            actor = plotter.add_mesh(
                preview_mesh,
                scalars="COLOR_0",
                rgb=True,
                smooth_shading=False,
                opacity=opacity,
                lighting=False,
            )
        else:
            actor = plotter.add_mesh(preview_mesh, color="lightgray", smooth_shading=False, opacity=opacity, lighting=False)

        if index > 0:
            actor.GetMapper().SetResolveCoincidentTopologyToPolygonOffset()
            actor.GetMapper().SetRelativeCoincidentTopologyPolygonOffsetParameters(-8.0, -8.0)

    plotter.add_axes()
    plotter.camera_position = "xy"
    plotter.show(title=f"Kikuchi preview - {gltf_path.name}", interactive=True, auto_close=True)


if __name__ == "__main__":
    main()
