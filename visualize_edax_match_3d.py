from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import pyvista as pv

from kikuchipy.detectors import EBSDDetector
from kikuchipy.signals.util._master_pattern import _get_direction_cosines_from_detector

from project_edax_oim_to_sphere import (
    EdaxMapInputs,
    estimate_circular_detector_mask,
    load_master_samplers,
    make_master_sampler,
    preprocess_pattern,
    preprocess_master_hemisphere,
    read_edax_inputs,
    sample_master,
)


def parse_circle(value: str | None) -> tuple[int, int, int] | None:
    if value is None:
        return None
    parts = [int(part.strip()) for part in value.split(",")]
    if len(parts) != 3:
        raise ValueError("--circle must be formatted as cx,cy,r")
    return parts[0], parts[1], parts[2]


def circular_mask(shape: tuple[int, int], circle: tuple[int, int, int]) -> np.ndarray:
    cx, cy, radius = circle
    yy, xx = np.indices(shape)
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2


def detector_directions_for_projection(projection) -> np.ndarray:
    detector = EBSDDetector(
        shape=projection.shape,
        pc=projection.pc_edax,
        convention="edax",
        tilt=projection.camera_elevation,
        azimuthal=projection.camera_azimuthal,
        sample_tilt=projection.sample_tilt,
    )
    return _get_direction_cosines_from_detector(detector)


def build_master_sphere_mesh(
    upper_interp,
    lower_interp,
    lon_count: int = 241,
    colat_count: int = 121,
) -> pv.PolyData:
    lon = np.linspace(-np.pi, np.pi, lon_count)
    colat = np.linspace(0.0, np.pi, colat_count)
    lon_grid, colat_grid = np.meshgrid(lon, colat)
    x = np.sin(colat_grid) * np.cos(lon_grid)
    y = np.sin(colat_grid) * np.sin(lon_grid)
    z = np.cos(colat_grid)

    vectors = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    texture = sample_master(vectors, upper_interp, lower_interp).reshape(colat_grid.shape)
    rgba = (cm.gray(texture) * 255).astype(np.uint8)

    grid = pv.StructuredGrid(x, y, z)
    surface = grid.extract_surface(algorithm="dataset_surface")
    surface["rgba"] = rgba.reshape(-1, 4)
    return surface


def build_experimental_patch_mesh(
    projection,
    detector_mask: np.ndarray,
    radius_scale: float = 1.018,
) -> pv.PolyData:
    corrected = preprocess_pattern(projection.pattern)
    detector_directions = detector_directions_for_projection(projection)
    g_h5 = projection.orientation_flat.reshape(3, 3)

    # HDF5 Orientations in this EDAX data behave as sample/lab -> crystal.
    # The code stores directions as row vectors, hence row @ G_h5.T.
    crystal_points = detector_directions @ g_h5.T
    crystal_points /= np.linalg.norm(crystal_points, axis=1, keepdims=True) + 1e-12
    crystal_points = crystal_points.reshape(projection.shape + (3,))
    crystal_points = crystal_points * radius_scale

    values = exposure_to_rgba(corrected)
    values[..., 3] = detector_mask.astype(np.float32)
    rgba = (values * 255).astype(np.uint8)

    valid_indices = np.flatnonzero(detector_mask.ravel())
    vertex_lookup = -np.ones(detector_mask.size, dtype=np.int32)
    vertex_lookup[valid_indices] = np.arange(len(valid_indices), dtype=np.int32)

    faces: list[int] = []
    nrows, ncols = detector_mask.shape
    for row in range(nrows - 1):
        for col in range(ncols - 1):
            if (
                detector_mask[row, col]
                and detector_mask[row + 1, col]
                and detector_mask[row + 1, col + 1]
                and detector_mask[row, col + 1]
            ):
                p00 = int(vertex_lookup[row * ncols + col])
                p10 = int(vertex_lookup[(row + 1) * ncols + col])
                p11 = int(vertex_lookup[(row + 1) * ncols + col + 1])
                p01 = int(vertex_lookup[row * ncols + col + 1])
                faces.extend([3, p00, p10, p11, 3, p00, p11, p01])

    points = crystal_points.reshape(-1, 3)[valid_indices]
    mesh = pv.PolyData(points, np.array(faces, dtype=np.int32))
    mesh["rgba"] = rgba.reshape(-1, 4)[valid_indices]
    return mesh


def exposure_to_rgba(image: np.ndarray) -> np.ndarray:
    normalized = image.astype(np.float32)
    normalized = (normalized - normalized.min()) / (normalized.max() - normalized.min() + 1e-8)
    rgba = cm.magma(normalized)
    return rgba.astype(np.float32)


def save_html_viewer(gltf_path: Path, html_path: Path) -> None:
    gltf_text = gltf_path.read_text(encoding="utf-8")
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>EDAX Kikuchi Sphere Match</title>
  <style>
    html, body {{ margin: 0; padding: 0; overflow: hidden; background: #fff; }}
    #wrap {{ width: 100vw; height: 100vh; }}
    canvas {{ display: block; }}
    #label {{ position: absolute; left: 12px; top: 12px; padding: 8px 10px; background: rgba(255,255,255,.86); font: 12px/1.35 sans-serif; border-radius: 6px; }}
  </style>
</head>
<body>
  <div id="wrap"></div>
  <div id="label">Drag: rotate | Wheel: zoom | Right-drag: pan</div>
  <script type="module">
    import * as THREE from 'https://unpkg.com/three@0.160.0/build/three.module.js';
    import {{ OrbitControls }} from 'https://unpkg.com/three@0.160.0/examples/jsm/controls/OrbitControls.js';
    import {{ GLTFLoader }} from 'https://unpkg.com/three@0.160.0/examples/jsm/loaders/GLTFLoader.js';

    const container = document.getElementById('wrap');
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0xffffff);

    const camera = new THREE.PerspectiveCamera(45, container.clientWidth / container.clientHeight, 0.01, 100);
    camera.position.set(0, -2.9, 1.35);

    const renderer = new THREE.WebGLRenderer({{ antialias: true }});
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setSize(container.clientWidth, container.clientHeight);
    container.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;

    scene.add(new THREE.AmbientLight(0xffffff, 1.1));
    const dir = new THREE.DirectionalLight(0xffffff, 0.9);
    dir.position.set(2, -3, 4);
    scene.add(dir);

    const gltfText = {json.dumps(gltf_text)};
    const blob = new Blob([gltfText], {{ type: 'model/gltf+json' }});
    const url = URL.createObjectURL(blob);
    const loader = new GLTFLoader();
    loader.load(url, (gltf) => {{
      gltf.scene.traverse((obj) => {{
        if (obj.isMesh) {{
          obj.material = obj.material.clone();
          obj.material.vertexColors = true;
          obj.material.side = THREE.DoubleSide;
          obj.material.roughness = 1.0;
        }}
      }});
      scene.add(gltf.scene);
      const box = new THREE.Box3().setFromObject(gltf.scene);
      const center = box.getCenter(new THREE.Vector3());
      controls.target.copy(center);
      controls.update();
    }});

    window.addEventListener('resize', () => {{
      camera.aspect = container.clientWidth / container.clientHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(container.clientWidth, container.clientHeight);
    }});

    function animate() {{
      requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    }}
    animate();
  </script>
</body>
</html>
"""
    html_path.write_text(html, encoding="utf-8")


def render_scene(
    master_mesh: pv.PolyData,
    patch_mesh: pv.PolyData,
    screenshot_path: Path,
    gltf_path: Path,
    title: str,
) -> None:
    plotter = pv.Plotter(off_screen=True, window_size=(1500, 1100))
    plotter.set_background("white")
    plotter.add_text(title, position="upper_left", font_size=10, color="black")
    plotter.add_mesh(
        master_mesh,
        scalars="rgba",
        rgb=True,
        smooth_shading=False,
        opacity=0.82,
        lighting=False,
    )
    patch_actor = plotter.add_mesh(
        patch_mesh,
        scalars="rgba",
        rgb=True,
        smooth_shading=False,
        opacity=1.0,
        lighting=False,
    )
    patch_actor.GetMapper().SetResolveCoincidentTopologyToPolygonOffset()
    patch_actor.GetMapper().SetRelativeCoincidentTopologyPolygonOffsetParameters(-8.0, -8.0)
    plotter.add_axes()
    plotter.camera_position = [(2.2, -4.2, 1.8), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)]
    plotter.camera.zoom(0.9)
    screenshot_path.parent.mkdir(parents=True, exist_ok=True)
    plotter.screenshot(screenshot_path)
    plotter.export_gltf(gltf_path)
    plotter.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a 3D standard Kikuchi sphere with an EDAX experimental pattern patch matched onto it."
    )
    parser.add_argument(
        "--h5",
        type=Path,
        default=Path(r"F:\kikuchi-super resolution\20260512Cu resolution-contrast.edaxh5"),
    )
    parser.add_argument(
        "--up2",
        type=Path,
        default=Path(r"F:\kikuchi-super resolution\20260512_Cu_Area 2_OIM Map 1.up2"),
    )
    parser.add_argument("--map-group", default="/20260512/Cu/Area 2/OIM Map 2HighR")
    parser.add_argument("--pattern-index", type=int, default=19802)
    parser.add_argument(
        "--master",
        type=Path,
        default=Path(
            r"E:\EBSD-projiect\.venv\Lib\site-packages\kikuchipy\data\emsoft_ebsd_master_pattern\ni_mc_mp_20kv_uint8_gzip_opts9.h5"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(r"E:\EBSD-projiect\EBSD\outputs\edax_3d_match"),
    )
    parser.add_argument(
        "--circle",
        default=None,
        help="Optional fixed detector circle formatted as cx,cy,r. Example: 234,235,236",
    )
    parser.add_argument(
        "--master-display",
        choices=("raw", "corrected", "band"),
        default="band",
        help="Contrast mode for the standard sphere background.",
    )
    args = parser.parse_args()

    projection = read_edax_inputs(
        EdaxMapInputs(
            h5_path=args.h5,
            up2_path=args.up2,
            map_group=args.map_group,
            pattern_index=args.pattern_index,
        )
    )
    circle = parse_circle(args.circle)
    if circle is None:
        _mask, circle = estimate_circular_detector_mask(projection.pattern)
    detector_mask = circular_mask(projection.pattern.shape, circle)

    upper, lower, _upper_interp, _lower_interp = load_master_samplers(args.master)
    visual_upper = preprocess_master_hemisphere(upper, args.master_display)
    visual_lower = preprocess_master_hemisphere(lower, args.master_display)
    upper_interp = make_master_sampler(visual_upper)
    lower_interp = make_master_sampler(visual_lower)
    master_mesh = build_master_sphere_mesh(upper_interp, lower_interp)
    patch_mesh = build_experimental_patch_mesh(projection, detector_mask)

    stem = f"{Path(args.up2).stem}_index{args.pattern_index}_3d_match"
    screenshot_path = args.output_dir / f"{stem}.png"
    gltf_path = args.output_dir / f"{stem}.gltf"
    html_path = args.output_dir / f"{stem}_viewer.html"
    title = (
        f"{Path(args.up2).stem}, index {args.pattern_index}; "
        f"PC internal={tuple(round(v, 4) for v in projection.pc_internal)}; "
        f"circle={circle}; master={args.master_display}"
    )
    render_scene(master_mesh, patch_mesh, screenshot_path, gltf_path, title)
    save_html_viewer(gltf_path, html_path)

    print(f"Saved screenshot: {screenshot_path}")
    print(f"Saved glTF: {gltf_path}")
    print(f"Saved HTML viewer: {html_path}")


if __name__ == "__main__":
    main()
