from __future__ import annotations

from dataclasses import dataclass
from math import gcd
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class ZoneAxisCandidate:
    uvw: tuple[int, int, int]
    vector: np.ndarray
    index_order: int


@dataclass(frozen=True)
class ZoneAxisMatch:
    uvw: tuple[int, int, int]
    vector: np.ndarray
    row: int
    col: int
    angle_deg: float
    local_strength: float
    score: float

    @property
    def label(self) -> str:
        return format_direction(self.uvw)


def format_direction(uvw: Iterable[int]) -> str:
    return "[" + " ".join(str(int(v)) for v in uvw) + "]"


def lattice_basis_from_parameters(
    a: float,
    b: float,
    c: float,
    alpha_deg: float,
    beta_deg: float,
    gamma_deg: float,
) -> np.ndarray:
    alpha = np.deg2rad(alpha_deg)
    beta = np.deg2rad(beta_deg)
    gamma = np.deg2rad(gamma_deg)

    a_vec = np.array([a, 0.0, 0.0], dtype=np.float64)
    b_vec = np.array([b * np.cos(gamma), b * np.sin(gamma), 0.0], dtype=np.float64)

    c_x = c * np.cos(beta)
    sin_gamma = np.sin(gamma)
    c_y = 0.0 if abs(sin_gamma) < 1e-8 else c * (np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / sin_gamma
    c_z_sq = max(c * c - c_x * c_x - c_y * c_y, 0.0)
    c_vec = np.array([c_x, c_y, np.sqrt(c_z_sq)], dtype=np.float64)

    return np.vstack([a_vec, b_vec, c_vec])


def _gcd3(a: int, b: int, c: int) -> int:
    return gcd(gcd(abs(a), abs(b)), abs(c))


def enumerate_zone_axes(
    lattice_parameters: np.ndarray,
    max_index: int = 4,
) -> list[ZoneAxisCandidate]:
    basis = lattice_basis_from_parameters(*lattice_parameters)
    candidates: list[ZoneAxisCandidate] = []
    seen: set[tuple[int, int, int]] = set()

    for u in range(-max_index, max_index + 1):
        for v in range(-max_index, max_index + 1):
            for w in range(-max_index, max_index + 1):
                if u == 0 and v == 0 and w == 0:
                    continue

                divisor = _gcd3(u, v, w)
                uvw = (u // divisor, v // divisor, w // divisor)
                if uvw in seen:
                    continue

                direction = np.array(uvw, dtype=np.float64) @ basis
                norm = np.linalg.norm(direction)
                if norm < 1e-10:
                    continue

                seen.add(uvw)
                candidates.append(
                    ZoneAxisCandidate(
                        uvw=uvw,
                        vector=(direction / norm).astype(np.float32),
                        index_order=int(abs(uvw[0]) + abs(uvw[1]) + abs(uvw[2])),
                    )
                )

    candidates.sort(key=lambda item: (item.index_order, item.uvw))
    return candidates


def _local_mean(image: np.ndarray, row: int, col: int, radius: int) -> float:
    r0 = max(0, row - radius)
    r1 = min(image.shape[0], row + radius + 1)
    c0 = max(0, col - radius)
    c1 = min(image.shape[1], col + radius + 1)
    patch = image[r0:r1, c0:c1]
    if patch.size == 0:
        return 0.0
    return float(np.mean(patch))


def find_visible_zone_axes(
    matched_points_grid: np.ndarray,
    detector_mask: np.ndarray,
    band_image: np.ndarray,
    lattice_parameters: np.ndarray,
    max_index: int = 4,
    max_angle_deg: float = 2.2,
    neighborhood_radius: int = 7,
    top_n: int = 18,
    min_pixel_spacing: float = 28.0,
) -> list[ZoneAxisMatch]:
    valid_rows, valid_cols = np.nonzero(detector_mask)
    valid_vectors = matched_points_grid[detector_mask].astype(np.float32)

    candidates = enumerate_zone_axes(lattice_parameters=lattice_parameters, max_index=max_index)
    max_angle_rad = np.deg2rad(max_angle_deg)
    accepted: list[ZoneAxisMatch] = []

    for candidate in candidates:
        dots = valid_vectors @ candidate.vector
        best_idx = int(np.argmax(dots))
        best_dot = float(np.clip(dots[best_idx], -1.0, 1.0))
        angle = float(np.arccos(best_dot))
        if angle > max_angle_rad:
            continue

        row = int(valid_rows[best_idx])
        col = int(valid_cols[best_idx])
        local_strength = _local_mean(band_image, row, col, neighborhood_radius)
        score = local_strength + 0.35 * best_dot - 0.04 * candidate.index_order
        accepted.append(
            ZoneAxisMatch(
                uvw=candidate.uvw,
                vector=candidate.vector.copy(),
                row=row,
                col=col,
                angle_deg=float(np.rad2deg(angle)),
                local_strength=local_strength,
                score=score,
            )
        )

    accepted.sort(key=lambda item: (-item.score, item.angle_deg, sum(abs(v) for v in item.uvw)))

    filtered: list[ZoneAxisMatch] = []
    for match in accepted:
        if all(np.hypot(match.row - other.row, match.col - other.col) >= min_pixel_spacing for other in filtered):
            filtered.append(match)
        if len(filtered) >= top_n:
            break

    filtered.sort(key=lambda item: (item.row, item.col))
    return filtered


def overlay_zone_axes(
    image: np.ndarray,
    matches: list[ZoneAxisMatch],
    output_path: str | None = None,
    title: str = "Kikuchi Pattern With Zone Axes",
    cmap: str = "gray",
    show: bool = True,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image, cmap=cmap)

    for match in matches:
        ax.scatter(match.col, match.row, s=22, c="#ff4d4f", edgecolors="white", linewidths=0.6, zorder=3)
        ax.text(
            match.col + 8,
            match.row - 8,
            match.label,
            color="black",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "alpha": 0.82, "edgecolor": "#666666"},
            zorder=4,
        )

    ax.set_title(title)
    ax.axis("off")
    plt.tight_layout()

    if output_path is not None:
        plt.savefig(output_path, dpi=220, bbox_inches="tight")

    if show and "agg" not in plt.get_backend().lower():
        plt.show()
    else:
        plt.close(fig)


def build_notebook_input_panel(
    defaults: dict[str, object] | None = None,
    target_namespace: dict[str, object] | None = None,
) -> object:
    try:
        import ipywidgets as widgets
        from IPython.display import display
    except ImportError as exc:
        raise ImportError("Notebook interactive inputs require ipywidgets and IPython.") from exc

    defaults = dict(defaults or {})
    path_defaults = {
        "pattern_path": "patterns/1.png",
        "master_h5_path": ".venv/Lib/site-packages/kikuchipy/data/emsoft_ebsd_master_pattern/ni_mc_mp_20kv_uint8_gzip_opts9.h5",
        "output_dir": "outputs",
    }
    numeric_defaults = {
        "pcx": 0.5,
        "pcy": 0.5,
        "pcz": 0.6,
        "zone_axis_max_index": 4,
        "zone_axis_max_angle_deg": 2.0,
        "zone_axis_label_count": 12,
        "zone_axis_min_pixel_spacing": 40.0,
    }

    path_widgets = {
        name: widgets.Text(
            value=str(defaults.get(name, fallback)),
            description=name,
            style={"description_width": "160px"},
            layout=widgets.Layout(width="95%"),
        )
        for name, fallback in path_defaults.items()
    }

    numeric_widgets = {
        "pcx": widgets.FloatText(value=float(defaults.get("pcx", numeric_defaults["pcx"])), description="pcx"),
        "pcy": widgets.FloatText(value=float(defaults.get("pcy", numeric_defaults["pcy"])), description="pcy"),
        "pcz": widgets.FloatText(value=float(defaults.get("pcz", numeric_defaults["pcz"])), description="pcz"),
        "zone_axis_max_index": widgets.IntSlider(
            value=int(defaults.get("zone_axis_max_index", numeric_defaults["zone_axis_max_index"])),
            min=1,
            max=8,
            step=1,
            description="max index",
            continuous_update=False,
        ),
        "zone_axis_max_angle_deg": widgets.FloatSlider(
            value=float(defaults.get("zone_axis_max_angle_deg", numeric_defaults["zone_axis_max_angle_deg"])),
            min=0.5,
            max=5.0,
            step=0.1,
            description="max angle",
            readout_format=".1f",
            continuous_update=False,
        ),
        "zone_axis_label_count": widgets.IntSlider(
            value=int(defaults.get("zone_axis_label_count", numeric_defaults["zone_axis_label_count"])),
            min=1,
            max=30,
            step=1,
            description="label count",
            continuous_update=False,
        ),
        "zone_axis_min_pixel_spacing": widgets.FloatSlider(
            value=float(defaults.get("zone_axis_min_pixel_spacing", numeric_defaults["zone_axis_min_pixel_spacing"])),
            min=10.0,
            max=100.0,
            step=1.0,
            description="min spacing",
            readout_format=".0f",
            continuous_update=False,
        ),
    }

    output = widgets.Output()
    apply_button = widgets.Button(
        description="Apply Inputs",
        button_style="primary",
        icon="check",
    )

    def collect_values() -> dict[str, object]:
        values: dict[str, object] = {
            key: Path(widget.value.strip()) for key, widget in path_widgets.items()
        }
        for key, widget in numeric_widgets.items():
            values[key] = widget.value
        return values

    def apply_values(*_args: object) -> dict[str, object]:
        values = collect_values()
        if target_namespace is not None:
            target_namespace.update(values)

        with output:
            output.clear_output()
            print("Current interactive inputs:")
            for key, value in values.items():
                print(f"  {key} = {value}")
            print("Parameters have been applied. Re-run the downstream cells to refresh the result.")
        return values

    apply_button.on_click(apply_values)

    panel = widgets.VBox(
        [
            widgets.HTML("<h3>Interactive Inputs</h3><p>Adjust parameters here, click <b>Apply Inputs</b>, then re-run the following cells.</p>"),
            path_widgets["pattern_path"],
            widgets.HBox([numeric_widgets["pcx"], numeric_widgets["pcy"], numeric_widgets["pcz"]]),
            path_widgets["master_h5_path"],
            path_widgets["output_dir"],
            widgets.HBox([numeric_widgets["zone_axis_max_index"], numeric_widgets["zone_axis_max_angle_deg"]]),
            widgets.HBox([numeric_widgets["zone_axis_label_count"], numeric_widgets["zone_axis_min_pixel_spacing"]]),
            apply_button,
            output,
        ]
    )

    display(panel)
    apply_values()
    return panel
