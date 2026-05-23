from __future__ import annotations

import argparse
import json
import math
import struct
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import cv2
import h5py
import matplotlib
import numpy as np
from scipy import ndimage as ndi
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial.transform import Rotation as R
from skimage import exposure, filters, morphology

matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", message=r"Parameter `area_threshold` is deprecated.*", category=FutureWarning)
warnings.filterwarnings("ignore", message=r"Parameter `min_size` is deprecated.*", category=FutureWarning)

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - tqdm is a display helper only.
    tqdm = None


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = SCRIPT_DIR.parent
DEFAULT_DATA_DIR = Path(r"C:\Users\WHJ\Desktop\kikuchi-super resolution")
DEFAULT_H5_PATH = WORKSPACE_DIR / "ebsd.edaxh5"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs" / "h5_band_enhanced"
MASTER_CANDIDATES = [
    SCRIPT_DIR / ".venv" / "Lib" / "site-packages" / "kikuchipy" / "data" / "emsoft_ebsd_master_pattern" / "ni_mc_mp_20kv_uint8_gzip_opts9.h5",
    Path(r"D:\anaconda3\envs\torch\Lib\site-packages\kikuchipy\data\emsoft_ebsd_master_pattern\ni_mc_mp_20kv_uint8_gzip_opts9.h5"),
    Path(r"D:\anaconda3\envs\nb\Lib\site-packages\kikuchipy\data\emsoft_ebsd_master_pattern\ni_mc_mp_20kv_uint8_gzip_opts9.h5"),
    Path(r"D:\anaconda3\envs\proj\Lib\site-packages\kikuchipy\data\emsoft_ebsd_master_pattern\ni_mc_mp_20kv_uint8_gzip_opts9.h5"),
]


@dataclass(frozen=True)
class Up2Info:
    path: Path
    version: int
    width: int
    height: int
    header_bytes: int
    count: int
    dtype: str = "<u2"

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.count, self.height, self.width)

    @property
    def pattern_bytes(self) -> int:
        return self.width * self.height * np.dtype(self.dtype).itemsize

    def to_json_dict(self) -> dict:
        data = asdict(self)
        data["path"] = str(self.path)
        data["shape"] = self.shape
        return data


def read_up2_info(path: str | Path) -> Up2Info:
    path = Path(path)
    with path.open("rb") as f:
        header = f.read(16)
    if len(header) != 16:
        raise ValueError(f"{path} is too small to contain a UP2 header")

    version, width, height, header_bytes = struct.unpack("<4I", header)
    pattern_bytes = width * height * np.dtype("<u2").itemsize
    payload_bytes = path.stat().st_size - header_bytes
    if payload_bytes < 0 or pattern_bytes <= 0 or payload_bytes % pattern_bytes:
        raise ValueError(
            f"{path} has an inconsistent UP2 header: "
            f"width={width}, height={height}, header={header_bytes}, size={path.stat().st_size}"
        )

    return Up2Info(
        path=path,
        version=version,
        width=width,
        height=height,
        header_bytes=header_bytes,
        count=payload_bytes // pattern_bytes,
    )


class Up2Stack:
    def __init__(self, path: str | Path):
        self.info = read_up2_info(path)
        self._data: np.memmap | None = None

    def __len__(self) -> int:
        return self.info.count

    def _array(self) -> np.memmap:
        if self._data is None:
            self._data = np.memmap(
                self.info.path,
                dtype=self.info.dtype,
                mode="r",
                offset=self.info.header_bytes,
                shape=self.info.shape,
                order="C",
            )
        return self._data

    def __getitem__(self, index: int) -> np.ndarray:
        return self._array()[index]


@dataclass(frozen=True)
class MapSpec:
    key: str
    label: str
    h5_group: str
    up2_path: Path
    rows: int
    cols: int


@dataclass(frozen=True)
class OHPHeader:
    circle_size: int
    max_band_count: int
    max_rho_fraction: float
    max_band_width: float
    theta_step_size: float


@dataclass(frozen=True)
class Band:
    rho_bin: float
    theta_deg: float
    width: float
    intensity: float


@dataclass(frozen=True)
class LineVariant:
    name: str
    theta_is_line_angle: bool
    rho_sign: float
    y_axis: str


@dataclass
class LineSegment:
    band_index: int
    band: Band
    row0: float
    col0: float
    row1: float
    col1: float


@dataclass(frozen=True)
class MatchWeights:
    image_line: float
    intensity: float
    h5_band: float


@dataclass
class PatternBundle:
    map_spec: MapSpec
    index: int
    row: int
    col: int
    pattern_u16: np.ndarray
    pc: tuple[float, float, float]
    bands: list[Band]
    ohp_header: OHPHeader
    ang_record: dict
    up2_info: Up2Info


@dataclass
class PreparedPattern:
    bundle: PatternBundle
    image: np.ndarray
    valid_mask: np.ndarray
    corrected: np.ndarray
    image_band_score: np.ndarray
    corrected_score: np.ndarray
    h5_band_score: np.ndarray
    h5_line_mask: np.ndarray
    combined_response: np.ndarray
    match_mask: np.ndarray
    full_points_grid: np.ndarray
    exp_points: np.ndarray
    exp_image_band_z: np.ndarray
    exp_intensity_z: np.ndarray
    exp_h5_band_z: np.ndarray
    line_segments: list[LineSegment]
    line_variant: LineVariant
    line_variant_score: float
    weights: MatchWeights
    label: str


@dataclass
class MatchResult:
    label: str
    score: float
    rotation: R
    convention_name: str
    detector_transform: np.ndarray
    prepared: PreparedPattern

    def to_json_dict(self) -> dict:
        return {
            "label": self.label,
            "score": float(self.score),
            "convention_name": self.convention_name,
            "rotation_quat_xyzw": self.rotation.as_quat().tolist(),
            "rotation_matrix": self.rotation.as_matrix().tolist(),
            "weights": asdict(self.prepared.weights),
            "match_points": int(len(self.prepared.exp_points)),
        }


class MasterSphere:
    def __init__(
        self,
        upper_intensity: np.ndarray,
        lower_intensity: np.ndarray,
        upper_band: np.ndarray,
        lower_band: np.ndarray,
    ):
        self.upper_intensity = upper_intensity.astype(np.float32)
        self.lower_intensity = lower_intensity.astype(np.float32)
        self.upper_band = upper_band.astype(np.float32)
        self.lower_band = lower_band.astype(np.float32)
        axis = np.linspace(-1.0, 1.0, self.upper_intensity.shape[0])
        self.upper_intensity_sampler = RegularGridInterpolator((axis, axis), self.upper_intensity, bounds_error=False, fill_value=0.0)
        self.lower_intensity_sampler = RegularGridInterpolator((axis, axis), self.lower_intensity, bounds_error=False, fill_value=0.0)
        self.upper_band_sampler = RegularGridInterpolator((axis, axis), self.upper_band, bounds_error=False, fill_value=0.0)
        self.lower_band_sampler = RegularGridInterpolator((axis, axis), self.lower_band, bounds_error=False, fill_value=0.0)

    @staticmethod
    def _sample(vectors: np.ndarray, upper_sampler: RegularGridInterpolator, lower_sampler: RegularGridInterpolator) -> np.ndarray:
        x = vectors[:, 0]
        y = vectors[:, 1]
        z = vectors[:, 2]
        sampled = np.zeros(len(vectors), dtype=np.float32)
        upper = z >= 0
        lower = ~upper
        if np.any(upper):
            xy_upper = np.column_stack(
                [
                    y[upper] / (1.0 + z[upper] + 1e-8),
                    x[upper] / (1.0 + z[upper] + 1e-8),
                ]
            )
            sampled[upper] = upper_sampler(xy_upper)
        if np.any(lower):
            xy_lower = np.column_stack(
                [
                    y[lower] / (1.0 - z[lower] + 1e-8),
                    x[lower] / (1.0 - z[lower] + 1e-8),
                ]
            )
            sampled[lower] = lower_sampler(xy_lower)
        return sampled

    def sample_intensity(self, vectors: np.ndarray) -> np.ndarray:
        return self._sample(vectors, self.upper_intensity_sampler, self.lower_intensity_sampler)

    def sample_band(self, vectors: np.ndarray) -> np.ndarray:
        return self._sample(vectors, self.upper_band_sampler, self.lower_band_sampler)


DETECTOR_CONVENTIONS: dict[str, np.ndarray] = {
    "identity": np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32),
    "flip_x": np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32),
    "flip_y": np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32),
    "flip_xy": np.array([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32),
    "swap_xy": np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32),
    "swap_xy_flip_x": np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32),
    "swap_xy_flip_y": np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32),
    "swap_xy_flip_xy": np.array([[0.0, -1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32),
}


def default_map_specs(data_dir: str | Path) -> dict[str, MapSpec]:
    data_dir = Path(data_dir)
    return {
        "area1_high": MapSpec(
            key="area1_high",
            label="Area 1 HighR",
            h5_group="20260512/Cu/Area 1/OIM Map 1HighR",
            up2_path=data_dir / "20260512_Cu_Area 1_OIM Map 1.up2",
            rows=195,
            cols=218,
        ),
        "area1_low": MapSpec(
            key="area1_low",
            label="Area 1 LowR",
            h5_group="20260512/Cu/Area 1/OIM Map 1LowR",
            up2_path=data_dir / "20260512_Cu_Area 1_OIM Map 2.up2",
            rows=195,
            cols=218,
        ),
        "area2_high": MapSpec(
            key="area2_high",
            label="Area 2 HighR",
            h5_group="20260512/Cu/Area 2/OIM Map 2HighR",
            up2_path=data_dir / "20260512_Cu_Area 2_OIM Map 1.up2",
            rows=159,
            cols=178,
        ),
        "area2_low": MapSpec(
            key="area2_low",
            label="Area 2 LowR",
            h5_group="20260512/Cu/Area 2/OIM Map 2LowR",
            up2_path=data_dir / "20260512_Cu_Area 2_OIM Map 2.up2",
            rows=159,
            cols=178,
        ),
    }


def scalar(group: h5py.Group, path: str) -> float:
    return float(np.asarray(group[path][()]).reshape(-1)[0])


def scalar_int(group: h5py.Group, path: str) -> int:
    return int(np.asarray(group[path][()]).reshape(-1)[0])


def read_ohp_header(map_group: h5py.Group) -> OHPHeader:
    header = map_group["EBSD/OHP/HEADER"]
    return OHPHeader(
        circle_size=scalar_int(header, "Circle Size"),
        max_band_count=scalar_int(header, "Maximum Band Count"),
        max_rho_fraction=scalar(header, "Maximum Rho Fraction"),
        max_band_width=scalar(header, "Maximum Band Width"),
        theta_step_size=scalar(header, "Theta Step Size"),
    )


def read_pattern_center(map_group: h5py.Group) -> tuple[float, float, float]:
    pc_group = map_group["EBSD/ANG/HEADER/Pattern Center Calibration"]
    return (
        scalar(pc_group, "X-Star"),
        scalar(pc_group, "Y-Star"),
        scalar(pc_group, "Z-Star"),
    )


def read_ang_record(map_group: h5py.Group, index: int) -> dict:
    record = map_group["EBSD/ANG/DATA/DATA"][index]
    out: dict[str, object] = {}
    for name in record.dtype.names or []:
        value = record[name]
        if isinstance(value, np.ndarray):
            out[name] = value.astype(float).tolist()
        elif np.issubdtype(np.asarray(value).dtype, np.integer):
            out[name] = int(value)
        else:
            out[name] = float(value)
    return out


def read_bands(map_group: h5py.Group, index: int) -> list[Band]:
    raw = np.asarray(map_group["EBSD/OHP/DATA/DATA"][index], dtype=np.float32).reshape(-1, 4)
    bands: list[Band] = []
    for rho_bin, theta_deg, width, intensity in raw:
        if not np.isfinite([rho_bin, theta_deg, width, intensity]).all():
            continue
        if intensity <= 0 or width <= 0:
            continue
        bands.append(Band(float(rho_bin), float(theta_deg), float(width), float(intensity)))
    bands.sort(key=lambda item: item.intensity, reverse=True)
    return bands


def read_pattern_bundle(h5_path: Path, map_spec: MapSpec, index: int) -> PatternBundle:
    stack = Up2Stack(map_spec.up2_path)
    if index < 0:
        index += len(stack)
    if index < 0 or index >= len(stack):
        raise IndexError(f"{map_spec.key} index {index} outside 0..{len(stack)-1}")
    if map_spec.rows * map_spec.cols != len(stack):
        raise ValueError(f"{map_spec.key}: rows*cols does not match UP2 count")

    with h5py.File(h5_path, "r") as f:
        map_group = f[map_spec.h5_group]
        pc = read_pattern_center(map_group)
        ohp_header = read_ohp_header(map_group)
        bands = read_bands(map_group, index)
        ang_record = read_ang_record(map_group, index)

    pattern = np.array(stack[index], copy=True)
    return PatternBundle(
        map_spec=map_spec,
        index=index,
        row=index // map_spec.cols,
        col=index % map_spec.cols,
        pattern_u16=pattern,
        pc=pc,
        bands=bands,
        ohp_header=ohp_header,
        ang_record=ang_record,
        up2_info=stack.info,
    )


def circular_mask(height: int, width: int, radius_fraction: float = 0.49) -> np.ndarray:
    yy, xx = np.ogrid[:height, :width]
    cy = (height - 1) / 2.0
    cx = (width - 1) / 2.0
    radius = min(height, width) * radius_fraction
    return ((yy - cy) ** 2 + (xx - cx) ** 2) <= radius**2


def zscore01(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    output = np.zeros_like(values, dtype=np.float32)
    masked_values = values[mask].astype(np.float32)
    if masked_values.size == 0:
        return output
    normalized = (masked_values - masked_values.mean()) / (masked_values.std() + 1e-8)
    normalized = np.clip(normalized, -2.5, 2.5)
    normalized = (normalized + 2.5) / 5.0
    output[mask] = normalized.astype(np.float32)
    return output


def zscore_vector(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32)
    return (values - values.mean()) / (values.std() + 1e-8)


def normalize_u16(pattern: np.ndarray) -> np.ndarray:
    return pattern.astype(np.float32) / 65535.0


def preprocess_pattern(
    image: np.ndarray,
    valid_mask: np.ndarray,
    mask_erosion: int,
    background_sigma: float,
    band_sigma_min: int,
    band_sigma_max: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    valid_mask = ndi.binary_erosion(valid_mask, iterations=mask_erosion)
    valid_mask = morphology.remove_small_holes(valid_mask, area_threshold=64)
    valid_mask = morphology.remove_small_objects(valid_mask, min_size=256)

    background = filters.gaussian(image, sigma=background_sigma)
    corrected = image - background
    corrected[~valid_mask] = 0.0
    corrected = exposure.rescale_intensity(corrected, in_range="image", out_range=(0.0, 1.0)).astype(np.float32)

    band_response = filters.meijering(
        corrected,
        sigmas=range(band_sigma_min, band_sigma_max + 1),
        black_ridges=False,
    )
    band_response = exposure.rescale_intensity(band_response, in_range="image", out_range=(0.0, 1.0)).astype(np.float32)
    band_response[~valid_mask] = 0.0

    return valid_mask, corrected, zscore01(corrected, valid_mask), zscore01(band_response, valid_mask)


def hough_rho_to_pixels(rho_bin: float, header: OHPHeader, height: int, width: int) -> float:
    center_bin = header.circle_size / 2.0
    # EDAX stores rho in the Hough image coordinate system whose center is
    # Circle Size / 2. Maximum Rho Fraction is a peak-search limit, not an
    # additional detector-coordinate scale factor.
    return (rho_bin - center_bin) * (min(height, width) / header.circle_size)


def coordinate_bounds(height: int, width: int, y_axis: str) -> tuple[float, float, float, float]:
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    x_min = -cx
    x_max = (width - 1) - cx
    if y_axis == "down":
        y_min = -cy
        y_max = (height - 1) - cy
    else:
        y_min = -((height - 1) - cy)
        y_max = cy
    return x_min, x_max, y_min, y_max


def coord_to_pixel(x: float, y: float, height: int, width: int, y_axis: str) -> tuple[float, float]:
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    col = x + cx
    row = y + cy if y_axis == "down" else cy - y
    return row, col


def line_segment_from_band(
    band: Band,
    header: OHPHeader,
    height: int,
    width: int,
    variant: LineVariant,
    band_index: int,
) -> LineSegment | None:
    rho_px = variant.rho_sign * hough_rho_to_pixels(band.rho_bin, header, height, width)
    normal_angle = band.theta_deg + (90.0 if variant.theta_is_line_angle else 0.0)
    theta = math.radians(normal_angle)
    c = math.cos(theta)
    s = math.sin(theta)
    x_min, x_max, y_min, y_max = coordinate_bounds(height, width, variant.y_axis)
    points: list[tuple[float, float]] = []

    if abs(s) > 1e-8:
        for x in (x_min, x_max):
            y = (rho_px - x * c) / s
            if y_min - 1e-5 <= y <= y_max + 1e-5:
                points.append((x, y))
    if abs(c) > 1e-8:
        for y in (y_min, y_max):
            x = (rho_px - y * s) / c
            if x_min - 1e-5 <= x <= x_max + 1e-5:
                points.append((x, y))

    unique: list[tuple[float, float]] = []
    for pt in points:
        if all((pt[0] - old[0]) ** 2 + (pt[1] - old[1]) ** 2 > 1e-4 for old in unique):
            unique.append(pt)
    if len(unique) < 2:
        return None

    if len(unique) > 2:
        best_pair = (unique[0], unique[1])
        best_dist = -1.0
        for i in range(len(unique)):
            for j in range(i + 1, len(unique)):
                dist = (unique[i][0] - unique[j][0]) ** 2 + (unique[i][1] - unique[j][1]) ** 2
                if dist > best_dist:
                    best_dist = dist
                    best_pair = (unique[i], unique[j])
        p0, p1 = best_pair
    else:
        p0, p1 = unique

    row0, col0 = coord_to_pixel(p0[0], p0[1], height, width, variant.y_axis)
    row1, col1 = coord_to_pixel(p1[0], p1[1], height, width, variant.y_axis)
    return LineSegment(band_index=band_index, band=band, row0=row0, col0=col0, row1=row1, col1=col1)


def line_variants() -> list[LineVariant]:
    variants: list[LineVariant] = []
    for theta_is_line_angle in (False, True):
        for rho_sign in (1.0, -1.0):
            for y_axis in ("down", "up"):
                name = f"{'line' if theta_is_line_angle else 'normal'}_theta_rho{'+' if rho_sign > 0 else '-'}_y{y_axis}"
                variants.append(LineVariant(name, theta_is_line_angle, rho_sign, y_axis))
    return variants


def rasterize_bands(
    bands: list[Band],
    header: OHPHeader,
    height: int,
    width: int,
    variant: LineVariant,
    valid_mask: np.ndarray | None = None,
    width_scale: float = 1.4,
) -> tuple[np.ndarray, list[LineSegment]]:
    canvas = np.zeros((height, width), dtype=np.float32)
    if not bands:
        return canvas, []

    intensities = np.array([band.intensity for band in bands], dtype=np.float32)
    intensity_min = float(intensities.min())
    intensity_span = float(intensities.max() - intensity_min + 1e-8)
    segments: list[LineSegment] = []

    for band_index, band in enumerate(bands):
        segment = line_segment_from_band(band, header, height, width, variant, band_index)
        if segment is None:
            continue
        segments.append(segment)
        weight = 0.35 + 0.65 * ((band.intensity - intensity_min) / intensity_span)
        thickness = int(np.clip(round(abs(band.width) * width_scale), 2, 12))
        cv2.line(
            canvas,
            (int(round(segment.col0)), int(round(segment.row0))),
            (int(round(segment.col1)), int(round(segment.row1))),
            float(weight),
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )

    if valid_mask is not None:
        canvas[~valid_mask] = 0.0
    if canvas.max() > 0:
        canvas /= float(canvas.max())
    return canvas.astype(np.float32), segments


def correlation_on_mask(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    av = a[mask].astype(np.float32)
    bv = b[mask].astype(np.float32)
    if av.size < 2 or av.std() < 1e-8 or bv.std() < 1e-8:
        return 0.0
    return float(np.mean(zscore_vector(av) * zscore_vector(bv)))


def choose_line_variant(
    bands: list[Band],
    header: OHPHeader,
    height: int,
    width: int,
    image_band_score: np.ndarray,
    valid_mask: np.ndarray,
    requested: str,
) -> tuple[LineVariant, float, np.ndarray, list[LineSegment], list[dict]]:
    variants = line_variants()
    if requested != "auto":
        matches = [variant for variant in variants if variant.name == requested]
        if not matches:
            valid_names = ", ".join(variant.name for variant in variants)
            raise ValueError(f"Unknown line variant '{requested}'. Valid values: auto, {valid_names}")
        variant = matches[0]
        line_mask, segments = rasterize_bands(bands, header, height, width, variant, valid_mask)
        score = correlation_on_mask(line_mask, image_band_score, valid_mask)
        return variant, score, line_mask, segments, [{"variant": variant.name, "score": score}]

    diagnostics = []
    best: tuple[LineVariant, float, np.ndarray, list[LineSegment]] | None = None
    for variant in variants:
        line_mask, segments = rasterize_bands(bands, header, height, width, variant, valid_mask)
        score = correlation_on_mask(line_mask, image_band_score, valid_mask)
        diagnostics.append({"variant": variant.name, "score": score, "segments": len(segments)})
        if best is None or score > best[1]:
            best = (variant, score, line_mask, segments)
    assert best is not None
    return (*best, diagnostics)


def detector_to_sphere_grid(height: int, width: int, pc: tuple[float, float, float]) -> np.ndarray:
    pcx, pcy, pcz = pc
    jj, ii = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    cx = pcx * (width - 1)
    cy = pcy * (height - 1)
    x = (jj - cx) / (pcz * height)
    y = -(ii - cy) / (pcz * height)
    z = np.ones_like(x)
    norm = np.sqrt(x**2 + y**2 + z**2) + 1e-8
    return np.stack([x / norm, y / norm, z / norm], axis=-1).astype(np.float32)


def detector_pixels_to_sphere(rows: np.ndarray, cols: np.ndarray, height: int, width: int, pc: tuple[float, float, float]) -> np.ndarray:
    pcx, pcy, pcz = pc
    cx = pcx * (width - 1)
    cy = pcy * (height - 1)
    x = (cols.astype(np.float32) - cx) / (pcz * height)
    y = -(rows.astype(np.float32) - cy) / (pcz * height)
    z = np.ones_like(x, dtype=np.float32)
    vectors = np.column_stack([x, y, z])
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8
    return vectors.astype(np.float32)


def make_match_mask(combined_response: np.ndarray, valid_mask: np.ndarray, quantile: float, top_k: int) -> np.ndarray:
    threshold = float(np.quantile(combined_response[valid_mask], quantile))
    match_mask = valid_mask & (combined_response >= threshold)
    if top_k > 0 and int(match_mask.sum()) > top_k:
        candidate_indices = np.flatnonzero(match_mask.ravel())
        candidate_values = combined_response.ravel()[candidate_indices]
        keep = np.argpartition(candidate_values, -top_k)[-top_k:]
        limited = np.zeros(match_mask.size, dtype=bool)
        limited[candidate_indices[keep]] = True
        match_mask = limited.reshape(match_mask.shape)
    return match_mask


def prepare_pattern(
    bundle: PatternBundle,
    weights: MatchWeights,
    label: str,
    mask_radius_fraction: float,
    mask_erosion: int,
    background_sigma: float,
    band_sigma_min: int,
    band_sigma_max: int,
    match_quantile: float,
    top_k_points: int,
    line_variant_name: str,
) -> tuple[PreparedPattern, list[dict]]:
    image = normalize_u16(bundle.pattern_u16)
    height, width = image.shape
    valid_mask = circular_mask(height, width, mask_radius_fraction)
    valid_mask, corrected, corrected_score, image_band_score = preprocess_pattern(
        image,
        valid_mask,
        mask_erosion=mask_erosion,
        background_sigma=background_sigma,
        band_sigma_min=band_sigma_min,
        band_sigma_max=band_sigma_max,
    )
    line_variant, line_score, h5_line_mask, segments, variant_diagnostics = choose_line_variant(
        bundle.bands,
        bundle.ohp_header,
        height,
        width,
        image_band_score,
        valid_mask,
        line_variant_name,
    )
    h5_band_score = zscore01(h5_line_mask, valid_mask)
    combined_response = (
        weights.intensity * corrected_score
        + weights.image_line * image_band_score
        + weights.h5_band * h5_band_score
    ).astype(np.float32)
    combined_response[~valid_mask] = 0.0

    match_mask = make_match_mask(combined_response, valid_mask, match_quantile, top_k_points)
    full_points_grid = detector_to_sphere_grid(height, width, bundle.pc)
    exp_points = full_points_grid[match_mask]
    prepared = PreparedPattern(
        bundle=bundle,
        image=image,
        valid_mask=valid_mask,
        corrected=corrected,
        image_band_score=image_band_score,
        corrected_score=corrected_score,
        h5_band_score=h5_band_score,
        h5_line_mask=h5_line_mask,
        combined_response=combined_response,
        match_mask=match_mask,
        full_points_grid=full_points_grid,
        exp_points=exp_points,
        exp_image_band_z=zscore_vector(image_band_score[match_mask]),
        exp_intensity_z=zscore_vector(corrected_score[match_mask]),
        exp_h5_band_z=zscore_vector(h5_band_score[match_mask]),
        line_segments=segments,
        line_variant=line_variant,
        line_variant_score=line_score,
        weights=weights,
        label=label,
    )
    return prepared, variant_diagnostics


def resolve_master_path(path: str | Path | None) -> Path:
    if path:
        master_path = Path(path)
        if master_path.exists():
            return master_path
        raise FileNotFoundError(f"Master sphere H5 not found: {master_path}")
    for candidate in MASTER_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Could not find ni_mc_mp_20kv_uint8_gzip_opts9.h5. Pass --master-h5 explicitly.")


def load_master_sphere(
    master_h5_path: Path,
    background_sigma: float,
    band_sigma_min: int,
    band_sigma_max: int,
) -> MasterSphere:
    with h5py.File(master_h5_path, "r") as f:
        upper = f["EMData/EBSDmaster/masterSPNH"][0].astype(np.float32)
        lower = f["EMData/EBSDmaster/masterSPSH"][0].astype(np.float32)
    if upper.max() > 1.5:
        upper /= 255.0
        lower /= 255.0

    upper_blur = filters.gaussian(upper, sigma=background_sigma)
    lower_blur = filters.gaussian(lower, sigma=background_sigma)
    upper_corrected = exposure.rescale_intensity(upper - upper_blur, in_range="image", out_range=(0.0, 1.0)).astype(np.float32)
    lower_corrected = exposure.rescale_intensity(lower - lower_blur, in_range="image", out_range=(0.0, 1.0)).astype(np.float32)
    upper_band = exposure.rescale_intensity(
        filters.meijering(upper_corrected, sigmas=range(band_sigma_min, band_sigma_max + 1), black_ridges=False),
        in_range="image",
        out_range=(0.0, 1.0),
    ).astype(np.float32)
    lower_band = exposure.rescale_intensity(
        filters.meijering(lower_corrected, sigmas=range(band_sigma_min, band_sigma_max + 1), black_ridges=False),
        in_range="image",
        out_range=(0.0, 1.0),
    ).astype(np.float32)
    return MasterSphere(upper_corrected, lower_corrected, upper_band, lower_band)


def score_rotation(rotation: R, points: np.ndarray, prepared: PreparedPattern, master: MasterSphere) -> float:
    rotated = rotation.apply(points)
    master_band_z = zscore_vector(master.sample_band(rotated))
    master_intensity_z = zscore_vector(master.sample_intensity(rotated))
    score = 0.0
    weights = prepared.weights
    if weights.image_line:
        score += weights.image_line * float(np.mean(prepared.exp_image_band_z * master_band_z))
    if weights.intensity:
        score += weights.intensity * float(np.mean(prepared.exp_intensity_z * master_intensity_z))
    if weights.h5_band:
        score += weights.h5_band * float(np.mean(prepared.exp_h5_band_z * master_band_z))
    return score


def iterable_progress(values: Iterable, desc: str):
    if tqdm is None:
        return values
    return tqdm(values, desc=desc, leave=False)


def match_to_master(
    prepared: PreparedPattern,
    master: MasterSphere,
    coarse_rotation_count: int,
    refine_schedule: list[tuple[float, int]],
    random_seed: int,
) -> MatchResult:
    rng = np.random.default_rng(random_seed)
    best_rotation: R | None = None
    best_score = -np.inf
    best_convention_name = ""
    best_transform = np.eye(3, dtype=np.float32)

    for convention_name, transform in iterable_progress(DETECTOR_CONVENTIONS.items(), f"{prepared.label}: conventions"):
        convention_points = prepared.exp_points @ transform.T
        coarse_rotations = R.random(coarse_rotation_count, random_state=rng)
        coarse_scores = np.array([score_rotation(rot, convention_points, prepared, master) for rot in coarse_rotations])
        rotation = coarse_rotations[int(np.argmax(coarse_scores))]
        convention_score = float(coarse_scores.max())

        for step_deg, attempts in refine_schedule:
            for _ in range(attempts):
                delta = R.from_euler("zyx", rng.normal(scale=step_deg, size=3), degrees=True)
                candidate = delta * rotation
                candidate_score = score_rotation(candidate, convention_points, prepared, master)
                if candidate_score > convention_score:
                    rotation = candidate
                    convention_score = candidate_score

        if convention_score > best_score:
            best_score = convention_score
            best_rotation = rotation
            best_convention_name = convention_name
            best_transform = transform

    if best_rotation is None:
        raise RuntimeError("No rotation was evaluated")
    return MatchResult(
        label=prepared.label,
        score=best_score,
        rotation=best_rotation,
        convention_name=best_convention_name,
        detector_transform=best_transform,
        prepared=prepared,
    )


def sphere_texture(master: MasterSphere, lon_count: int, colat_count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lon = np.linspace(-np.pi, np.pi, lon_count)
    colat = np.linspace(0.0, np.pi, colat_count)
    lon_grid, colat_grid = np.meshgrid(lon, colat)
    x = np.sin(colat_grid) * np.cos(lon_grid)
    y = np.sin(colat_grid) * np.sin(lon_grid)
    z = np.cos(colat_grid)
    vectors = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    texture = master.sample_intensity(vectors).reshape(colat_grid.shape)
    return texture, lon_grid, colat_grid, vectors


def project_to_equirect(
    vectors: np.ndarray,
    values: np.ndarray,
    lon_count: int,
    colat_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    theta = np.arctan2(vectors[:, 1], vectors[:, 0])
    phi = np.arccos(np.clip(vectors[:, 2], -1.0, 1.0))
    u = np.clip(np.round((theta + np.pi) / (2 * np.pi) * (lon_count - 1)).astype(int), 0, lon_count - 1)
    v = np.clip(np.round(phi / np.pi * (colat_count - 1)).astype(int), 0, colat_count - 1)
    projection_sum = np.zeros((colat_count, lon_count), dtype=np.float32)
    projection_count = np.zeros((colat_count, lon_count), dtype=np.float32)
    np.add.at(projection_sum, (v, u), values.astype(np.float32))
    np.add.at(projection_count, (v, u), 1.0)
    projection = np.zeros_like(projection_sum)
    mask = projection_count > 0
    projection[mask] = projection_sum[mask] / projection_count[mask]
    return projection, mask


def segment_curve_vectors(prepared: PreparedPattern, segment: LineSegment, samples: int = 220) -> np.ndarray:
    rows = np.linspace(segment.row0, segment.row1, samples, dtype=np.float32)
    cols = np.linspace(segment.col0, segment.col1, samples, dtype=np.float32)
    height, width = prepared.image.shape
    return detector_pixels_to_sphere(rows, cols, height, width, prepared.bundle.pc)


def save_detector_overlay(prepared: PreparedPattern, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    axes[0].imshow(prepared.bundle.pattern_u16, cmap="gray", vmin=int(prepared.bundle.pattern_u16.min()), vmax=int(prepared.bundle.pattern_u16.max()))
    axes[0].set_title("Raw UP2 pattern")
    axes[0].axis("off")

    axes[1].imshow(prepared.bundle.pattern_u16, cmap="gray", vmin=int(prepared.bundle.pattern_u16.min()), vmax=int(prepared.bundle.pattern_u16.max()))
    intensities = np.array([segment.band.intensity for segment in prepared.line_segments], dtype=np.float32)
    if intensities.size == 0:
        intensities = np.array([0.0, 1.0], dtype=np.float32)
    cmap = plt.get_cmap("turbo")
    denom = float(intensities.max() - intensities.min() + 1e-8)
    for i, segment in enumerate(prepared.line_segments):
        color = cmap((segment.band.intensity - float(intensities.min())) / denom)
        axes[1].plot([segment.col0, segment.col1], [segment.row0, segment.row1], color=color, linewidth=1.6)
        mid_col = 0.5 * (segment.col0 + segment.col1)
        mid_row = 0.5 * (segment.row0 + segment.row1)
        axes[1].text(mid_col, mid_row, str(i + 1), color="white", fontsize=7, ha="center", va="center")
    axes[1].set_title(f"H5 OHP bands ({prepared.line_variant.name})")
    axes[1].axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_channel_visualization(prepared_base: PreparedPattern, prepared_enhanced: PreparedPattern, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(13.5, 7.0))
    panels = [
        ("Raw UP2", prepared_enhanced.image),
        ("Valid circular mask", prepared_enhanced.valid_mask.astype(np.float32)),
        ("Image line response", prepared_enhanced.image_band_score),
        ("H5 band raster", prepared_enhanced.h5_band_score),
        ("Corrected intensity", prepared_enhanced.corrected_score),
        ("Baseline combined", prepared_base.combined_response),
        ("Band-enhanced combined", prepared_enhanced.combined_response),
        ("Enhanced match points", prepared_enhanced.match_mask.astype(np.float32)),
    ]
    for ax, (title, data) in zip(axes.ravel(), panels):
        ax.imshow(data, cmap="gray", vmin=0.0, vmax=1.0)
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_sphere_match_visualization(results: list[MatchResult], master: MasterSphere, out_path: Path, lon_count: int, colat_count: int) -> None:
    texture, _, _, _ = sphere_texture(master, lon_count, colat_count)
    fig, axes = plt.subplots(len(results), 2, figsize=(13.5, 5.2 * len(results)), squeeze=False)
    for row, result in enumerate(results):
        prepared = result.prepared
        full_points = prepared.full_points_grid.reshape(-1, 3) @ result.detector_transform.T
        matched_full = result.rotation.apply(full_points).reshape(prepared.full_points_grid.shape)
        valid_points = matched_full[prepared.valid_mask]

        raw_projection, projection_mask = project_to_equirect(
            valid_points,
            prepared.image[prepared.valid_mask],
            lon_count,
            colat_count,
        )

        axes[row, 0].imshow(texture, cmap="gray", origin="upper", extent=[-180, 180, 180, 0], aspect="auto")
        axes[row, 0].imshow(
            raw_projection,
            cmap="magma",
            origin="upper",
            extent=[-180, 180, 180, 0],
            aspect="auto",
            alpha=np.where(projection_mask, 0.82, 0.0),
        )
        axes[row, 0].set_title(f"{result.label}: raw patch on master sphere")
        axes[row, 0].set_xlabel("Longitude (deg)")
        axes[row, 0].set_ylabel("Colatitude (deg)")

        axes[row, 1].imshow(texture, cmap="gray", origin="upper", extent=[-180, 180, 180, 0], aspect="auto")
        colors = plt.get_cmap("turbo")(np.linspace(0.05, 0.95, max(1, len(prepared.line_segments))))
        for color, segment in zip(colors, prepared.line_segments):
            curve = segment_curve_vectors(prepared, segment)
            curve = curve @ result.detector_transform.T
            curve = result.rotation.apply(curve)
            lon = np.degrees(np.arctan2(curve[:, 1], curve[:, 0]))
            colat = np.degrees(np.arccos(np.clip(curve[:, 2], -1.0, 1.0)))
            jumps = np.where(np.abs(np.diff(lon)) > 180)[0] + 1
            for part in np.split(np.arange(len(lon)), jumps):
                if len(part) >= 2:
                    axes[row, 1].plot(lon[part], colat[part], color=color, linewidth=1.7)
        axes[row, 1].set_title(f"{result.label}: H5 bands after matching, score={result.score:.4f}")
        axes[row, 1].set_xlabel("Longitude (deg)")
        axes[row, 1].set_ylabel("Colatitude (deg)")
        axes[row, 1].set_xlim(-180, 180)
        axes[row, 1].set_ylim(180, 0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_score_comparison(results: list[MatchResult], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    labels = [result.label for result in results]
    scores = [result.score for result in results]
    bars = ax.bar(labels, scores, color=["#5975a4", "#4c9f70"][: len(results)])
    ax.set_ylabel("Weighted correlation score")
    ax.set_title("Image-only vs H5-band-enhanced matching")
    ax.grid(axis="y", alpha=0.25)
    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(), f"{score:.4f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def parse_refine_schedule(text: str) -> list[tuple[float, int]]:
    schedule: list[tuple[float, int]] = []
    if not text.strip():
        return schedule
    for item in text.split(","):
        step, attempts = item.split(":")
        schedule.append((float(step), int(attempts)))
    return schedule


def jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read EDAX H5 OHP Kikuchi-band positions, project UP2 patterns to a sphere, and enhance master-sphere matching with the H5 bands.",
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--h5", type=Path, default=DEFAULT_H5_PATH)
    parser.add_argument("--master-h5", type=Path, default=None)
    parser.add_argument("--map", choices=["area1_high", "area1_low", "area2_high", "area2_low"], default="area1_high")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--mask-radius-frac", type=float, default=0.49)
    parser.add_argument("--mask-erosion", type=int, default=6)
    parser.add_argument("--background-sigma", type=float, default=9.0)
    parser.add_argument("--band-sigma-min", type=int, default=1)
    parser.add_argument("--band-sigma-max", type=int, default=5)
    parser.add_argument("--match-quantile", type=float, default=0.82)
    parser.add_argument("--top-k-points", type=int, default=8000)
    parser.add_argument("--line-variant", default="auto", help="auto or one of the line variant names reported in summary.json")
    parser.add_argument("--coarse-rotations", type=int, default=450)
    parser.add_argument("--refine-schedule", default="8:180,3:240,1:240", help="Comma list of step_deg:attempts, e.g. 8:300,3:500,1:500")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sphere-lon-count", type=int, default=240)
    parser.add_argument("--sphere-colat-count", type=int, default=120)
    parser.add_argument("--baseline-image-line-weight", type=float, default=0.75)
    parser.add_argument("--baseline-intensity-weight", type=float, default=0.25)
    parser.add_argument("--enhanced-image-line-weight", type=float, default=0.45)
    parser.add_argument("--enhanced-intensity-weight", type=float, default=0.15)
    parser.add_argument("--enhanced-h5-band-weight", type=float, default=0.40)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    map_specs = default_map_specs(args.data_dir)
    map_spec = map_specs[args.map]
    master_h5 = resolve_master_path(args.master_h5)
    out_dir = args.out_dir / args.map / f"idx_{args.index:05d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading UP2/H5 pattern: {map_spec.label}, index={args.index}")
    bundle = read_pattern_bundle(args.h5, map_spec, args.index)
    print(f"Pattern shape: {bundle.pattern_u16.shape}; PC={bundle.pc}; H5 bands={len(bundle.bands)}")
    print(f"Master sphere: {master_h5}")

    baseline_weights = MatchWeights(
        image_line=args.baseline_image_line_weight,
        intensity=args.baseline_intensity_weight,
        h5_band=0.0,
    )
    enhanced_weights = MatchWeights(
        image_line=args.enhanced_image_line_weight,
        intensity=args.enhanced_intensity_weight,
        h5_band=args.enhanced_h5_band_weight,
    )
    base_prepared, variant_diagnostics = prepare_pattern(
        bundle=bundle,
        weights=baseline_weights,
        label="image-only",
        mask_radius_fraction=args.mask_radius_frac,
        mask_erosion=args.mask_erosion,
        background_sigma=args.background_sigma,
        band_sigma_min=args.band_sigma_min,
        band_sigma_max=args.band_sigma_max,
        match_quantile=args.match_quantile,
        top_k_points=args.top_k_points,
        line_variant_name=args.line_variant,
    )
    enhanced_prepared, _ = prepare_pattern(
        bundle=bundle,
        weights=enhanced_weights,
        label="H5-band-enhanced",
        mask_radius_fraction=args.mask_radius_frac,
        mask_erosion=args.mask_erosion,
        background_sigma=args.background_sigma,
        band_sigma_min=args.band_sigma_min,
        band_sigma_max=args.band_sigma_max,
        match_quantile=args.match_quantile,
        top_k_points=args.top_k_points,
        line_variant_name=base_prepared.line_variant.name,
    )

    save_detector_overlay(enhanced_prepared, out_dir / "01_raw_pattern_with_h5_bands.png")
    save_channel_visualization(base_prepared, enhanced_prepared, out_dir / "02_matching_channels.png")

    master = load_master_sphere(
        master_h5,
        background_sigma=args.background_sigma,
        band_sigma_min=args.band_sigma_min,
        band_sigma_max=args.band_sigma_max,
    )
    refine_schedule = parse_refine_schedule(args.refine_schedule)

    print("Matching image-only baseline...")
    base_result = match_to_master(
        base_prepared,
        master,
        coarse_rotation_count=args.coarse_rotations,
        refine_schedule=refine_schedule,
        random_seed=args.seed,
    )
    print("Matching H5-band-enhanced...")
    enhanced_result = match_to_master(
        enhanced_prepared,
        master,
        coarse_rotation_count=args.coarse_rotations,
        refine_schedule=refine_schedule,
        random_seed=args.seed,
    )
    results = [base_result, enhanced_result]

    save_sphere_match_visualization(results, master, out_dir / "03_master_sphere_matches.png", args.sphere_lon_count, args.sphere_colat_count)
    save_score_comparison(results, out_dir / "04_score_comparison.png")

    summary = {
        "map": jsonable(asdict(map_spec)),
        "index": int(bundle.index),
        "row": int(bundle.row),
        "col": int(bundle.col),
        "up2_info": bundle.up2_info.to_json_dict(),
        "pattern_shape": list(bundle.pattern_u16.shape),
        "pattern_center": {"pcx": bundle.pc[0], "pcy": bundle.pc[1], "pcz": bundle.pc[2]},
        "ohp_header": asdict(bundle.ohp_header),
        "bands": [asdict(band) for band in bundle.bands],
        "ang_record": bundle.ang_record,
        "line_variant": asdict(enhanced_prepared.line_variant),
        "line_variant_score": enhanced_prepared.line_variant_score,
        "line_variant_diagnostics": variant_diagnostics,
        "match_results": [result.to_json_dict() for result in results],
        "outputs": {
            "detector_overlay": str(out_dir / "01_raw_pattern_with_h5_bands.png"),
            "channels": str(out_dir / "02_matching_channels.png"),
            "sphere_matches": str(out_dir / "03_master_sphere_matches.png"),
            "score_comparison": str(out_dir / "04_score_comparison.png"),
        },
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Saved visualizations and summary to: {out_dir}")
    for result in results:
        print(f"{result.label}: score={result.score:.4f}, convention={result.convention_name}, match_points={len(result.prepared.exp_points)}")


if __name__ == "__main__":
    main()
