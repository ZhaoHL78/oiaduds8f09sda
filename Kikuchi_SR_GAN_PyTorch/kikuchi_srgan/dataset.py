from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .mask import circular_mask
from .up2 import Up2Stack


@dataclass(frozen=True)
class PairSpec:
    name: str
    low_path: Path
    high_path: Path
    rows: int | None = None
    cols: int | None = None

    def to_json_dict(self) -> dict:
        data = asdict(self)
        data["low_path"] = str(self.low_path)
        data["high_path"] = str(self.high_path)
        return data


def default_pair_specs(data_dir: str | Path) -> list[PairSpec]:
    data_dir = Path(data_dir)
    return [
        PairSpec(
            name="Area 1",
            low_path=data_dir / "20260512_Cu_Area 1_OIM Map 2.up2",
            high_path=data_dir / "20260512_Cu_Area 1_OIM Map 1.up2",
            rows=195,
            cols=218,
        ),
        PairSpec(
            name="Area 2",
            low_path=data_dir / "20260512_Cu_Area 2_OIM Map 2.up2",
            high_path=data_dir / "20260512_Cu_Area 2_OIM Map 1.up2",
            rows=159,
            cols=178,
        ),
    ]


class KikuchiPairDataset(Dataset):
    """One-to-one LowR/HighR Kikuchi pairs from the raw UP2 stacks.

    No contrast, filtering, cropping, or preprocessing is applied. The only
    numeric conversion is uint16 -> float32 in [0, 1] for PyTorch training.
    """

    def __init__(
        self,
        data_dir: str | Path,
        pairs: list[PairSpec] | None = None,
        use_mask: bool = True,
        mask_radius_fraction: float = 0.49,
    ):
        self.data_dir = Path(data_dir)
        self.pairs = pairs or default_pair_specs(self.data_dir)
        self.use_mask = use_mask
        self.mask_radius_fraction = mask_radius_fraction
        self.low_stacks = [Up2Stack(spec.low_path) for spec in self.pairs]
        self.high_stacks = [Up2Stack(spec.high_path) for spec in self.pairs]

        counts: list[int] = []
        for spec, low, high in zip(self.pairs, self.low_stacks, self.high_stacks):
            if len(low) != len(high):
                raise ValueError(f"{spec.name}: LowR count {len(low)} != HighR count {len(high)}")
            if spec.rows is not None and spec.cols is not None and spec.rows * spec.cols != len(low):
                raise ValueError(
                    f"{spec.name}: rows*cols={spec.rows * spec.cols} does not match count {len(low)}"
                )
            counts.append(len(low))

        self.counts = counts
        self.offsets = np.cumsum([0] + counts).astype(np.int64)
        low_h, low_w = self.low_shape
        high_h, high_w = self.high_shape
        self.low_mask_np = circular_mask(low_h, low_w, self.mask_radius_fraction)
        self.high_mask_np = circular_mask(high_h, high_w, self.mask_radius_fraction)
        self.low_mask = torch.from_numpy(self.low_mask_np).unsqueeze(0)
        self.high_mask = torch.from_numpy(self.high_mask_np).unsqueeze(0)

    def __len__(self) -> int:
        return int(self.offsets[-1])

    @property
    def low_shape(self) -> tuple[int, int]:
        info = self.low_stacks[0].info
        return (info.height, info.width)

    @property
    def high_shape(self) -> tuple[int, int]:
        info = self.high_stacks[0].info
        return (info.height, info.width)

    def locate(self, index: int) -> tuple[int, int]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        pair_idx = int(np.searchsorted(self.offsets, index, side="right") - 1)
        local_idx = int(index - self.offsets[pair_idx])
        return pair_idx, local_idx

    def metadata(self, index: int) -> dict:
        pair_idx, local_idx = self.locate(index)
        spec = self.pairs[pair_idx]
        row = None
        col = None
        if spec.cols:
            row = local_idx // spec.cols
            col = local_idx % spec.cols
        return {
            "global_index": int(index),
            "pair_index": pair_idx,
            "area": spec.name,
            "local_index": local_idx,
            "row": row,
            "col": col,
        }

    def preview_indices(self, count: int = 6) -> list[int]:
        if count <= 0:
            return []
        indices = set()
        for start, stop in zip(self.offsets[:-1], self.offsets[1:]):
            indices.update([int(start), int((start + stop - 1) // 2), int(stop - 1)])
        ordered = sorted(indices)
        if len(ordered) <= count:
            return ordered
        positions = np.linspace(0, len(ordered) - 1, count).round().astype(int)
        return [ordered[i] for i in positions]

    def manifest(self) -> dict:
        return {
            "data_dir": str(self.data_dir),
            "total_pairs": len(self),
            "low_shape": self.low_shape,
            "high_shape": self.high_shape,
            "use_mask": self.use_mask,
            "mask_radius_fraction": self.mask_radius_fraction,
            "low_mask_valid_pixels": int(self.low_mask_np.sum()),
            "high_mask_valid_pixels": int(self.high_mask_np.sum()),
            "pairs": [
                {
                    **spec.to_json_dict(),
                    "count": count,
                    "low_up2": low.info.to_json_dict(),
                    "high_up2": high.info.to_json_dict(),
                }
                for spec, count, low, high in zip(
                    self.pairs, self.counts, self.low_stacks, self.high_stacks
                )
            ],
        }

    def get_pair(self, index: int, apply_mask: bool | None = None) -> dict[str, torch.Tensor]:
        if apply_mask is None:
            apply_mask = self.use_mask
        pair_idx, local_idx = self.locate(int(index))
        low = np.array(self.low_stacks[pair_idx][local_idx], copy=True)
        high = np.array(self.high_stacks[pair_idx][local_idx], copy=True)

        low_t = torch.from_numpy(low.astype(np.float32, copy=False)).unsqueeze(0).div_(65535.0)
        high_t = torch.from_numpy(high.astype(np.float32, copy=False)).unsqueeze(0).div_(65535.0)
        if apply_mask:
            low_t = low_t * self.low_mask
            high_t = high_t * self.high_mask
        return {
            "lr": low_t,
            "hr": high_t,
            "lr_mask": self.low_mask,
            "hr_mask": self.high_mask,
        }

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.get_pair(index, apply_mask=self.use_mask)
