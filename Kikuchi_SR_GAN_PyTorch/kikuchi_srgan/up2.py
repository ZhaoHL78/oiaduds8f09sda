from __future__ import annotations

import struct
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


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
    """Lazy memmap reader for a UP2 pattern stack."""

    def __init__(self, path: str | Path):
        self.info = read_up2_info(path)
        self._data: np.memmap | None = None

    def __len__(self) -> int:
        return self.info.count

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.info.shape

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

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_data"] = None
        return state
