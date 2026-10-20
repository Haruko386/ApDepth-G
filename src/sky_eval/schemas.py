"""Sky metric settings and file helpers."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import hashlib
import json


@dataclass(frozen=True)
class Protocol:
    name: str = "SCOE"
    version: str = "1.0"
    min_sky_fraction: float = 0.05
    sky_interior_erosion_fraction: float = 0.01
    boundary_band_fraction: float = 0.02
    boundary_inner_offset_fraction: float = 0.01
    min_boundary_pairs: int = 50
    boundary_stride: int = 2
    percentile_method: str = "linear"
    aggregation: str = "per_image_macro_mean"
    allow_scene_leakage: bool = False
    max_failure_fraction: float = 0.0

    def __post_init__(self) -> None:
        if self.name != "SCOE" or self.version != "1.0":
            raise ValueError("Unsupported SCOE protocol version")
        if self.min_sky_fraction != 0.05 or self.boundary_band_fraction != 0.02:
            raise ValueError("SCOE v1 fixes min_sky_fraction=0.05 and boundary_band_fraction=0.02")
        if not 0 < self.sky_interior_erosion_fraction <= 1:
            raise ValueError("Invalid erosion fraction")
        if not 0 < self.boundary_inner_offset_fraction <= self.boundary_band_fraction:
            raise ValueError("Invalid boundary inner offset")
        if self.min_boundary_pairs < 1 or self.boundary_stride < 1:
            raise ValueError("Pair count and stride must be positive")
        if self.percentile_method != "linear" or self.aggregation != "per_image_macro_mean":
            raise ValueError("SCOE v1 uses linear percentiles and equal-image means")
        if not 0 <= self.max_failure_fraction <= 1:
            raise ValueError("Invalid max_failure_fraction")


def confined(root: Path, relative: str) -> Path:
    """Reject path traversal and accidental absolute paths."""
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"Expected portable relative path: {relative}")
    path = (root / candidate).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Path escapes root: {relative}")
    if any(p.casefold() == "apdepth-g-paper" for p in path.parts):
        raise ValueError("Protected directory: ApDepth-G-paper")
    return path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def object_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode("utf-8")).hexdigest()


def write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
