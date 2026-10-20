"""Model-independent Euclidean regions and deterministic normal samples."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage

from .schemas import Protocol

BoolArray = NDArray[np.bool_]
FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass
class Regions:
    sky: BoolArray
    interior: BoolArray
    sky_band: BoolArray
    foreground_band: BoolArray
    points: FloatArray
    normals: FloatArray
    sky_indices: IntArray
    foreground_indices: IntArray

    @property
    def n_pairs(self) -> int:
        return len(self.points)

    def save(self, path: Path) -> None:
        with path.open("xb") as stream:
            np.savez_compressed(stream, **self.__dict__)

    @classmethod
    def load(cls, path: Path) -> Regions:
        with np.load(path, allow_pickle=False) as data:
            return cls(**{key: data[key] for key in cls.__dataclass_fields__})


def build_regions(sky: BoolArray, protocol: Protocol) -> Regions:
    """Image exterior is ignored; only in-image opposite-class pixels define EDT.

    Boundary points are all 4-neighbour class-transition midpoints, sorted by
    (y,x), then every boundary_stride-th point. Normals use central differences
    of foreground-positive signed EDT. Every rounded half-ray must stay in its
    own class (including the unsampled inner gap), preventing hole crossings.
    """
    if sky.ndim != 2 or sky.dtype != np.bool_ or min(sky.shape) < 2:
        raise ValueError("Expected a boolean HxW mask with both dimensions >=2")
    h, w = sky.shape
    length = min(h, w)
    erosion = math.ceil(protocol.sky_interior_erosion_fraction * length)
    inner = math.ceil(protocol.boundary_inner_offset_fraction * length)
    radius = math.ceil(protocol.boundary_band_fraction * length)
    slots = radius - inner + 1
    empty = Regions(sky.copy(), sky.copy(), np.zeros_like(sky), np.zeros_like(sky),
                    np.empty((0, 2)), np.empty((0, 2)),
                    np.empty((0, slots), dtype=np.int64), np.empty((0, slots), dtype=np.int64))
    if not sky.any() or sky.all():
        return empty
    ds = ndimage.distance_transform_edt(sky)
    df = ndimage.distance_transform_edt(~sky)
    empty.interior = sky & (ds >= erosion)
    empty.sky_band = sky & (ds <= radius)
    empty.foreground_band = ~sky & (df <= radius)
    signed = df - ds
    gy, gx = np.gradient(signed)
    vertical = np.argwhere(sky[:-1] != sky[1:]).astype(np.float64)
    horizontal = np.argwhere(sky[:, :-1] != sky[:, 1:]).astype(np.float64)
    vertical[:, 0] += 0.5
    horizontal[:, 1] += 0.5
    points = np.concatenate((vertical, horizontal))
    points = points[np.lexsort((points[:, 1], points[:, 0]))][::protocol.boundary_stride]
    normals = np.column_stack([ndimage.map_coordinates(g, points.T, order=1)
                               for g in (gy, gx)])
    norms = np.linalg.norm(normals, axis=1)
    good = norms > 1e-12
    points, normals, norms = points[good], normals[good], norms[good]
    normals /= norms[:, None]
    # Require the normal to point across the actual transition edge, not along it.
    vertical_edge = points[:, 0] % 1 != 0
    axis = np.where(vertical_edge, 0, 1)
    lo = np.floor(points).astype(int)
    edge_sign = np.where(sky[lo[:, 0], lo[:, 1]], 1, -1)
    stable = normals[np.arange(len(points)), axis] * edge_sign >= 0.5
    points, normals = points[stable], normals[stable]
    indices: list[IntArray] = []
    valid = np.ones(len(points), dtype=bool)
    distances = np.arange(0.5, radius + 0.01, 0.5)
    for sign, expected in ((-1, True), (1, False)):
        coordinates = points[:, None, :] + sign * distances[None, :, None] * normals[:, None, :]
        lower = np.floor(coordinates)
        rounded = np.floor(coordinates + 0.5)
        ties = coordinates - lower == 0.5
        # Half-pixel ties round toward the boundary midpoint, symmetrically on
        # both sides. Always rounding upward would shift foreground rays by 1px.
        rounded = np.where(ties & (coordinates > points[:, None, :]), lower, rounded)
        ray = rounded.astype(np.int64)
        inside = ((ray[..., 0] >= 0) & (ray[..., 0] < h)
                  & (ray[..., 1] >= 0) & (ray[..., 1] < w))
        flat = np.clip(ray[..., 0], 0, h - 1) * w + np.clip(ray[..., 1], 0, w - 1)
        classes = sky.ravel()[flat] == expected
        valid &= (inside & classes).all(axis=1)
        chosen = np.arange(inner, radius + 1) * 2 - 1
        selected = flat[:, chosen]
        # Keep the discrete samples in the declared Euclidean candidate bands.
        selected_distances = (ds if expected else df).ravel()[selected]
        valid &= ((selected_distances >= inner) & (selected_distances <= radius)).all(axis=1)
        indices.append(selected)
    empty.points = points[valid]
    empty.normals = normals[valid]
    empty.sky_indices = indices[0][valid]
    empty.foreground_indices = indices[1][valid]
    return empty
