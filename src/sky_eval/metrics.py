"""Pure SCOE metrics; no GT alignment, clipping, or model-dependent masks."""
from __future__ import annotations

from typing import Any
import numpy as np
from numpy.typing import NDArray

from .masks import Regions
from .schemas import Protocol

METRICS = ("sky_iqr", "sky_p95_p5", "sfoe")


def row_medians(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Avoid overflow from summing two very large finite middle elements."""
    ordered = np.sort(values, axis=1)
    mid = ordered.shape[1] // 2
    if ordered.shape[1] % 2:
        return ordered[:, mid]
    return ordered[:, mid - 1] / 2 + ordered[:, mid] / 2


def score(depth: NDArray[np.floating], regions: Regions, protocol: Protocol) -> dict[str, Any]:
    result: dict[str, Any] = dict.fromkeys((*METRICS, "reference_range"))
    reasons: list[str] = []
    sky = regions.sky
    result.update(n_sky_pixels=int(sky.sum()), n_sky_interior_pixels=int(regions.interior.sum()),
                  n_boundary_pairs=regions.n_pairs)
    if depth.ndim != 2 or depth.shape != sky.shape:
        raise ValueError("prediction_shape_mismatch")
    if not np.isfinite(depth).all():
        raise ValueError("nonfinite_prediction")
    d = depth.astype(np.float64)
    if sky.mean() < protocol.min_sky_fraction:
        reasons.append("insufficient_sky_area")
    elif sky.all():
        reasons.append("no_foreground")
    else:
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            ref = np.percentile(d[~sky], [5, 95], method=protocol.percentile_method)
            dynamic = float(ref[1] - ref[0])
            result["reference_range"] = dynamic if np.isfinite(dynamic) else None
            if dynamic <= 0 or not np.isfinite(dynamic):
                reasons.append("degenerate_foreground_range")
            if not regions.interior.any():
                reasons.append("insufficient_sky_interior")
            elif dynamic > 0 and np.isfinite(dynamic):
                q = np.percentile(d[regions.interior], [5, 25, 75, 95],
                                  method=protocol.percentile_method)
                for key, value in (("sky_iqr", (q[2] - q[1]) / dynamic),
                                   ("sky_p95_p5", (q[3] - q[0]) / dynamic)):
                    if np.isfinite(value):
                        result[key] = float(value)
                    else:
                        reasons.append("nonfinite_" + key)
        if regions.n_pairs < protocol.min_boundary_pairs:
            reasons.append("insufficient_boundary_pairs")
        else:
            sky_depth = row_medians(d.ravel()[regions.sky_indices])
            foreground_depth = row_medians(d.ravel()[regions.foreground_indices])
            result["sfoe"] = float(np.mean(sky_depth <= foreground_depth))
    result["failure_reason"] = ";".join(reasons)
    result["status"] = "ok" if not reasons else "partial" if any(result[m] is not None for m in METRICS) else "failed"
    return result
