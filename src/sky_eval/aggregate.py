"""Equal-image sky metric statistics."""
from __future__ import annotations

from collections import Counter
from typing import Any
import numpy as np

from .metrics import METRICS


def summary(values: list[float], total: int) -> dict[str, Any]:
    if not values:
        return dict(N=0, failures=total, mean=None, std=None, median=None, p25=None, p75=None)
    a = np.asarray(values, dtype=np.float64)
    return dict(N=len(a), failures=total - len(a), mean=float(a.mean()),
                std=float(a.std(ddof=0)), median=float(np.median(a)),
                p25=float(np.percentile(a, 25)), p75=float(np.percentile(a, 75)))


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for dataset in sorted({r["dataset"] for r in rows}):
        subset = [r for r in rows if r["dataset"] == dataset]
        output[dataset] = {metric: summary([r[metric] for r in subset if r.get(metric) is not None], len(subset))
                           for metric in (*METRICS, "reference_range")}
        output[dataset]["total"] = len(subset)
        output[dataset]["failure_reasons"] = dict(Counter(
            reason for row in subset for reason in row.get("failure_reason", "").split(";") if reason))
    return output
