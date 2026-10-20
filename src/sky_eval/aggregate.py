"""Equal-image statistics and predeclared per-metric shared intersections."""
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
        output[dataset]["complete_predictions"] = sum(r.get("prediction_sha256") is not None for r in subset)
        output[dataset]["failure_reasons"] = dict(Counter(
            reason for row in subset for reason in row.get("failure_reason", "").split(";") if reason))
    return output


def compare(models: dict[str, list[dict[str, Any]]], comparable_amplitude: bool) -> list[dict[str, Any]]:
    """Intersection rule is fixed before scoring and applied separately per metric."""
    sample_sets = [{r["sample_id"] for r in rows} for rows in models.values()]
    if not sample_sets or any(s != sample_sets[0] for s in sample_sets):
        raise ValueError("Model sample sets differ from frozen manifest")
    output = []
    datasets = sorted({r["dataset"] for rows in models.values() for r in rows})
    for dataset in datasets:
        by_model = {model: {r["sample_id"]: r for r in rows if r["dataset"] == dataset}
                    for model, rows in models.items()}
        for metric in METRICS:
            shared = set.intersection(*[{sid for sid, row in rows.items() if row.get(metric) is not None}
                                        for rows in by_model.values()])
            for model, rows in by_model.items():
                allowed = metric == "sfoe" or comparable_amplitude
                stats = summary([rows[sid][metric] for sid in sorted(shared)] if allowed else [], len(rows))
                output.append(dict(model_id=model, dataset=dataset, metric=metric, **stats,
                                   sample_ids=sorted(shared) if allowed else [],
                                   status="ok" if allowed else "incomparable_representations"))
    return output
