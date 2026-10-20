"""Strict raw-numpy adapters for existing run.py and infer.py outputs."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .schemas import Sample, confined


def validate_model(model: dict[str, Any]) -> None:
    required = ("root", "layout", "representation", "distance_direction", "semantics_source",
                "checkpoint", "inference_command", "git_commit", "seed", "resolution",
                "denoise_steps", "ensemble_size", "scheduler", "inference_config", "native_postprocessing")
    missing = [k for k in required if k not in model]
    if missing:
        raise ValueError(f"Missing prediction metadata: {missing}")
    for key in ("root", "representation", "semantics_source", "checkpoint", "inference_command",
                "git_commit", "resolution", "scheduler", "inference_config", "native_postprocessing"):
        if not model[key]:
            raise ValueError(f"Empty prediction metadata: {key}")
    if not model["semantics_source"] or not model["representation"]:
        raise ValueError("Output representation and semantics evidence are required")
    if model["distance_direction"] not in ("larger_is_farther", "smaller_is_farther"):
        raise ValueError("Unknown distance_direction")
    if model.get("geometry", "full_frame") != "full_frame":
        raise ValueError("Cropped predictions cannot be stretched into full-frame SCOE inputs")
    if model.get("resampling", "reject") != "reject":
        raise ValueError("v1 requires full-resolution predictions; resampling is not implemented")
    if model["layout"] not in ("run", "infer_vkitti", "infer_split", "explicit"):
        raise ValueError("Unknown prediction layout")
    if model["layout"] == "explicit" and not isinstance(model.get("files"), dict):
        raise ValueError("explicit layout requires sample_id -> relative .npy files")


def prediction_path(model: dict[str, Any], sample: Sample) -> Path:
    rgb = Path(sample.rgb_path)
    if model["layout"] == "run":
        relative = f"depth_npy/{rgb.stem}_pred.npy"
    elif model["layout"] == "infer_vkitti":
        if sample.dataset != "vkitti2":
            raise ValueError("infer_vkitti layout is only valid for VKITTI id naming")
        relative = str(rgb.parent / f"pred_{rgb.stem}.npy")
    elif model["layout"] == "infer_split":
        if sample.dataset not in ("vkitti2", "kitti", "diode"):
            raise ValueError("infer_split supports VKITTI2, KITTI and DIODE id-mode outputs")
        relative = str(Path(sample.dataset) / rgb.parent / f"pred_{rgb.stem}.npy")
    else:
        if sample.sample_id not in model["files"]:
            raise ValueError("missing_prediction_mapping")
        relative = model["files"][sample.sample_id]
    path = confined(Path(model["root"]), relative)
    if path.suffix.lower() != ".npy":
        raise ValueError("Raw float .npy predictions required")
    return path


def read_prediction(path: Path, model: dict[str, Any], shape: tuple[int, int]) -> tuple[NDArray[np.float64], str]:
    if not path.is_file():
        raise ValueError(f"missing_prediction: {path}")
    values = np.load(path, allow_pickle=False)
    if not isinstance(values, np.ndarray):
        values.close()
        raise ValueError("Expected a single .npy array, not an archive")
    if values.dtype not in (np.dtype("float32"), np.dtype("float64")):
        raise ValueError(f"prediction_dtype_not_float32_or_float64: {values.dtype}")
    dtype = str(values.dtype)
    if values.ndim == 3 and values.shape[0] == 1 and model.get("squeeze_leading_channel", False):
        values = values[0]
    if values.ndim != 2 or values.shape != shape:
        raise ValueError(f"prediction_shape_mismatch: {values.shape} != {shape}")
    if not np.isfinite(values).all():
        raise ValueError("nonfinite_prediction")
    values = values.astype(np.float64)
    if model["distance_direction"] == "smaller_is_farther":
        values = -values  # Direction only: this is NOT a reciprocal depth conversion.
    return values, dtype
