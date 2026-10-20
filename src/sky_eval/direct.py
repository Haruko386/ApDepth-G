"""Evaluate depth predictions with sky masks stored beside a dataset."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from PIL import Image
from tabulate import tabulate

from src.dataset import get_pred_name

from .aggregate import aggregate
from .masks import build_regions
from .metrics import METRICS, score
from .schemas import Protocol


def mask_root(base_data_dir: str, dataset_dir: str, mask_name: str) -> Path:
    if not mask_name or Path(mask_name).name != mask_name or mask_name in (".", ".."):
        raise ValueError("--sky_mask_name must be one folder name")
    source = Path(base_data_dir) / dataset_dir
    return (source if source.is_dir() else source.parent) / "sky_masks" / mask_name


def mask_path(root: Path, rgb_relative_path: str) -> Path:
    relative = Path(rgb_relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Invalid RGB path: {rgb_relative_path}")
    return root / f"{relative}.sky.png"


def select_masked_files(dataset, root: Path) -> tuple[int, int]:
    """Use the requested split's images that already have sky masks."""
    original_count = len(dataset.filenames)
    dataset.filenames = [row for row in dataset.filenames if row and mask_path(root, row[0]).is_file()]
    if not dataset.filenames:
        raise FileNotFoundError(f"No sky masks for this split under {root}")
    return len(dataset.filenames), original_count - len(dataset.filenames)


def evaluate_sky(dataset, root: Path, prediction_dir: str, output_dir: str) -> Path:
    protocol = Protocol()
    rows = []
    missing_predictions = 0
    invalid_predictions = 0
    for entry in dataset.filenames:
        rgb = entry[0]
        prediction = Path(prediction_dir) / Path(rgb).parent / get_pred_name(
            Path(rgb).name, dataset.name_mode, suffix=".npy"
        )
        if not prediction.is_file():
            missing_predictions += 1
            continue
        try:
            with Image.open(mask_path(root, rgb)) as image:
                sky = np.asarray(image.convert("L")) > 0
            depth = np.load(prediction, allow_pickle=False)
            result = score(depth, build_regions(sky, protocol), protocol)
        except (OSError, ValueError, TypeError) as exc:
            invalid_predictions += 1
            print(f"Skipping {rgb}: {exc}")
            continue
        rows.append({"dataset": dataset.disp_name, "rgb": rgb, **result})

    if not rows:
        raise ValueError("No image has both a usable sky mask and depth prediction")
    stats = aggregate(rows)[dataset.disp_name]
    lines = [
        "Sky evaluation metrics:",
        f"of predictions: {prediction_dir}",
        f"on dataset: {dataset.disp_name}",
        f"with samples in: {dataset.filename_ls_path}",
        f"sky masks: {root}",
        f"evaluated images: {len(rows)}",
        f"missing predictions: {missing_predictions}",
        f"invalid predictions: {invalid_predictions}",
        "mask source: generated sky segmentation",
        "",
        tabulate(
            [[name, stats[name]["mean"], stats[name]["N"]] for name in METRICS],
            headers=["metric", "mean", "N"],
        ),
    ]
    os.makedirs(output_dir, exist_ok=True)
    result_path = Path(output_dir) / "eval_metrics-sky.txt"
    result_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result_path
