"""Explicit HRWSI manifests and official VKITTI2 RGB class labels."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any
import json
import re

import numpy as np
from PIL import Image

from .masks import BoolArray
from .schemas import Sample, confined, sha256
from .rgb_io import normalized_relative, read_rgb

VKITTI_KEY = re.compile(r"(Scene\d+)/([^/]+)/frames/rgb/(Camera_\d+)/rgb_(\d+)\.[^/\s]+")


def read_split(path: Path) -> list[tuple[str, str, str, str]]:
    keys: list[tuple[str, str, str, str]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        match = VKITTI_KEY.search(line.split()[0].replace("\\", "/"))
        if match is None:
            raise ValueError(f"Invalid VKITTI RGB key: {path}:{number}")
        keys.append(tuple(match.groups()))
    if len(keys) != len(set(keys)):
        raise ValueError(f"Duplicate VKITTI keys: {path}")
    if not keys:
        raise ValueError(f"Empty split: {path}")
    return keys


def read_colors(path: Path) -> dict[str, tuple[int, int, int]]:
    colors: dict[str, tuple[int, int, int]] = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        parts = line.split()
        if not parts or parts[0].startswith("#") or parts[0].lower() == "category":
            continue
        if len(parts) != 4:
            raise ValueError(f"Malformed class palette: {path}: {line}")
        color = tuple(int(v) for v in parts[1:])
        if any(v < 0 or v > 255 for v in color) or parts[0] in colors:
            raise ValueError(f"Invalid/duplicate palette entry: {line}")
        colors[parts[0]] = color
    if colors.get("Sky") != (90, 200, 255):
        raise ValueError(f"VKITTI2 Sky RGB must be verified against official (90,200,255): {path}")
    if len(set(colors.values())) != len(colors):
        raise ValueError(f"Ambiguous class palette: {path}")
    return colors


def rgb_root(config: dict[str, Any]) -> Path:
    return Path(config.get("rgb_root") or config["root"])


def mask_root(config: dict[str, Any]) -> Path:
    return Path(config.get("mask_root") or config.get("class_seg_root") or config["root"])


def sample_rgb(sample: Sample, config: dict[str, Any]) -> Image.Image:
    return read_rgb(rgb_root(config), sample.rgb_path)[0]


def load_mask(sample: Sample, config: dict[str, Any]) -> BoolArray:
    path = confined(mask_root(config), sample.sky_mask_path)
    with Image.open(path) as image:
        if sample.dataset == "vkitti2" and config.get("mask_format") != "binary_manifest":
            palette = read_colors(confined(mask_root(config), str(sample.colors_path)))
            rgb = np.asarray(image.convert("RGB"))
            packed = (rgb[..., 0].astype(np.uint32) << 16) | (rgb[..., 1].astype(np.uint32) << 8) | rgb[..., 2]
            known = [(r << 16) | (g << 8) | b for r, g, b in palette.values()]
            unknown = np.setdiff1d(np.unique(packed), known)
            if len(unknown):
                raise ValueError(f"unknown_semantic_colors: {path}: {unknown[:10].tolist()}")
            return np.all(rgb == palette["Sky"], axis=-1)
        values = np.asarray(image)
    if values.ndim != 2:
        raise ValueError(f"Expected single-channel semantic mask: {path}")
    if "sky_values" not in config or "non_sky_values" not in config:
        raise ValueError("Explicit sky_values and non_sky_values required; validity is not sky")
    sky_values, other_values = config["sky_values"], config["non_sky_values"]
    if not sky_values or set(sky_values) & set(other_values):
        raise ValueError("Ambiguous semantic mask values")
    if not np.isin(values, sky_values + other_values).all():
        raise ValueError(f"unknown_mask_values: {path}")
    return np.isin(values, sky_values)


def index_dataset(name: str, config: dict[str, Any]) -> list[Sample]:
    if config.get("mask_format") == "binary_manifest":
        if name not in ("vkitti2", "kitti", "diode"):
            raise ValueError(f"Unsupported pseudo-label dataset: {name}")
        for key in ("rgb_root", "mask_root", "manifest", "segmentation_report"):
            if not config.get(key):
                raise ValueError(f"Missing pseudo-label configuration: {key}")
        manifest = Path(config["manifest"])
        report = json.loads(Path(config["segmentation_report"]).read_text(encoding="utf-8"))
        if report["dataset"] != name or report["failed"] or report["succeeded"] != report["requested"]:
            raise ValueError("Incomplete or mismatched segmentation run; do not silently omit failed images")
        if sha256(manifest) != report["manifest_sha256"]:
            raise ValueError("Segmentation manifest changed after generation")
        split_path = Path(config.get("data_split") or report["split_list"])
        if sha256(split_path) != report["split_list_sha256"]:
            raise ValueError("data_split changed after mask generation; regenerate masks for the new split")
        expected_paths = [normalized_relative(line.split()[0]) for line in
                          split_path.read_text(encoding="utf-8-sig").splitlines()
                          if line.strip() and not line.lstrip().startswith("#")]
        if len(expected_paths) != report["total_split"] or len(expected_paths) != len(set(expected_paths)):
            raise ValueError("data_split size mismatch or duplicate RGB paths")
        if report["requested"] > len(expected_paths) or (not report.get("limited") and report["requested"] != len(expected_paths)):
            raise ValueError("Incomplete split coverage in segmentation report")
        rows = [Sample(**json.loads(line)) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(rows) != report["requested"] or len({s.sample_id for s in rows}) != len(rows):
            raise ValueError("Segmentation sample count mismatch or duplicate IDs")
        if any(s.dataset != name or s.mask_source != "mask2former_pseudo_semantic_sky" for s in rows):
            raise ValueError("Pseudo-label manifests must retain their explicit Mask2Former source")
        if [s.rgb_path for s in rows] != expected_paths[:report["requested"]]:
            raise ValueError("Segmentation manifest does not match data_split paths/order")
        return rows
    if name == "vkitti2":
        for key in ("rgb_root", "class_seg_root", "test_list"):
            if not config.get(key):
                raise ValueError(f"Missing VKITTI2 {key}; semantic labels are required")
        for key in ("rgb_root", "class_seg_root"):
            if not Path(config[key]).is_dir():
                raise ValueError(f"VKITTI2 directory does not exist: {config[key]}")
        keys = read_split(Path(config["test_list"]))
        rows = []
        for scene, variation, camera, frame in keys:
            stem = f"{scene}/{variation}"
            rows.append(Sample(
                sample_id=f"vkitti2/{scene}/{variation}/{camera}/{frame}", dataset=name,
                split=config.get("split", "unverified_evaluation_subset"),
                rgb_path=f"{stem}/frames/rgb/{camera}/rgb_{frame}.jpg",
                sky_mask_path=f"{stem}/frames/classSegmentation/{camera}/classgt_{frame}.png",
                colors_path=f"{stem}/colors.txt", mask_source="official_class_segmentation",
                split_provenance=config.get("split_provenance", "pending"),
                scene_id=scene, variation=variation, camera=camera, frame_id=frame))
        return rows
    if name not in ("hrwsi", "synthetic"):
        raise ValueError(f"Unsupported dataset: {name}")
    for key in ("root", "manifest", "sky_values", "non_sky_values"):
        if not config.get(key):
            raise ValueError(f"Missing {name} {key}; inspect the actual semantic annotation package first")
    rows = []
    for line in Path(config["manifest"]).read_text(encoding="utf-8-sig").splitlines():
        if line.strip():
            sample = Sample(**json.loads(line))
            if sample.dataset != name or not sample.mask_source or not sample.split_provenance:
                raise ValueError("Manifest dataset/provenance mismatch")
            if any(term in sample.mask_source.lower() for term in ("validity", "vgc", "da2")):
                raise ValueError("Validity/DA2/VGC masks are not semantic sky annotations")
            rows.append(sample)
    if not rows or len({s.sample_id for s in rows}) != len(rows):
        raise ValueError("Empty manifest or duplicate sample_id")
    return rows


def inspect_sample(sample: Sample, config: dict[str, Any]) -> tuple[Sample, BoolArray]:
    image, rgb_digest = read_rgb(rgb_root(config), sample.rgb_path)
    width, height = image.size
    sky = load_mask(sample, config)
    if sky.shape != (height, width):
        raise ValueError("rgb_mask_shape_mismatch")
    mask_path = confined(mask_root(config), sample.sky_mask_path)
    colors_path = confined(mask_root(config), sample.colors_path) if sample.colors_path else None
    mask_digest = sha256(mask_path)
    if config.get("mask_format") == "binary_manifest":
        if rgb_digest != sample.rgb_sha256 or mask_digest != sample.mask_sha256:
            raise ValueError("RGB/pseudo-mask content changed after segmentation")
    return replace(sample, width=width, height=height, sky_ratio=float(sky.mean()),
                   rgb_sha256=rgb_digest, mask_sha256=mask_digest,
                   colors_sha256=sha256(colors_path) if colors_path else None), sky
