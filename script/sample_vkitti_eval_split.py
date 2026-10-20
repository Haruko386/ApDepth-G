"""Build a deterministic 540-image subset of the existing VKITTI val split.

Reads only existing split text files; no images, weights, or network access.
Only val supplies sample candidates. Train is read solely to audit overlap.
Existing train/validation lists are left untouched.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re

REPO = Path(__file__).resolve().parents[1]
KEY = re.compile(r"(?:^|/)(Scene\d+)/([^/]+)/frames/rgb/(Camera_\d+)/rgb_(\d+)\.[^/]+$")
FrameKey = tuple[str, str, str, int]


def repo_path(value: str) -> Path:
    path = Path(value)
    resolved = (REPO / path).resolve() if not path.is_absolute() else path.resolve()
    if any(p.casefold() == "apdepth-g-paper" for p in resolved.parts):
        raise ValueError("Protected directory: ApDepth-G-paper")
    return resolved


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path: Path) -> dict[FrameKey, str]:
    rows: dict[FrameKey, str] = {}
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        columns = line.split()
        match = KEY.search(columns[0].replace("\\", "/"))
        if match is None or len(columns) < 2:
            raise ValueError(f"Expected VKITTI RGB/depth row: {path}:{number}")
        scene, variation, camera, frame = match.groups()
        key = scene, variation, camera, int(frame)
        if key in rows:
            raise ValueError(f"Duplicate frame key: {path}:{number}: {key}")
        rows[key] = line
    if not rows:
        raise ValueError(f"Empty split: {path}")
    return rows


def spread(frames: list[int], count: int, offset: int = 0) -> list[int]:
    """Choose midpoint quantiles without randomness, then optionally phase-shift."""
    if count < 1 or count > len(frames):
        raise ValueError(f"Cannot select {count} unique frame positions from {len(frames)}")
    indices = [((2 * i + 1) * len(frames) // (2 * count) + offset) % len(frames) for i in range(count)]
    return sorted(frames[index] for index in indices)


def validation_subset(rows: dict[FrameKey, str], count: int) -> dict[FrameKey, str]:
    """Balance original validation sequences and spread frames over each one."""
    groups: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for scene, variation, camera, frame in rows:
        groups[(scene, variation, camera)].append(frame)
    if count % len(groups):
        raise ValueError("Sample count must divide evenly across original validation sequences")
    quota = count // len(groups)
    selected: dict[FrameKey, str] = {}
    for number, (group, frames) in enumerate(sorted(groups.items())):
        ordered = sorted(frames)
        offset = number * len(ordered) // len(groups)
        for frame in spread(ordered, quota, offset):
            key = (*group, frame)
            selected[key] = rows[key]
    return dict(sorted(selected.items()))


def describe(rows: dict[FrameKey, str], train: dict[FrameKey, str], label: str) -> dict[str, object]:
    scene_overlap = sorted({key[0] for key in rows} & {key[0] for key in train})
    gaps: dict[str, int | None] = {}
    for scene in sorted({key[0] for key in rows}):
        frames = sorted({key[3] for key in rows if key[0] == scene})
        gaps[scene] = min((b - a for a, b in zip(frames, frames[1:])), default=None)
    groups = Counter("/".join(key[:3]) for key in rows)
    return dict(label=label, count=len(rows), scene_counts=dict(sorted(Counter(k[0] for k in rows).items())),
                variation_counts=dict(sorted(Counter(k[1] for k in rows).items())),
                camera_counts=dict(sorted(Counter(k[2] for k in rows).items())), sequence_counts=dict(sorted(groups.items())),
                train_exact_overlap_count=len(rows.keys() & train.keys()), train_scene_overlap=scene_overlap,
                unique_scene_timestamps=len({(k[0], k[3]) for k in rows}),
                min_timestamp_gap_per_scene=gaps,
                independence="seen_scene_diagnostic" if scene_overlap else "repository_train_scene_disjoint; target checkpoint history still requires verification")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-list", default="data_split/vkitti/vkitti_train.txt")
    parser.add_argument("--val-list", default="data_split/vkitti/vkitti_val.txt")
    parser.add_argument("--output-dir", default="data_split/vkitti")
    parser.add_argument("--count", type=int, default=540)
    args = parser.parse_args()
    train_path, val_path, output = repo_path(args.train_list), repo_path(args.val_list), repo_path(args.output_dir)
    if args.count < 1:
        parser.error("--count must be positive")
    train_digest, val_digest = digest(train_path), digest(val_path)
    train, val = read_rows(train_path), read_rows(val_path)
    if train.keys() & val.keys():
        raise ValueError("Source train/val exact frame overlap; audit input lists before sampling")
    if {k[0] for k in train} & {k[0] for k in val}:
        raise ValueError("Original validation scenes overlap training; cannot label a sampled subset held-out")
    sampled = validation_subset(val, args.count)
    products = {
        f"vkitti_val_sampled_{args.count}.txt": sampled,
    }
    report_path = output / f"vkitti_sampled_{args.count}_audit.json"
    if report_path.exists() or any((output / name).exists() for name in products):
        raise FileExistsError("Sampled output already exists; choose a new --output-dir")
    if digest(train_path) != train_digest or digest(val_path) != val_digest:
        raise ValueError("Source lists changed during sampling")
    output.mkdir(parents=True, exist_ok=True)
    audit: dict[str, object] = dict(
        protocol="custom VKITTI val-only sparse sequence sampling v1; not the official KITTI Eigen split",
        selection_uses="only original val scene/variation/camera/frame keys; no RGB, masks, depth predictions or scores",
        training_list_role="overlap audit only; never a sampling source",
        sampling_rule="original val only; equal scene/variation/camera quotas; midpoint-quantile sampling with deterministic per-sequence phase offsets",
        sources={str(train_path.relative_to(REPO)) if train_path.is_relative_to(REPO) else str(train_path): train_digest,
                 str(val_path.relative_to(REPO)) if val_path.is_relative_to(REPO) else str(val_path): val_digest}, products={})
    reports: dict[str, object] = {}
    for name, rows in products.items():
        if len(rows) != args.count or len(set(rows.values())) != args.count:
            raise ValueError("Wrong sample count or duplicate source lines")
        path = output / name
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write("\n".join(rows.values()) + "\n")
        reports[name] = {**describe(rows, train, name), "sha256": digest(path)}
        print(json.dumps(reports[name], ensure_ascii=False))
    audit["products"] = reports
    with report_path.open("x", encoding="utf-8") as stream:
        json.dump(audit, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(f"Audit written: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
