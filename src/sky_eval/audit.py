"""Leakage audit and data-only sample selection, independent of predictions."""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
import json

from .datasets import index_dataset, inspect_sample, read_split
from .schemas import Protocol, Sample, sha256


def split_audit(train_list: Path, test_list: Path) -> dict[str, Any]:
    train, test = read_split(train_list), read_split(test_list)
    train_set, test_set = set(train), set(test)
    scenes = sorted({k[0] for k in train} & {k[0] for k in test})
    exact = sorted(train_set & test_set)
    return {
        "train_count": len(train), "test_count": len(test),
        "train_scenes": dict(sorted(Counter(k[0] for k in train).items())),
        "test_scenes": dict(sorted(Counter(k[0] for k in test).items())),
        "train_variations": dict(sorted(Counter(k[1] for k in train).items())),
        "test_variations": dict(sorted(Counter(k[1] for k in test).items())),
        "scene_overlap": scenes, "exact_overlap_count": len(exact),
        "exact_overlap": [list(k) for k in exact],
        "train_list_sha256": sha256(train_list), "test_list_sha256": sha256(test_list),
        "status": "BLOCKED" if scenes else "scene_disjoint_lists_only",
    }


def audit_datasets(config: dict[str, Any], protocol: Protocol,
                   limit: int | None = None) -> tuple[dict[str, Any], list[Sample], list[dict[str, Any]]]:
    report: dict[str, Any] = {"datasets": {}, "blockers": [], "limited": limit is not None,
                              "duplicate_rgb": [], "duplicate_masks": [],
                              "near_duplicate_review": "pending; byte hashes do not detect near duplicates"}
    selected: list[Sample] = []
    stats: list[dict[str, Any]] = []
    rgb_hashes: dict[str, list[str]] = defaultdict(list)
    mask_hashes: dict[str, list[str]] = defaultdict(list)
    for name, cfg in config["datasets"].items():
        if not cfg.get("enabled", True):
            continue
        item: dict[str, Any] = {"status": "pending", "total": 0, "inspected": 0, "selected": 0,
                                "failures": [], "source": cfg}
        report["datasets"][name] = item
        if cfg.get("mask_format") == "binary_manifest":
            # Keep the original official-label SCOE table separate from this user-requested diagnostic protocol.
            report["blockers"].append(f"{name}: Mask2Former pseudo labels; diagnostic protocol, not official sky GT")
        if name == "vkitti2":
            try:
                item["split_audit"] = split_audit(Path(cfg["train_list"]), Path(cfg["test_list"]))
                if item["split_audit"]["scene_overlap"]:
                    report["blockers"].append(f"{name}: seen_scene {item['split_audit']['scene_overlap']}")
            except (OSError, ValueError, KeyError, TypeError) as exc:
                report["blockers"].append(f"{name}: split_audit: {exc}")
        try:
            rows = index_dataset(name, cfg)
            item["total"] = len(rows)
            if cfg.get("mask_format") == "binary_manifest":
                item["input_files"] = {cfg[key]: sha256(Path(cfg[key])) for key in ("manifest", "segmentation_report")}
                segmentation = json.loads(Path(cfg["segmentation_report"]).read_text(encoding="utf-8"))
                split_path = cfg.get("data_split") or segmentation["split_list"]
                item["input_files"][str(split_path)] = sha256(Path(split_path))
                item["segmentation"] = segmentation
                if segmentation.get("limited"):
                    report["blockers"].append(f"{name}: limited segmentation subset")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            item["failures"].append({"failure_reason": str(exc)})
            item["status"] = "BLOCKED"
            report["blockers"].append(f"{name}: dataset_unavailable: {exc}")
            continue
        for sample in rows[:limit]:
            item["inspected"] += 1
            try:
                sample, _ = inspect_sample(sample, cfg)
                rgb_hashes[sample.rgb_sha256].append(sample.sample_id)
                mask_hashes[sample.mask_sha256].append(sample.sample_id)
                eligible = sample.sky_ratio >= protocol.min_sky_fraction and sample.sky_ratio < 1
                stats.append({**sample.to_dict(), "selected": eligible,
                              "failure_reason": "" if eligible else "no_foreground" if sample.sky_ratio == 1 else "insufficient_sky_area"})
                if eligible:
                    selected.append(sample)
                    item["selected"] += 1
            except (OSError, ValueError, KeyError, TypeError) as exc:
                failure = {"sample_id": sample.sample_id, "failure_reason": str(exc)}
                item["failures"].append(failure)
                stats.append({**sample.to_dict(), "selected": False, "failure_reason": str(exc)})
        item["status"] = "BLOCKED" if item["failures"] else "inspected"
        if item["failures"]:
            report["blockers"].append(f"{name}: {len(item['failures'])} data failures")
        if not item["selected"]:
            report["blockers"].append(f"{name}: no eligible images")
        if not cfg.get("split_provenance") or not cfg.get("mask_provenance"):
            report["blockers"].append(f"{name}: missing split/mask provenance")
        if not cfg.get("qa_reviewed", False):
            report["blockers"].append(f"{name}: manual mask QA pending")
        if name == "hrwsi" and cfg.get("qa_review_count", 0) < 50:
            report["blockers"].append("hrwsi: fewer than 50 manually reviewed masks")
    report["duplicate_rgb"] = [v for v in rgb_hashes.values() if len(v) > 1]
    report["duplicate_masks"] = [v for v in mask_hashes.values() if len(v) > 1]
    if report["duplicate_rgb"]:
        report["blockers"].append("duplicate RGB content; freeze a deduplicated input manifest first")
    if not report["datasets"]:
        report["blockers"].append("no enabled datasets")
    if limit is not None:
        report["blockers"].append("limited smoke subset, not a complete dataset audit")
    if "synthetic" in report["datasets"]:
        report["blockers"].append("synthetic fixture; not real experiment results")
    if len(selected) != len({s.sample_id for s in selected}):
        raise ValueError("Duplicate sample_id across datasets")
    report["status"] = "BLOCKED" if report["blockers"] else "data_audit_passed"
    return report, selected, stats


def audit_markdown(report: dict[str, Any]) -> str:
    lines = ["# SCOE 数据审计", "", f"状态：{report['status']}", "",
             "| Dataset | 总数 | 已检查 | >=5% 且有非天空 | 数据失败 |", "|---|---:|---:|---:|---:|"]
    for name, item in report["datasets"].items():
        lines.append(f"| {name} | {item['total']} | {item['inspected']} | {item['selected']} | {len(item['failures'])} |")
        if "split_audit" in item:
            audit = item["split_audit"]
            lines.extend(["", f"{name} 场景交集：{audit['scene_overlap']}；精确帧交集：{audit['exact_overlap_count']}。",
                          "名单无交集不自动证明指定 checkpoint 的训练历史。"])
    lines.extend(["", "阻塞项：", ""] + [f"- {v}" for v in report["blockers"]])
    lines.extend(["", f"重复 RGB 组数：{len(report['duplicate_rgb'])}；重复 mask 组数：{len(report['duplicate_masks'])}。",
                  "相同 mask 不代表 RGB 重复；哈希仅检测字节重复，近重复人工审查仍待完成。"])
    return "\n".join(lines) + "\n"
