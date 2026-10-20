"""Frozen, non-overwriting audit -> prepare -> evaluate -> report workflow."""
from __future__ import annotations

from dataclasses import asdict
from collections import Counter
from importlib import metadata
from pathlib import Path
from typing import Any
import csv
import json
import logging
import platform
import re
import subprocess

import yaml

from .aggregate import aggregate, compare
from .audit import audit_datasets, audit_markdown, split_audit
from .datasets import inspect_sample, sample_rgb
from .masks import Regions, build_regions
from .metrics import METRICS, score
from .predictions import prediction_path, read_prediction, validate_model
from .schemas import Protocol, Sample, confined, object_hash, sha256, write_json
from .visuals import mask_diagnostic, prediction_diagnostic

REPO = Path(__file__).resolve().parents[2]


def safe_absolute(value: str) -> Path:
    path = Path(value)
    path = (REPO / path).resolve() if not path.is_absolute() else path.resolve()
    if any(p.casefold() == "apdepth-g-paper" for p in path.parts):
        raise ValueError("Protected directory: ApDepth-G-paper")
    return path


def load_config(path: Path, run_id: str | None) -> tuple[dict[str, Any], Protocol, Path]:
    config = yaml.safe_load(safe_absolute(str(path)).read_text(encoding="utf-8-sig"))
    if not isinstance(config, dict) or not isinstance(config.get("datasets"), dict):
        raise ValueError("Configuration requires a datasets mapping")
    protocol = Protocol(**config.get("protocol", {}))
    config["protocol"] = asdict(protocol)
    config.setdefault("models", {})
    for section in config["datasets"].values():
        for key in ("root", "rgb_root", "mask_root", "class_seg_root", "manifest", "segmentation_report", "data_split", "train_list", "test_list"):
            if section.get(key):
                section[key] = str(safe_absolute(section[key]))
    for model in config["models"].values():
        if model.get("root"):
            model["root"] = str(safe_absolute(model["root"]))
        evidence = model.get("training_evidence", {})
        for key in ("vkitti_train_list", "training_record"):
            if evidence.get(key):
                evidence[key] = str(safe_absolute(evidence[key]))
    output = config.setdefault("output", {})
    output["root"] = str(safe_absolute(output.get("root", "output/sky_eval")))
    output["run_id"] = run_id or output.get("run_id", "scoe_v1")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", output["run_id"]) or output["run_id"] in (".", ".."):
        raise ValueError("Invalid run_id")
    for model_id in config["models"]:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", model_id) or model_id in (".", ".."):
            raise ValueError("Invalid model_id")
    return config, protocol, confined(Path(output["root"]), output["run_id"])


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v
                             for k, v in row.items()})


def write_text(path: Path, text: str) -> None:
    with path.open("x", encoding="utf-8") as stream:
        stream.write(text)


def source_hashes() -> dict[str, str]:
    paths = sorted((REPO / "src/sky_eval").glob("*.py")) + [REPO / "script/sky_eval.py"]
    return {str(p.relative_to(REPO)).replace("\\", "/"): sha256(p) for p in paths}


def git_output(*args: str) -> str:
    result = subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True,
                            encoding="utf-8", errors="replace", check=False)
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def audit(config: dict[str, Any], protocol: Protocol, run: Path, limit: int | None) -> int:
    if run.exists():
        raise ValueError(f"Output exists; use a new run_id: {run}")
    run.mkdir(parents=True, exist_ok=False)
    write_json(run / "config_resolved.json", config)
    write_text(run / "config_resolved.yaml", yaml.safe_dump(config, allow_unicode=True, sort_keys=False))
    versions: dict[str, Any] = {"python": platform.python_version(), "platform": platform.platform()}
    for package in ("numpy", "scipy", "Pillow", "PyYAML", "torch"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "not installed"
    write_json(run / "environment.json", versions)
    write_json(run / "repository_state.json", {"commit": git_output("rev-parse", "HEAD"),
               "branch": git_output("branch", "--show-current"),
               "status": git_output("status", "--short", "--", ".", ":!ApDepth-G-paper"),
               "source_hashes": source_hashes()})
    result, selected, stats = audit_datasets(config, protocol, limit)
    write_json(run / "dataset_audit.json", result)
    write_text(run / "dataset_audit.md", audit_markdown(result))
    write_csv(run / "mask_stats.csv", stats)
    write_text(run / "candidate_manifest.jsonl", "".join(json.dumps(s.to_dict(), ensure_ascii=False) + "\n" for s in selected))
    write_json(run / "audit_lock.json", {name: sha256(run / name) for name in
               ("config_resolved.json", "dataset_audit.json", "candidate_manifest.jsonl", "mask_stats.csv", "repository_state.json")})
    logging.info("Audit: %s, eligible images=%d; details: %s", result["status"], len(selected), run)
    return 2 if result["blockers"] else 0


def verify_config(config: dict[str, Any], run: Path) -> None:
    for relative, expected in read_json(run / "audit_lock.json").items():
        if sha256(confined(run, relative)) != expected:
            raise ValueError(f"Audit artifact changed: {relative}")
    if config != read_json(run / "config_resolved.json"):
        raise ValueError("Configuration changed after audit; use a new run_id")
    if source_hashes() != read_json(run / "repository_state.json")["source_hashes"]:
        raise ValueError("SCOE source changed after audit; use a new run_id")
    for name, item in read_json(run / "dataset_audit.json")["datasets"].items():
        for path, expected in item.get("input_files", {}).items():
            if sha256(safe_absolute(path)) != expected:
                raise ValueError(f"Segmentation input changed after audit: {path}")
        if "split_audit" in item:
            for key in ("train_list", "test_list"):
                if sha256(Path(config["datasets"][name][key])) != item["split_audit"][key + "_sha256"]:
                    raise ValueError(f"Split list changed after audit: {name}/{key}")


def read_manifest(path: Path) -> list[Sample]:
    return [Sample(**json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def cache_key(sample: Sample) -> str:
    return object_hash(sample.sample_id)


def prepare(config: dict[str, Any], protocol: Protocol, run: Path) -> int:
    verify_config(config, run)
    result = read_json(run / "dataset_audit.json")
    if any(d["failures"] for d in result["datasets"].values()):
        raise ValueError("Cannot prepare: unresolved data failures; see dataset_audit.json")
    if any(d.get("split_audit", {}).get("scene_overlap") for d in result["datasets"].values()) and not protocol.allow_scene_leakage:
        raise ValueError("Cannot prepare seen-scene dataset with allow_scene_leakage=false")
    samples = read_manifest(run / "candidate_manifest.jsonl")
    if not samples:
        raise ValueError("Cannot prepare an empty manifest")
    if result["duplicate_rgb"]:
        raise ValueError("Duplicate RGB images require input-manifest deduplication before prepare")
    cache = run / "sample_pairs"
    cache.mkdir(exist_ok=False)
    diagnostics = run / "mask_diagnostics"
    diagnostics.mkdir(exist_ok=False)
    coverage = []
    qa_counts: dict[str, int] = {}
    dataset_counts = Counter(s.dataset for s in samples)
    for sample in samples:
        ds = config["datasets"][sample.dataset]
        actual, sky = inspect_sample(sample, ds)
        if actual != sample:
            raise ValueError(f"Dataset changed after audit: {sample.sample_id}")
        regions = build_regions(sky, protocol)
        regions.save(cache / f"{cache_key(sample)}.npz")
        coverage.append({"sample_id": sample.sample_id, "dataset": sample.dataset,
                         "n_boundary_pairs": regions.n_pairs, "n_sky_interior_pixels": int(regions.interior.sum()),
                         "sfoe_eligible": regions.n_pairs >= protocol.min_boundary_pairs})
        # Spread QA selection over the frozen list, not just the first sequence.
        ordinal = qa_counts.get(sample.dataset, 0)
        qa_counts[sample.dataset] = ordinal + 1
        stride = max(1, dataset_counts[sample.dataset] // 50)
        if ordinal % stride == 0 and ordinal // stride < 50:
            mask_diagnostic(sample_rgb(sample, ds), regions,
                            diagnostics / f"{cache_key(sample)}.png")
    write_csv(run / "pair_coverage.csv", coverage)
    write_text(run / "manifest.jsonl", "".join(json.dumps(s.to_dict(), ensure_ascii=False) + "\n" for s in samples))
    assets = [run / name for name in ("config_resolved.json", "dataset_audit.json", "manifest.jsonl", "pair_coverage.csv")]
    assets += sorted(cache.glob("*.npz"))
    write_json(run / "protocol_lock.json", {"config_hash": object_hash(config), "source_hashes": source_hashes(),
               "files": {str(p.relative_to(run)).replace("\\", "/"): sha256(p) for p in assets},
               "intersection_rule": "per-dataset, per-metric intersection of finite valid scores across ALL configured models",
               "bootstrap": "not requested; no independent-frame confidence interval reported"})
    logging.info("Prepared %d images; boundary-pair retention %d/%d", len(samples),
                 sum(r["sfoe_eligible"] for r in coverage), len(coverage))
    return 0


def verify_lock(config: dict[str, Any], run: Path) -> dict[str, Any]:
    verify_config(config, run)
    lock = read_json(run / "protocol_lock.json")
    if lock["config_hash"] != object_hash(config) or lock["source_hashes"] != source_hashes():
        raise ValueError("Protocol lock mismatch")
    for relative, expected in lock["files"].items():
        if sha256(confined(run, relative)) != expected:
            raise ValueError(f"Frozen artifact changed: {relative}")
    return lock


def evaluate(config: dict[str, Any], protocol: Protocol, run: Path, model_id: str) -> int:
    verify_lock(config, run)
    if model_id not in config["models"]:
        raise ValueError(f"Unknown model_id: {model_id}")
    model = config["models"][model_id]
    validate_model(model)
    samples = read_manifest(run / "manifest.jsonl")
    mapped: dict[Path, str] = {}
    for sample in samples:
        try:
            path = prediction_path(model, sample)
        except ValueError:
            continue  # Written as a per-image failure below.
        if path in mapped:
            raise ValueError(f"Prediction naming collision: {mapped[path]} and {sample.sample_id}; use explicit layout")
        mapped[path] = sample.sample_id
    destination = run / "models" / model_id
    destination.mkdir(parents=True, exist_ok=False)
    rows: list[dict[str, Any]] = []
    for sample in samples:
        row: dict[str, Any] = {**sample.to_dict(), "model_id": model_id, **dict.fromkeys(METRICS),
                               "reference_range": None, "prediction_sha256": None, "prediction_dtype": None,
                               "prediction_path": None}
        regions = Regions.load(run / "sample_pairs" / f"{cache_key(sample)}.npz")
        row.update(n_sky_pixels=int(regions.sky.sum()), n_sky_interior_pixels=int(regions.interior.sum()),
                   n_boundary_pairs=regions.n_pairs)
        # Data drift invalidates the whole run; it is not a model-specific failure.
        actual, _ = inspect_sample(sample, config["datasets"][sample.dataset])
        if actual != sample:
            raise ValueError(f"Dataset changed after prepare: {sample.sample_id}")
        try:
            path = prediction_path(model, sample)
            row["prediction_path"] = str(path)
            digest = sha256(path) if path.is_file() else None
            depth, dtype = read_prediction(path, model, (sample.height, sample.width))
            if sha256(path) != digest:
                raise ValueError("prediction_changed_while_reading")
            row.update(score(depth, regions, protocol))
            row.update(prediction_dtype=dtype, prediction_sha256=digest)
        except (OSError, ValueError, TypeError) as exc:
            row.update(status="failed", failure_reason=str(exc))
        rows.append(row)
    write_json(destination / "prediction_manifest.json", model)
    write_json(destination / "per_image.json", rows)
    write_csv(destination / "per_image.csv", rows)
    write_json(destination / "metrics.json", aggregate(rows))
    write_json(destination / "result_lock.json", {
        "protocol_lock_sha256": sha256(run / "protocol_lock.json"),
        "files": {name: sha256(destination / name) for name in
                  ("prediction_manifest.json", "per_image.json", "per_image.csv", "metrics.json")}})
    failed = sum(r["status"] != "ok" for r in rows)
    logging.info("%s: %d images, %d failed/partial (see per_image.csv)", model_id, len(rows), failed)
    return 2 if failed else 0


def formal_blockers(config: dict[str, Any], run: Path, models: dict[str, list[dict[str, Any]]]) -> list[str]:
    audit_result = read_json(run / "dataset_audit.json")
    blockers = list(audit_result["blockers"])
    if not {"hrwsi", "vkitti2"}.issubset(audit_result["datasets"]):
        blockers.append("HRWSI and VKITTI2 are both required for the formal two-dataset table")
    if not config.get("cross_dataset_duplicate_reviewed", False):
        blockers.append("cross-dataset near-duplicate review pending")
    vkitti = config["datasets"].get("vkitti2", {})
    for name, rows in models.items():
        evidence = config["models"][name].get("training_evidence", {})
        if any(r["dataset"] == "vkitti2" for r in rows):
            try:
                record = Path(evidence["training_record"])
                if sha256(record) != evidence["training_record_sha256"]:
                    raise ValueError("training-record hash mismatch")
                if evidence.get("no_vkitti_training", False):
                    if not evidence.get("reviewed", False):
                        raise ValueError("No-VKITTI-training evidence needs explicit review")
                else:
                    if not evidence.get("confirms_list_snapshot", False):
                        raise ValueError("training-list snapshot not confirmed")
                    actual = split_audit(Path(evidence["vkitti_train_list"]), Path(vkitti["test_list"]))
                    if actual["scene_overlap"]:
                        raise ValueError(f"seen-scene: {actual['scene_overlap']}")
                    if actual["train_list_sha256"] != evidence["train_list_sha256"]:
                        raise ValueError("training-list hash mismatch")
            except (KeyError, ValueError, OSError, TypeError) as exc:
                blockers.append(f"{name}: {exc}")
        for dataset in {r["dataset"] for r in rows}:
            subset = [r for r in rows if r["dataset"] == dataset]
            for metric in METRICS:
                fraction = sum(r.get(metric) is None for r in subset) / len(subset)
                if fraction > config["protocol"]["max_failure_fraction"]:
                    blockers.append(f"{name}/{dataset}/{metric}: failure fraction {fraction:.3f} exceeds preregistered limit")
    return blockers


def tex_escape(value: str) -> str:
    return "".join({"_": r"\_", "&": r"\&", "%": r"\%", "#": r"\#", "$": r"\$",
                    "{": r"\{", "}": r"\}", "\\": r"\textbackslash{}"}.get(c, c) for c in value)


def report(config: dict[str, Any], protocol: Protocol, run: Path, formal: bool) -> int:
    verify_lock(config, run)
    if not config["models"]:
        raise ValueError("No configured models/predictions; results remain pending")
    models: dict[str, list[dict[str, Any]]] = {}
    frozen_samples = read_manifest(run / "manifest.jsonl")
    frozen_ids = {s.sample_id for s in frozen_samples}
    for sample in frozen_samples:
        actual, _ = inspect_sample(sample, config["datasets"][sample.dataset])
        if actual != sample:
            raise ValueError(f"Dataset changed after prepare: {sample.sample_id}")
    for name in config["models"]:
        folder = run / "models" / name
        lock = read_json(folder / "result_lock.json")
        if lock["protocol_lock_sha256"] != sha256(run / "protocol_lock.json"):
            raise ValueError(f"Stale model result: {name}")
        for relative, expected in lock["files"].items():
            if sha256(confined(folder, relative)) != expected:
                raise ValueError(f"Modified model result: {name}/{relative}")
        rows = read_json(folder / "per_image.json")
        if len(rows) != len(frozen_ids) or {r["sample_id"] for r in rows} != frozen_ids:
            raise ValueError(f"Model does not cover the frozen manifest: {name}")
        for row in rows:
            if row["prediction_sha256"] and sha256(safe_absolute(row["prediction_path"])) != row["prediction_sha256"]:
                raise ValueError(f"Prediction changed since evaluation: {row['prediction_path']}")
        models[name] = rows
    representations = {m["representation"] for m in config["models"].values()}
    comparable = len(representations) == 1 and all(m.get("amplitude_comparability_verified", False)
                                                 for m in config["models"].values())
    comparisons = compare(models, comparable)
    blockers = formal_blockers(config, run, models)
    if not comparable:
        blockers.append("Amplitude representations not verified comparable; only SFOE can be compared")
    folder = run / ("formal_report" if formal else "diagnostic_report")
    folder.mkdir(exist_ok=False)
    write_json(folder / "report_status.json", {"formal_requested": formal, "blockers": blockers,
               "status": "BLOCKED" if formal and blockers else "formal" if formal else "diagnostic_only"})
    if formal and blockers:
        write_text(folder / "BLOCKED.md", "# 正式主表被阻断\n\n" + "\n".join(f"- {b}" for b in blockers) + "\n")
        logging.error("Formal report blocked: %s", "; ".join(blockers))
        return 2
    write_csv(folder / "comparison.csv", comparisons)
    write_json(folder / "comparison.json", comparisons)
    pseudo = any(cfg.get("mask_format") == "binary_manifest" for cfg in config["datasets"].values())
    title = ("SCOE formal comparison" if formal else "SCOE DIAGNOSTIC ONLY: Mask2Former pseudo labels"
             if pseudo else "SCOE DIAGNOSTIC ONLY (synthetic if fixture data)")
    lines = [f"# {title}", "", "逐图等权；各数据集、各指标使用所有模型的共同有效样本。N/A 表示未核实或无有效样本。", "",
             "| Model | Dataset | Metric | Mean | N | Missing |", "|---|---|---|---:|---:|---:|"]
    for row in comparisons:
        mean = f"{row['mean']:.8g}" if row["mean"] is not None else "N/A"
        lines.append(f"| {row['model_id']} | {row['dataset']} | {row['metric']} | {mean} | {row['N']} | {row['failures']} |")
    table_datasets = ("hrwsi", "vkitti2") if formal else sorted({r["dataset"] for r in comparisons})
    columns = [(dataset, metric) for dataset in table_datasets for metric in METRICS]
    lookup = {(r["model_id"], r["dataset"], r["metric"]): r for r in comparisons}
    headers = ["Method"] + [f"{ds} {metric}" for ds, metric in columns]
    tex = ["% " + title, r"\begin{tabular}{l" + "r" * len(columns) + "}",
           " & ".join(tex_escape(h) for h in headers) + r" \\", r"\hline"]
    wide_rows = []
    for name in models:
        values = []
        wide: dict[str, Any] = {"model_id": name}
        for ds, metric in columns:
            row = lookup.get((name, ds, metric), {})
            mean = row.get("mean")
            values.append(f"{mean:.8g}" if mean is not None else "N/A")
            wide[f"{ds}_{metric}"] = mean
        tex.append(" & ".join([tex_escape(name), *values]) + r" \\")
        wide_rows.append(wide)
    write_csv(folder / "comparison_wide.csv", wide_rows)
    tex.append(r"\end{tabular}")
    lines.extend(["", "限制 / 审计状态：", ""] + [f"- {b}" for b in blockers])
    write_text(folder / "comparison.md", "\n".join(lines) + "\n")
    write_text(folder / ("paper_table.tex" if formal else "diagnostic_table.tex"), "\n".join(tex) + "\n")
    # High/low valid SFOE cases, plus shared predictions; all use one display range.
    diagnostics = folder / "diagnostics"
    diagnostics.mkdir()
    first = next(iter(models.values()))
    ordered = sorted((r for r in first if r.get("sfoe") is not None), key=lambda r: (r["sfoe"], r["sample_id"]))
    chosen = {r["sample_id"] for r in ordered[:3] + ordered[-3:]}
    samples = {s.sample_id: s for s in read_manifest(run / "manifest.jsonl")}
    for sid in sorted(chosen):
        sample = samples[sid]
        predictions = {}
        for name, model in config["models"].items():
            row = next(r for r in models[name] if r["sample_id"] == sid)
            if row["prediction_sha256"]:
                predictions[name] = read_prediction(Path(row["prediction_path"]), model, (sample.height, sample.width))[0]
        if predictions:
            prediction_diagnostic(sample_rgb(sample, config["datasets"][sample.dataset]), predictions,
                                  diagnostics / f"{cache_key(sample)}.png")
    logging.info("Report written: %s", folder)
    return 0
