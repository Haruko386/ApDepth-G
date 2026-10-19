"""Evaluate all ablations on NYUv2/KITTI, with optional full-model extra datasets."""

import argparse
import csv
import json
from pathlib import Path
import shlex
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASETS = {
    "nyu": "config/dataset/data_nyu_test.yaml",
    "kitti": "config/dataset/data_kitti_eigen_test.yaml",
    "eth3d": "config/dataset/data_eth3d.yaml",
    "scannet": "config/dataset/data_scannet_val.yaml",
    "diode": "config/dataset/data_diode_all.yaml",
}


def evaluation_plan(ids, main_extra=False):
    for experiment in ids:
        for dataset in ("nyu", "kitti"):
            yield experiment, dataset
        if main_extra and experiment == "00_full":
            for dataset in ("eth3d", "scannet", "diode"):
                yield experiment, dataset


def collect_results(plan, output_dir):
    rows = []
    for experiment, dataset in plan:
        path = output_dir / experiment / dataset / "eval" / "metrics.json"
        result = json.loads(path.read_text(encoding="utf-8"))
        if result["alignment"] != "least_square":
            raise ValueError(f"Expected least_square alignment: {path}")
        metrics = result["metrics"]
        rows.append({"experiment": experiment, "dataset": dataset,
                     "samples": result["samples"],
                     "AbsRel_x100": 100 * metrics["abs_relative_difference"],
                     "delta1_percent": 100 * metrics["delta1_acc"]})
    destination = output_dir / "paper_results.csv"
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_root", type=Path, default=Path("output/ablation"))
    parser.add_argument("--output_dir", type=Path, default=Path("output/ablation_eval"))
    parser.add_argument("--base_checkpoint", help="The same SD2 base pipeline used in training")
    parser.add_argument("--base_data_dir")
    parser.add_argument("--ids", nargs="+", help="Config stems; default: every config/ablation YAML")
    parser.add_argument("--main_extra", action="store_true",
                        help="Also evaluate 00_full on ETH3D, ScanNet, and DIODE")
    parser.add_argument("--collect_only", action="store_true", help="Read existing metrics.json files")
    parser.add_argument("--dry_run", action="store_true", help="Print commands without loading models/data")
    args = parser.parse_args()
    if Path.cwd().resolve() != REPO_ROOT:
        parser.error(f"Run this script from the repository root: {REPO_ROOT}")
    known = {p.stem for p in (REPO_ROOT / "config/ablation").glob("*.yaml")}
    ids = sorted(known) if args.ids is None else args.ids
    if not ids or len(set(ids)) != len(ids) or set(ids) - known:
        parser.error("Use distinct config stems from config/ablation")
    if args.main_extra and "00_full" not in ids:
        parser.error("--main_extra requires 00_full in --ids")
    plan = list(evaluation_plan(ids, args.main_extra))
    if not args.collect_only:
        if not args.base_checkpoint or not args.base_data_dir:
            parser.error("--base_checkpoint and --base_data_dir are required for evaluation")
        for experiment, dataset in plan:
            run_dir = args.run_root / experiment
            config_path = run_dir / "config.yaml"
            max_iter = 23000
            if config_path.is_file():
                from omegaconf import OmegaConf
                max_iter = int(OmegaConf.load(config_path).max_iter)
            checkpoint = run_dir / "checkpoint" / f"iter_{max_iter:06d}"
            prediction_dir = args.output_dir / experiment / dataset / "prediction"
            eval_dir = prediction_dir.parent / "eval"
            if not args.dry_run:
                if not config_path.is_file() or not checkpoint.is_dir():
                    raise FileNotFoundError(f"Training run/final checkpoint is missing: {run_dir}")
                if prediction_dir.exists() or eval_dir.exists():
                    raise FileExistsError(f"Use a new output directory or --collect_only: {prediction_dir.parent}")
            commands = [
                [sys.executable, "infer.py", "--checkpoint", args.base_checkpoint,
                 "--unet_checkpoint", str(checkpoint), "--dataset_config", DATASETS[dataset],
                 "--base_data_dir", args.base_data_dir, "--output_dir", str(prediction_dir),
                 "--denoise_steps", "50", "--ensemble_size", "1", "--processing_res", "0", "--seed", "2024"],
                [sys.executable, "eval.py", "--prediction_dir", str(prediction_dir),
                 "--output_dir", str(eval_dir), "--dataset_config", DATASETS[dataset],
                 "--base_data_dir", args.base_data_dir, "--alignment", "least_square"],
            ]
            for command in commands:
                print(shlex.join(command), flush=True)
                if not args.dry_run:
                    subprocess.run(command, check=True)
    if not args.dry_run:
        print(f"Paper-scale metrics: {collect_results(plan, args.output_dir)}")


if __name__ == "__main__":
    main()
