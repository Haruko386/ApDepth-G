"""Run dataset inference and/or evaluation with one matching depth checkpoint."""

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


def build_commands(args, dataset):
    prediction = args.output_dir / dataset / "prediction"
    evaluation = args.output_dir / dataset / "eval"
    commands = []
    if args.phase in ("both", "infer"):
        command = [
            sys.executable, "infer.py", "--backbone", args.backbone,
            "--checkpoint", args.checkpoint, "--unet_checkpoint", str(args.unet_checkpoint),
            "--dataset_config", DATASETS[dataset], "--base_data_dir", args.base_data_dir,
            "--output_dir", str(prediction), "--denoise_steps", str(args.denoise_steps),
            "--ensemble_size", str(args.ensemble_size), "--processing_res", str(args.processing_res),
            "--seed", str(args.seed),
        ]
        if args.da2_config:
            command.extend(["--da2_config", args.da2_config])
        if args.fp16:
            command.append("--fp16")
        commands.append(command)
    if args.phase in ("both", "eval"):
        commands.append([
            sys.executable, "eval.py", "--prediction_dir", str(prediction),
            "--output_dir", str(evaluation), "--dataset_config", DATASETS[dataset],
            "--base_data_dir", args.base_data_dir, "--alignment", "least_square",
        ])
    return commands


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", choices=["sd2", "sdxl"], default="sdxl")
    parser.add_argument("--checkpoint", help="Base pipeline used in training")
    parser.add_argument("--unet_checkpoint", type=Path, help="Directory containing unet/, scheduler/, metadata")
    parser.add_argument("--da2_config", help="Optional explicit DA2 config/path override")
    parser.add_argument("--fp16", "--half_precision", action="store_true",
                        help="Explicit FP16 inference; defaults to FP32")
    parser.add_argument("--base_data_dir", default="/root/Dataset")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", choices=list(DATASETS), default=list(DATASETS))
    parser.add_argument("--phase", choices=["both", "infer", "eval"], default="both")
    parser.add_argument("--denoise_steps", type=int, default=50)
    parser.add_argument("--ensemble_size", type=int, default=1)
    parser.add_argument("--processing_res", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    if Path.cwd().resolve() != REPO_ROOT:
        parser.error(f"Run from the repository root: {REPO_ROOT}")
    if len(set(args.datasets)) != len(args.datasets):
        parser.error("Dataset names must be distinct")
    if args.denoise_steps < 1 or args.ensemble_size < 1 or args.processing_res < 0:
        parser.error("Steps/ensemble must be positive; processing_res must be nonnegative")
    if args.phase != "eval" and (not args.checkpoint or not args.unet_checkpoint):
        parser.error("Inference requires --checkpoint and --unet_checkpoint")
    if not args.dry_run:
        if args.phase != "eval":
            for component in ("unet", "scheduler"):
                if not (args.unet_checkpoint / component).is_dir():
                    raise FileNotFoundError(args.unet_checkpoint / component)
        # Check the entire plan before starting a long dataset inference run.
        for dataset in args.datasets:
            prediction = args.output_dir / dataset / "prediction"
            evaluation = args.output_dir / dataset / "eval"
            if args.phase != "eval" and prediction.exists():
                raise FileExistsError(f"Use a new output directory: {prediction}")
            if args.phase != "infer" and evaluation.exists():
                raise FileExistsError(f"Use a new evaluation output directory: {evaluation}")
            if args.phase == "eval" and not prediction.is_dir():
                raise FileNotFoundError(prediction)
    for dataset in args.datasets:
        for command in build_commands(args, dataset):
            print(shlex.join(command), flush=True)
            if not args.dry_run:
                subprocess.run(command, check=True)
    if not args.dry_run and args.phase != "infer":
        rows = []
        for dataset in args.datasets:
            result = json.loads((args.output_dir / dataset / "eval/metrics.json").read_text(encoding="utf-8"))
            rows.append({"dataset": dataset, "samples": result["samples"],
                         "AbsRel_x100": 100 * result["metrics"]["abs_relative_difference"],
                         "delta1_percent": 100 * result["metrics"]["delta1_acc"]})
        with (args.output_dir / "paper_results.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
