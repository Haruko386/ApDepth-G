"""Run SSD benchmark or full-frame sky inference and main's evaluation."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
DATASETS = {
    "nyu": "data_nyu_test.yaml", "kitti": "data_kitti_eigen_test.yaml",
    "eth3d": "data_eth3d.yaml", "scannet": "data_scannet_val.yaml",
    "diode": "data_diode_all.yaml", "vkitti2": "data_vkitti_val_sampled.yaml",
}
SKY_DATASETS = ("vkitti2", "kitti", "diode")


def commands(args):
    names = args.datasets or (SKY_DATASETS if args.sky_mask_name else ("nyu", "kitti", "eth3d", "scannet", "diode"))
    if args.sky_mask_name and any(name not in SKY_DATASETS for name in names):
        raise ValueError("Sky evaluation supports vkitti2, kitti and diode")
    if not args.eval_only and not args.unet_checkpoint:
        raise ValueError("Inference requires --unet_checkpoint from your SSD training run")
    for name in names:
        config = str(ROOT / "config/dataset" / DATASETS[name])
        output = Path(args.output_dir).resolve() / ("sky" if args.sky_mask_name else "depth") / name
        prediction = output / "prediction"
        data_args = ["--dataset_config", config, "--base_data_dir", str(Path(args.base_data_dir).resolve())]
        sky_args = ["--sky_mask_name", args.sky_mask_name] if args.sky_mask_name else []
        if not args.eval_only:
            infer = [sys.executable, str(ROOT / "infer.py"), "--backbone", "ssd",
                     "--checkpoint", str(Path(args.checkpoint).resolve()),
                     "--unet_checkpoint", str(Path(args.unet_checkpoint).resolve()),
                     *data_args, *sky_args, "--output_dir", str(prediction),
                     "--denoise_steps", str(args.denoise_steps), "--ensemble_size", str(args.ensemble_size),
                     "--processing_res", str(args.processing_res), "--seed", str(args.seed)]
            if args.da2_config:
                infer += ["--da2_config", str(Path(args.da2_config).resolve())]
            if args.fp16:
                infer.append("--fp16")
            yield infer
        evaluation = [sys.executable, str(ROOT / "eval.py"), *data_args, *sky_args,
                      "--prediction_dir", str(prediction), "--output_dir", str(output / "metrics")]
        if not args.sky_mask_name:
            evaluation += ["--alignment", "least_square"]
            if name == "eth3d":
                evaluation += ["--alignment_max_res", "1024"]
        yield evaluation


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_data_dir", default=os.environ.get("BASE_DATA_DIR"))
    parser.add_argument("--checkpoint", default=str(Path(os.environ.get("BASE_CKPT_DIR", "checkpoint")) / "SSD-1B"))
    parser.add_argument("--unet_checkpoint")
    parser.add_argument("--output_dir", default="output/ssd_eval")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS)
    parser.add_argument("--sky_mask_name")
    parser.add_argument("--da2_config")
    parser.add_argument("--denoise_steps", type=int, default=50)
    parser.add_argument("--ensemble_size", type=int, default=1)
    parser.add_argument("--processing_res", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if not args.base_data_dir:
        parser.error("Set BASE_DATA_DIR or pass --base_data_dir")
    if args.denoise_steps < 1 or args.ensemble_size < 1 or args.processing_res < 0:
        parser.error("Steps and ensemble size must be positive; resolution must be nonnegative")
    try:
        plan = list(commands(args))
    except ValueError as exc:
        parser.error(str(exc))
    for command in plan:
        print(shlex.join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
