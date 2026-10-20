"""Generate full-frame Mask2Former sky masks from BASE_DATA_DIR evaluation RGB."""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/sky_segmentation.yaml")
    parser.add_argument("--base_data_dir", "--base-data-dir", default=os.environ.get("BASE_DATA_DIR"))
    parser.add_argument("--datasets", "--dataset", nargs="+", choices=("vkitti", "vkitti2", "kitti", "diode"),
                        help="Dataset(s) in the same order as --data_split; inferred from standard data_split paths if omitted")
    parser.add_argument("--data_split", "--data-split", nargs="+", required=True,
                        help="Existing RGB split list(s); one per dataset, in the same order")
    parser.add_argument("--checkpoint", help="Local Transformers Mask2Former folder; never a download request")
    parser.add_argument("--output_dir", "--output-dir", help="New metadata/config directory; masks go inside each dataset's sky_masks/<run name>/")
    parser.add_argument("--device", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--limit-samples", type=int, help="At most N images per dataset; labelled limited")
    parser.add_argument("--dry-run", action="store_true", help="Check listed RGB paths without loading a model or writing outputs")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        import tarfile
        import yaml
        from src.sky_eval.rgb_io import RGBSource
        from script.dataset_preprocess.sky_mask.segmentation import (
            Mask2FormerSky, generate, make_plans, mask_location, repository_path, resolve_splits,
        )

        if not args.base_data_dir:
            raise ValueError("Set BASE_DATA_DIR or pass --base_data_dir")
        if args.limit_samples is not None and args.limit_samples < 1:
            raise ValueError("--limit-samples must be positive")
        config = yaml.safe_load(repository_path(args.config).read_text(encoding="utf-8-sig"))
        if not isinstance(config, dict) or not isinstance(config.get("datasets"), dict):
            raise ValueError("Configuration requires a datasets mapping")
        split_overrides = resolve_splits(args.data_split, args.datasets)
        names = list(split_overrides)
        base = repository_path(args.base_data_dir)
        config["base_data_dir"] = str(base)
        plans = make_plans(config, base, names, args.limit_samples, split_overrides)
        output = repository_path(args.output_dir or config["output_dir"])
        if args.dry_run:
            failed = 0
            for plan in plans:
                with RGBSource(plan.source) as source:
                    missing = [path for path in plan.paths if not source.contains(path)]
                failed += len(missing)
                logging.info("%s: source=%s, split=%s, selected=%d/%d, missing=%d",
                             plan.dataset, plan.source, plan.split_name, len(plan.paths), plan.total, len(missing))
                logging.info("%s: data_split=%s; masks=%s", plan.dataset, plan.split_list, mask_location(plan, output)[1])
                for path in missing[:10]:
                    logging.error("Missing %s RGB: %s", plan.dataset, path)
            logging.info("Dry run complete; no model loaded and no files written. Planned output: %s", output)
            return 2 if failed else 0
        if output.exists():
            raise FileExistsError(f"Output exists; choose a new --output_dir: {output}")
        for plan in plans:
            if mask_location(plan, output)[1].exists():
                raise FileExistsError(f"Dataset mask directory exists; use a new --output_dir run name: {mask_location(plan, output)[1]}")
        model_cfg = config["model"]
        checkpoint = args.checkpoint or model_cfg.get("checkpoint")
        if not checkpoint:
            raise ValueError("Supply --checkpoint pointing to an existing local Mask2Former folder")
        previews = config.get("preview_count", 50)
        if not isinstance(previews, int) or previews < 0:
            raise ValueError("preview_count must be a nonnegative integer")
        segmenter = Mask2FormerSky(repository_path(checkpoint), model_cfg.get("sky_labels", ["sky"]),
                                  args.device or model_cfg.get("device", "auto"), model_cfg.get("seed", 2024))
        return generate(plans, segmenter, output, config, previews, args.limit_samples is not None)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, ImportError, tarfile.TarError) as exc:
        logging.error("%s", exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
