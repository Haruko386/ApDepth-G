"""Combine coverage records from any servers and plot locally; no datasets needed."""

import argparse
import glob
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.util.coverage_records import collect_records, image_inventory, save_charts, save_combined_chart, write_csv


def dataset_file_order(path):
    prefixes = ("hypersim", "vkitti", "nyu", "kitti", "eth3d", "scannet", "diode")
    name = path.name.lower()
    rank = next((index for index, prefix in enumerate(prefixes) if name.startswith(prefix)), len(prefixes))
    return rank, name


def expand_record_paths(inputs):
    paths = []
    for value in inputs:
        candidate = Path(value).expanduser()
        if candidate.is_dir():
            matches = sorted(candidate.glob("*.jsonl"), key=dataset_file_order)
        elif glob.has_magic(str(candidate)):
            matches = [Path(path) for path in sorted(glob.glob(str(candidate)))]
        else:
            matches = [candidate] if candidate.is_file() else []
        if not matches:
            raise FileNotFoundError(f"No dataset records found: {value}")
        paths.extend(matches)
    if len({path.resolve() for path in paths}) != len(paths):
        raise ValueError("The same record file was specified more than once")
    return paths


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", "--record_files", nargs="+", action="extend", required=True,
                        help="Dataset JSONL files, directories, or glob patterns; argument order is chart order")
    parser.add_argument("--output_dir", type=Path, default=Path("output/coverage_figure"))
    parser.add_argument("--average", choices=("pixel", "image"), default="pixel",
                        help="Pixel-weighted or equal-image-weighted chart; both are exported in CSV")
    parser.add_argument("--combined_only", action="store_true",
                        help="Generate only the new single-column combined chart; preserve existing separate figures")
    args = parser.parse_args(argv)
    paths = expand_record_paths(args.records)
    metadata, summaries, latent = collect_records(paths)
    inventory = image_inventory(summaries)
    # This process only reads portable JSON counts. Dataset/config paths stored
    # in metadata are provenance; no dataset, split, model, or torch is loaded.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.combined_only:
        save_charts(summaries, latent, args.output_dir, args.average)
    save_combined_chart(summaries, latent, args.output_dir, args.average, metadata[0]["latent_scale"])
    write_csv(args.output_dir / "summary.csv", summaries)
    write_csv(args.output_dir / "latent_summary.csv", latent)
    write_csv(args.output_dir / "image_counts.csv", inventory["datasets"])
    report = {"record_files": [str(path.resolve()) for path in paths], "chart_average": args.average,
              "protocols": metadata, "summary": summaries, "latent_summary": latent,
              "image_counts": inventory}
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print("Chart order: " + " -> ".join(meta["dataset"] for meta in metadata))
    print(f"Images: {inventory['total_images']:,} total; {inventory['training_images']:,} training; {inventory['test_images']:,} test")
    print(f"Saved figures and summary tables to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
