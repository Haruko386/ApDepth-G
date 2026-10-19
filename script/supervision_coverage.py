"""Record per-image depth-mask coverage, one portable JSONL file per dataset.

Run from any directory. Paths in existing repository configs remain relative to
the repository root. See doc/supervision_coverage.md for server examples.
"""

import argparse
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, ListConfig, OmegaConf
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode, Resize
from tqdm.auto import tqdm

from src.dataset import DatasetMode, get_dataset
from src.dataset.base_depth_dataset import BaseDepthDataset
from src.dataset.kitti_dataset import KITTIDataset
from src.util.coverage_records import FORMAT_VERSION, COUNT_KEYS, aggregate, record_filename, write_line


def repo_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def load_config(path, parents=()):
    """Same base_config merging as training, with an explicit cycle check."""
    path = repo_path(path).resolve()
    if path in parents:
        raise ValueError(f"Circular base_config: {path}")
    config = OmegaConf.load(path)
    if isinstance(config, ListConfig):
        return config
    merged = OmegaConf.create({})
    for base in config.get("base_config", []):
        merged = OmegaConf.merge(merged, load_config(base, (*parents, path)))
    return OmegaConf.merge(merged, config)


def dataset_configs(config, split="train"):
    """Accept standalone, mixed, dataset.train/val/vis, and dataset-list YAML."""
    if isinstance(config, ListConfig):
        return [item for child in config for item in dataset_configs(child, split)]
    if not isinstance(config, DictConfig):
        raise ValueError("Expected a dataset YAML mapping or list")
    if config.get("name") == "mixed":
        return dataset_configs(config.dataset_list, split)
    if "name" in config and "filenames" in config:
        return [OmegaConf.create(OmegaConf.to_container(config, resolve=True))]
    if "dataset" in config:
        return dataset_configs(config.dataset, split)
    if split in config:
        return dataset_configs(config[split], split)
    raise ValueError(f"No dataset configuration found for split '{split}'")


def directory_overrides(items):
    result = {}
    for item in items:
        key, separator, value = item.partition("=")
        if not separator or not key or not value or key in result:
            raise ValueError("Use distinct --dataset_dir NAME=/absolute/path entries")
        result[key] = str(Path(value).expanduser())
    return result


def read_sky_list(options):
    """Optional two-column list: original RGB relative path, binary-mask path."""
    if not options.get("sky_mask_list"):
        return None
    list_path = repo_path(options["sky_mask_list"])
    root = Path(options.get("sky_mask_dir", list_path.parent)).expanduser()
    root = repo_path(root)
    mapping = {}
    for number, line in enumerate(list_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        fields = line.split()
        if len(fields) != 2 or fields[0] in mapping:
            raise ValueError(f"Invalid/duplicate sky-mask entry at {list_path}:{number}")
        mask_path = Path(fields[1])
        mapping[fields[0]] = str(mask_path if mask_path.is_absolute() else root / mask_path)
    return mapping


class CoverageDataset(Dataset):
    """Reuse repository depth decoding and masks; skip RGB and normalization."""

    def __init__(self, config, base_data_dir, indices, depth_type="raw", native=False):
        self.name = str(config.name)
        self.depth_type = depth_type
        self.resize_hw = None if native else config.get("resize_to_hw")
        self.sky_files = read_sky_list(config.get("coverage", {}))
        self.source = get_dataset(config, str(base_data_dir), DatasetMode.EVAL)
        # BaseDepthDataset stores a default lambda for RGB. RGB is unused here;
        # remove it so this wrapper is pickleable with spawn workers on any OS.
        self.source.rgb_transform = None
        self.indices = indices
        if self.sky_files is not None:
            missing = [self.source.filenames[i][0] for i in indices
                       if self.source.filenames[i][0] not in self.sky_files]
            if missing:
                raise ValueError(f"Missing {len(missing)} sky-mask list entries; first: {missing[0]}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        index = self.indices[position]
        entry = self.source.filenames[index]
        rgb_path = entry[0]
        column = 2 if self.depth_type == "filled" and self.source.has_filled_depth else 1
        depth = torch.from_numpy(np.array(self.source._read_depth_file(entry[column]),
                                         dtype=np.float32, copy=True)).squeeze()
        if depth.ndim != 2:
            raise ValueError(f"Expected 2D depth: {entry[column]} {tuple(depth.shape)}")
        original_shape = tuple(depth.shape)
        sky = None
        if self.sky_files is not None:
            path = Path(self.sky_files[rgb_path])
            if path.suffix.lower() == ".npy":
                array = np.asarray(np.load(path)).squeeze()
            else:
                with Image.open(path) as image:
                    array = np.array(image)
            if array.shape != original_shape or not np.isin(array, [0, 1, 255]).all():
                raise ValueError(f"Sky mask must be registered binary 0/1 or 0/255: {path}")
            sky = torch.from_numpy(np.array(array > 0, copy=True))

        if self.name == "diode":
            # DIODEDataset uses the supplied mask directly, not its nominal range.
            source_mask = torch.from_numpy(self.source._read_npy_file(entry[2]).astype(bool)).squeeze()
            if source_mask.shape != depth.shape:
                raise ValueError(f"DIODE depth/mask shape mismatch: {rgb_path}")
        else:
            source_mask = None

        if getattr(self.source, "kitti_bm_crop", False):
            if depth.shape[0] < 352 or depth.shape[1] < 1216:
                raise ValueError(f"KITTI benchmark crop requires >=352x1216 pixels: {rgb_path}")
            depth = KITTIDataset.kitti_benchmark_crop(depth)
            if sky is not None:
                sky = KITTIDataset.kitti_benchmark_crop(sky)

        if self.name == "diode":
            data_valid = source_mask
            protocol_valid = source_mask.clone()
        else:
            # Uncropped validity and configured evaluation mask are separate.
            data_valid = BaseDepthDataset._get_valid_mask(self.source, depth)
            protocol_valid = self.source._get_valid_mask(depth)

        bad = ~torch.isfinite(depth) | (depth <= 0)
        if (bad & protocol_valid).any():
            raise ValueError(f"Configured mask marks nonfinite/nonpositive depth as valid: {rgb_path}")

        if self.resize_hw is not None:
            resize = Resize(list(self.resize_hw), interpolation=InterpolationMode.NEAREST_EXACT)
            data_valid = resize(data_valid[None])[0]
            protocol_valid = resize(protocol_valid[None])[0]
            if sky is not None:
                sky = resize(sky[None])[0]

        result = {"index": index, "rgb": rgb_path, "data_valid": data_valid.bool(),
                  "valid": protocol_valid.bool(), "original_height": original_shape[0],
                  "original_width": original_shape[1]}
        if sky is not None:
            result["sky"] = sky.bool()
        return result


def unbatch(sample):
    # A batch of one allows differently sized images and bounded host/GPU memory.
    return sample[0]


def count_regions(sample, device, scale):
    valid = sample["valid"].to(device)
    data_valid = sample["data_valid"].to(device)
    regions = {"all": torch.ones_like(valid)}
    if "sky" in sample:
        sky = sample["sky"].to(device)
        regions.update(sky=sky, non_sky=~sky)
    rows = []
    for region, mask in regions.items():
        counts = torch.stack((mask.sum(), (mask & valid).sum(),
                              (mask & ~data_valid).sum(), (mask & data_valid & ~valid).sum()))
        row = dict(zip(COUNT_KEYS, (int(x) for x in counts.cpu().tolist())))
        row["region"] = region
        rows.append(row)
    height, width = valid.shape
    if min(height, width) < scale:
        latent = {"latent_cells": 0, "latent_valid_cells": 0, "latent_invalid_cells": 0,
                  "latent_mixed_cells": 0, "latent_border_pixels": height * width}
    else:
        # Identical to src.util.diffusion_training.latent_validity_masks, kept
        # here to avoid importing diffusers for a dataset-only CPU utility.
        block_mask = valid[None, None]
        full_valid = ~F.max_pool2d((~block_mask).float(), scale, scale).bool()
        full_invalid = ~F.max_pool2d(block_mask.float(), scale, scale).bool()
        counts = torch.stack((full_valid.sum(), full_invalid.sum())).cpu().tolist()
        cells = full_valid.numel()
        latent = {"latent_cells": cells, "latent_valid_cells": int(counts[0]),
                  "latent_invalid_cells": int(counts[1]),
                  "latent_mixed_cells": cells - int(counts[0]) - int(counts[1]),
                  "latent_border_pixels": height * width - cells * scale * scale}
    return rows, latent


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_config", "--dataset_configs", nargs="+", action="extend", required=True)
    parser.add_argument("--base_data_dir", type=Path, default=Path("/root/Dataset"))
    parser.add_argument("--dataset_dir", action="append", default=[], metavar="NAME=PATH",
                        help="Override dir by disp_name or dataset type; repeat for multiple datasets")
    parser.add_argument("--split", choices=("train", "val", "vis"), default="train")
    parser.add_argument("--output_dir", type=Path, default=Path("output/supervision_coverage"))
    parser.add_argument("--device", default="cpu", help="cpu, cuda, cuda:0, or auto")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--depth_type", choices=("raw", "filled"), default="raw")
    parser.add_argument("--native_resolution", action="store_true", help="Ignore configured resize_to_hw")
    parser.add_argument("--latent_scale", type=int, default=8)
    parser.add_argument("--max_samples", type=int, default=None, help="Random reproducible subset per dataset")
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--dry_run", action="store_true", help="Validate config/roots/lists and print plan; no depth I/O")
    args = parser.parse_args(argv)
    if args.num_workers < 0 or args.latent_scale < 1 or (args.max_samples is not None and args.max_samples < 1):
        parser.error("num_workers >= 0; latent_scale and max_samples must be positive")
    requested_device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested_device if args.device == "auto" else args.device)
    if device.type not in ("cpu", "cuda") or (device.type == "cuda" and not torch.cuda.is_available()):
        parser.error(f"Requested device unavailable: {device}; use --device cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    overrides = directory_overrides(args.dataset_dir)
    jobs = []
    names = set()
    used_overrides = set()
    for path in args.dataset_config:
        for config in dataset_configs(load_config(path), args.split):
            name = str(config.disp_name)
            if name in names:
                raise ValueError(f"Duplicate disp_name '{name}'; use distinct split names")
            names.add(name)
            key = name if name in overrides else str(config.name)
            if key in overrides:
                config.dir = overrides[key]
                used_overrides.add(key)
            config.filenames = str(repo_path(config.filenames))
            if not Path(config.filenames).is_file():
                raise FileNotFoundError(f"Filename list not found: {config.filenames}")
            # Instantiate once to count exactly the samples retained by the loader.
            source = get_dataset(config, str(args.base_data_dir.expanduser()), DatasetMode.EVAL)
            indices = list(range(len(source)))
            if args.max_samples is not None and len(indices) > args.max_samples:
                indices = sorted(np.random.default_rng(args.seed).choice(indices, args.max_samples, replace=False).tolist())
            if not indices:
                raise ValueError(f"No depth samples in {name}")
            job = CoverageDataset(config, args.base_data_dir.expanduser(), indices, args.depth_type, args.native_resolution)
            jobs.append((config, job, len(source)))
            print(f"{name}: {len(indices):,}/{len(source):,} images; root={source.dataset_dir}; "
                  f"resize={job.resize_hw}; eval_crop={getattr(source, 'valid_mask_crop', getattr(source, 'eigen_valid_mask', None))}")
    if set(overrides) - used_overrides:
        raise ValueError(f"Unused --dataset_dir keys: {sorted(set(overrides) - used_overrides)}")
    if args.dry_run:
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    destinations = [args.output_dir / record_filename(str(config.disp_name)) for config, _, _ in jobs]
    for destination in destinations:
        if destination.exists() or destination.with_suffix(".jsonl.partial").exists():
            raise FileExistsError(f"Record already exists; use a different output directory: {destination}")
    print(f"Device: {device}. Recording only; no plotting. Output: {args.output_dir.resolve()}")
    for order, ((config, job, available), destination) in enumerate(zip(jobs, destinations)):
        minimum, maximum = float(job.source.min_depth), float(job.source.max_depth)
        metadata = {
            "kind": "metadata", "format_version": FORMAT_VERSION,
            "dataset": str(config.disp_name), "dataset_type": str(config.name), "dataset_order": order,
            "available_images": available, "selected_images": len(job), "latent_scale": args.latent_scale,
            "config": OmegaConf.to_container(config, resolve=True),
            "resolved_dataset_dir": str(job.source.dataset_dir),
            "depth_type": args.depth_type, "resize_to_hw": list(job.resize_hw) if job.resize_hw else None,
            "min_depth_m": minimum, "max_depth_m": maximum if math.isfinite(maximum) else None,
            "validity_rule": "provided_mask_only" if config.name == "diode" else "strict_depth_range_and_configured_eval_mask",
            "selection": {"seed": args.seed, "max_samples": args.max_samples, "split": args.split},
            "notes": ["Invalid depth is not a semantic sky label.",
                      "Evaluation masks measure evaluation support, not training exposure.",
                      "Benchmark spatial crops are outside the pixel denominator; evaluation-mask exclusions are separate.",
                      "Latent blocks precede VGC sample gates and DA2 far-field selection."]}
        partial = destination.with_suffix(".jsonl.partial")
        total_pixels = total_valid = processed = 0
        with partial.open("x", encoding="utf-8") as handle:
            write_line(handle, metadata)
            loader = DataLoader(job, batch_size=1, shuffle=False, num_workers=args.num_workers,
                                collate_fn=unbatch, pin_memory=device.type == "cuda",
                                multiprocessing_context="spawn" if args.num_workers else None)
            with torch.inference_mode():
                for sample in tqdm(loader, total=len(job), desc=str(config.disp_name)):
                    rows, latent = count_regions(sample, device, args.latent_scale)
                    for row in rows:
                        row["valid_percent"] = 100 * row["valid_pixels"] / row["pixels"] if row["pixels"] else None
                    write_line(handle, {
                        "kind": "image", "index": sample["index"], "rgb": sample["rgb"],
                        "original_height": sample["original_height"], "original_width": sample["original_width"],
                        "height": sample["valid"].shape[0], "width": sample["valid"].shape[1],
                        "regions": rows, "latent": latent})
                    total_pixels += rows[0]["pixels"]
                    total_valid += rows[0]["valid_pixels"]
                    processed += 1
            if processed != len(job):
                raise RuntimeError(f"Incomplete dataset: {config.disp_name}: {processed}/{len(job)}")
            write_line(handle, {"kind": "complete", "images": processed})
        partial.replace(destination)
        print(f"Saved {destination.name}: {processed:,} images, {100 * total_valid / total_pixels:.4f}% valid pixels")
    print("Complete. Copy these dataset JSONL files to your local machine and run script/plot_supervision_coverage.py.")


if __name__ == "__main__":
    main()
