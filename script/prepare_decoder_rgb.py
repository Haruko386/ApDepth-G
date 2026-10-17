"""Build a small RGB-only calibration set from the project's dataset lists."""
import argparse
import hashlib
from pathlib import Path, PurePosixPath
import random
import re
import shutil


def read_paths(path):
    return list(dict.fromkeys(line.split()[0] for line in Path(path).read_text(encoding="utf-8").splitlines()
                              if line.strip()))


def vkitti_scene(path):
    return next(part for part in PurePosixPath(path).parts if re.fullmatch(r"Scene\d+", part))


def prepare(base_data_dir, output_dir, dataset_config="config/dataset/dataset_train.yaml",
            vkitti_val_list="data_split/vkitti/vkitti_val.txt", seed=2024,
            outdoor_train=160, outdoor_val=40, indoor_train=40, indoor_val=10):
    from omegaconf import OmegaConf

    counts = (outdoor_train, outdoor_val, indoor_train, indoor_val)
    if any(n < 1 for n in counts):
        raise ValueError("All sample counts must be positive")
    destination = Path(output_dir).resolve()
    if destination.exists():
        raise FileExistsError(f"Use a new output directory: {destination}")
    specs = {d.name: d for d in OmegaConf.load(dataset_config).dataset.train.dataset_list}
    rgb_roots = {name: Path(base_data_dir) / specs[name].dir for name in ("vkitti", "hypersim")}
    for path in rgb_roots.values():
        if not path.is_dir():
            raise ValueError(f"Expected extracted RGB directory from dataset config: {path}")
    outdoor = read_paths(specs["vkitti"].filenames)
    outdoor_holdout = read_paths(vkitti_val_list)
    if {vkitti_scene(p) for p in outdoor} & {vkitti_scene(p) for p in outdoor_holdout}:
        raise ValueError("VKITTI train and validation scenes overlap")
    indoor = read_paths(specs["hypersim"].filenames)
    scenes = sorted({PurePosixPath(p).parts[0] for p in indoor})
    if len(scenes) < 2:
        raise ValueError("Need at least two Hypersim scenes for calibration train/val")
    rng = random.Random(seed)
    rng.shuffle(scenes)
    heldout = set(scenes[:max(1, len(scenes) // 5)])
    plans = [
        ("train", "vkitti", outdoor, outdoor_train),
        ("val", "vkitti", outdoor_holdout, outdoor_val),
        ("train", "hypersim", [p for p in indoor if PurePosixPath(p).parts[0] not in heldout], indoor_train),
        ("val", "hypersim", [p for p in indoor if PurePosixPath(p).parts[0] in heldout], indoor_val),
    ]
    selected, seen = [], set()
    for split, name, candidates, count in plans:
        rng.shuffle(candidates)
        found = 0
        for relative in candidates:
            source = (rgb_roots[name] / relative).resolve()
            if not source.is_file():
                raise FileNotFoundError(f"RGB missing: {source}. Check dataset YAML paths or extract the data first.")
            identity = hashlib.sha256(source.read_bytes()).hexdigest()
            if identity in seen:
                continue
            seen.add(identity)
            selected.append((split, name, source, identity))
            found += 1
            if found == count:
                break
        if found < count:
            raise ValueError(f"Only {found} unique {name}/{split} images, requested {count}")
    # Only create output after checking every selected input. No source edits.
    for split in ("train", "val"):
        (destination / split).mkdir(parents=True)
    for split, name, source, identity in selected:
        shutil.copyfile(source, destination / split / f"{name}_{identity}{source.suffix.lower()}")
    print(f"Prepared {sum(counts)} RGB images in {destination}; train={outdoor_train + indoor_train}, val={outdoor_val + indoor_val}")
    print("These are held out from decoder calibration, not necessarily from the original VGC training.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_data_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--dataset_config", default="config/dataset/dataset_train.yaml")
    parser.add_argument("--vkitti_val_list", default="data_split/vkitti/vkitti_val.txt")
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--outdoor_train", type=int, default=160)
    parser.add_argument("--outdoor_val", type=int, default=40)
    parser.add_argument("--indoor_train", type=int, default=40)
    parser.add_argument("--indoor_val", type=int, default=10)
    args = parser.parse_args()
    prepare(**vars(args))


if __name__ == "__main__":
    main()
