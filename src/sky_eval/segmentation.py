"""Local-only Mask2Former sky pseudo-label generation for evaluation splits."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib import metadata
from pathlib import Path
from typing import Any, Protocol as TypingProtocol
import json
import logging
import os
import platform
import subprocess
import sys
import tarfile

import numpy as np
from PIL import Image
import yaml

from .datasets import VKITTI_KEY
from .rgb_io import RGBSource, normalized_relative
from .schemas import Protocol, Sample, confined, object_hash, sha256, write_json

REPO = Path(__file__).resolve().parents[2]
MASK_SOURCE = "mask2former_pseudo_semantic_sky"


def repository_path(value: str) -> Path:
    path = Path(value)
    path = (REPO / path).resolve() if not path.is_absolute() else path.resolve()
    if any(p.casefold() == "apdepth-g-paper" for p in path.parts):
        raise ValueError("Protected directory: ApDepth-G-paper")
    return path


@dataclass
class SplitPlan:
    dataset: str
    source: Path
    dataset_config: Path
    split_list: Path
    split_name: str
    paths: list[str]
    total: int
    split_sha256: str = ""
    dataset_config_sha256: str = ""


def split_paths(path: Path) -> list[str]:
    paths = [normalized_relative(line.split()[0]) for line in path.read_text(encoding="utf-8-sig").splitlines()
             if line.strip() and not line.lstrip().startswith("#")]
    if not paths or len(paths) != len(set(paths)):
        raise ValueError(f"Empty split or duplicate RGB paths: {path}")
    return paths


def resolve_splits(paths: list[str], names: list[str] | None) -> dict[str, Path]:
    """Use explicit user splits; never silently substitute dataset-config lists."""
    aliases = {"vkitti": "vkitti2", "vkitti2": "vkitti2", "kitti": "kitti", "diode": "diode"}
    resolved = [repository_path(value) for value in paths]
    if not names:
        names = []
        for path in resolved:
            matches = {aliases[part.casefold()] for part in path.parts if part.casefold() in aliases}
            if len(matches) != 1:
                raise ValueError(f"Cannot infer dataset for {path}; provide --dataset or --datasets")
            names.append(matches.pop())
    canonical = [aliases[name] for name in names]
    if len(canonical) != len(resolved) or len(set(canonical)) != len(canonical):
        raise ValueError("Supply exactly one --data_split per dataset, in the same order; dataset names must be unique")
    return dict(zip(canonical, resolved))


def mask_location(plan: SplitPlan, output: Path) -> tuple[Path, Path]:
    """Directory data: write inside the RGB root; tar data: beside the archive."""
    if not output.name:
        raise ValueError("--output_dir must have a run directory name")
    root = plan.source if plan.source.is_dir() else plan.source.parent
    return root, confined(root, f"sky_masks/{output.name}")


def make_plans(config: dict[str, Any], base: Path, names: list[str], limit: int | None,
               split_overrides: dict[str, Path] | None = None) -> list[SplitPlan]:
    base = base.resolve()
    if not base.is_dir():
        raise ValueError(f"BASE_DATA_DIR does not exist: {base}")
    plans = []
    for name in names:
        if name not in config["datasets"]:
            raise ValueError(f"Unknown dataset: {name}")
        settings = config["datasets"][name]
        dataset_config = repository_path(settings["dataset_config"])
        config_digest = sha256(dataset_config)
        ds = yaml.safe_load(dataset_config.read_text(encoding="utf-8-sig"))
        expected = {"vkitti2": "vkitti", "kitti": "kitti", "diode": "diode"}
        if ds.get("name") != expected[name]:
            raise ValueError(f"Dataset config/name mismatch: {name}: {dataset_config}")
        split_list = (split_overrides[name] if split_overrides is not None
                      else repository_path(ds["filenames"]))
        split_digest = sha256(split_list)
        paths = split_paths(split_list)
        if sha256(split_list) != split_digest or sha256(dataset_config) != config_digest:
            raise ValueError("Dataset configuration or split changed during indexing")
        # Explicit data_dir is authoritative. Otherwise follow a fixed candidate order.
        candidates = ([settings["data_dir"]] if settings.get("data_dir") else
                      [ds["dir"], *settings.get("directory_fallbacks", [])])
        selected = None
        errors = []
        for relative in dict.fromkeys(candidates):
            candidate = confined(base, normalized_relative(relative))
            if not candidate.exists():
                continue
            try:
                with RGBSource(candidate) as source:
                    if source.contains(paths[0]):
                        selected = candidate
                        break
                errors.append(f"{candidate}: first listed RGB is missing")
            except (OSError, ValueError, tarfile.TarError) as exc:
                errors.append(f"{candidate}: {exc}")
        if selected is None:
            raise ValueError(f"Cannot locate {name} RGB under {base}; candidates={candidates}; details={errors}. "
                             "Set datasets.<name>.data_dir to the directory/archive containing the split paths.")
        plans.append(SplitPlan(name, selected, dataset_config, split_list,
                               split_list.stem, paths[:limit], len(paths), split_digest, config_digest))
    return plans


def sky_class_ids(id2label: dict[Any, str], labels: list[str]) -> list[int]:
    if not labels or not all(isinstance(label, str) and label.strip() for label in labels):
        raise ValueError("sky_labels must contain explicit semantic class names")
    normalized = {int(index): label.strip().casefold() for index, label in id2label.items()}
    wanted = {label.strip().casefold() for label in labels}
    if wanted - set(normalized.values()):
        raise ValueError(f"Sky label(s) absent from model id2label: {sorted(wanted - set(normalized.values()))}")
    return sorted(index for index, label in normalized.items() if label in wanted)


class Segmenter(TypingProtocol):
    provenance: dict[str, Any]

    def predict(self, image: Image.Image) -> np.ndarray:
        """Return a boolean mask in original RGB coordinates."""
        ...


class Mask2FormerSky:
    def __init__(self, checkpoint: Path, labels: list[str], device: str, seed: int) -> None:
        checkpoint = repository_path(str(checkpoint))
        if not checkpoint.is_dir():
            raise ValueError(f"Local Mask2Former checkpoint directory required: {checkpoint}. "
                             "No automatic downloads are performed.")
        for filename in ("config.json", "preprocessor_config.json"):
            if not (checkpoint / filename).is_file():
                raise ValueError(f"Missing local checkpoint file: {filename}")
        model_config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
        if model_config.get("model_type") != "mask2former":
            raise ValueError("Expected a Transformers Mask2Former checkpoint")
        if model_config.get("use_pretrained_backbone", False):
            raise ValueError("Use a self-contained checkpoint with use_pretrained_backbone=false; nested downloads are disabled")
        sky_class_ids(model_config.get("id2label", {}), labels)
        # Enforce offline behavior for processor, model and any nested backbone load.
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        import torch
        from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor

        self.torch = torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        self.device = torch.device("cuda" if device == "auto" and torch.cuda.is_available()
                                   else "cpu" if device == "auto" else device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is unavailable; use --device cpu")
        self.processor = Mask2FormerImageProcessor.from_pretrained(str(checkpoint), local_files_only=True)
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(
            str(checkpoint), local_files_only=True).to(self.device).eval()
        self.ids = sky_class_ids(self.model.config.id2label, labels)
        # Hash local weight/config files to identify the actual segmenter, not just a display name.
        files = sorted(p for p in checkpoint.iterdir() if p.is_file() and
                       (p.suffix in (".json", ".safetensors") or p.name.startswith("pytorch_model") and p.suffix == ".bin"))
        logging.info("Recording checkpoint SHA256 fingerprints: %s", checkpoint)
        self.provenance = dict(checkpoint=str(checkpoint), checkpoint_files={p.name: sha256(p) for p in files},
                               sky_labels=labels, sky_class_ids=self.ids, id2label=self.model.config.id2label,
                               processor=self.processor.to_dict(), device=str(self.device), seed=seed,
                               dtype=str(next(self.model.parameters()).dtype), batch_size=1,
                               geometry="full_frame", postprocess="semantic argmax at original H,W",
                               mask_source=MASK_SOURCE, local_files_only=True)

    def predict(self, image: Image.Image) -> np.ndarray:
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            output = self.model(**inputs)
            semantic = self.processor.post_process_semantic_segmentation(
                output, target_sizes=[(image.height, image.width)])[0]
        labels = semantic.detach().cpu().numpy()
        return np.isin(labels, self.ids)


def sample_for(plan: SplitPlan, relative: str, digest: str, mask_digest: str,
               mask: np.ndarray, provenance: str, mask_prefix: str = "masks") -> Sample:
    identity: dict[str, str] = {}
    sid = f"{plan.dataset}/{relative}"
    if plan.dataset == "vkitti2":
        match = VKITTI_KEY.search(relative)
        if match is None:
            raise ValueError(f"Invalid VKITTI RGB path: {relative}")
        scene, variation, camera, frame = match.groups()
        identity = dict(scene_id=scene, variation=variation, camera=camera, frame_id=frame)
        sid = f"vkitti2/{scene}/{variation}/{camera}/{frame}"
    return Sample(sample_id=sid, dataset=plan.dataset, split=plan.split_name, rgb_path=relative,
                  sky_mask_path=f"{mask_prefix}/{relative}.sky.png", mask_source=MASK_SOURCE,
                  split_provenance=provenance, width=mask.shape[1], height=mask.shape[0],
                  sky_ratio=float(mask.mean()), rgb_sha256=digest, mask_sha256=mask_digest, **identity)


def save_overlay(image: Image.Image, mask: np.ndarray, path: Path) -> None:
    rgb = np.asarray(image).copy()
    rgb[mask] = (0.55 * rgb[mask] + 0.45 * np.array([40, 170, 255])).astype(np.uint8)
    overlay = Image.fromarray(rgb)
    overlay.thumbnail((1280, 1280))
    with path.open("xb") as stream:
        overlay.save(stream, format="PNG")


def generate(plans: list[SplitPlan], segmenter: Segmenter, output: Path,
             config: dict[str, Any], preview_count: int, limited: bool) -> int:
    """All selected images are recorded. Any failure prevents SCOE-config export."""
    if output.exists():
        raise FileExistsError(f"Output already exists; choose a new --output_dir: {output}")
    destinations = [mask_location(plan, output)[1] for plan in plans]
    if len(set(destinations)) != len(destinations):
        raise ValueError("Datasets resolve to the same mask directory; use distinct dataset roots or separate runs")
    for destination in destinations:
        if destination.exists():
            raise FileExistsError(f"Dataset mask directory exists; choose a new run name: {destination}")
    output.mkdir(parents=True, exist_ok=False)
    versions: dict[str, str] = {"python": platform.python_version()}
    for package in ("torch", "transformers", "numpy", "Pillow", "PyYAML"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "not installed"
    try:
        git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=False)
        commit = git.stdout.strip() if git.returncode == 0 else "unavailable"
    except OSError:
        commit = "unavailable"
    write_json(output / "run_metadata.json", dict(config=config, model=segmenter.provenance,
               environment=versions, limited=limited, mask_values={"sky": 255, "non_sky": 0},
               repository_commit=commit, invocation=sys.argv,
               source_hashes={p.name: sha256(p) for p in [Path(__file__), Path(__file__).with_name("rgb_io.py"),
                                                         REPO / "script/segment_sky.py"]}))
    evaluation: dict[str, Any] = dict(protocol=asdict(Protocol()), datasets={}, models={},
                                     cross_dataset_duplicate_reviewed=False,
                                     output=dict(root=str(output / "evaluation"), run_id="scoe_mask2former_v1"))
    summary: dict[str, Any] = {"status": "complete", "datasets": {}, "mask_source": MASK_SOURCE, "limited": limited}
    for plan in plans:
        logging.info("Segmenting %s: %d/%d images from %s", plan.dataset, len(plan.paths), plan.total, plan.source)
        folder = output / plan.dataset
        folder.mkdir()
        mask_root, mask_directory = mask_location(plan, output)
        mask_directory.mkdir(parents=True, exist_ok=False)
        mask_prefix = f"sky_masks/{output.name}"
        logging.info("%s: data_split=%s; masks=%s", plan.dataset, plan.split_list, mask_directory)
        previews = folder / "previews"
        previews.mkdir()
        selected_previews = set(np.linspace(0, len(plan.paths) - 1, min(preview_count, len(plan.paths)), dtype=int).tolist())
        counts = dict(total_split=plan.total, requested=len(plan.paths), succeeded=0, failed=0, zero_sky=0,
                      at_least_five_percent_sky=0)
        split_digest = sha256(plan.split_list)
        config_digest = sha256(plan.dataset_config)
        if (plan.split_sha256 and split_digest != plan.split_sha256) or (
                plan.dataset_config_sha256 and config_digest != plan.dataset_config_sha256):
            raise ValueError(f"Dataset config or split changed after indexing: {plan.dataset}")
        provenance = f"{plan.split_name}; list={plan.split_list}; sha256={split_digest}"
        records = folder / "per_image.jsonl"
        manifest = folder / "manifest.jsonl"
        with RGBSource(plan.source) as source, records.open("x", encoding="utf-8") as log, manifest.open("x", encoding="utf-8") as index:
            for number, relative in enumerate(plan.paths):
                record: dict[str, Any] = dict(rgb_path=relative, status="failed", failure_reason=None)
                try:
                    image, digest = source.read(relative)
                    mask = segmenter.predict(image)
                    if mask.dtype != np.bool_ or mask.shape != (image.height, image.width):
                        raise ValueError("Segmenter returned an invalid binary mask or wrong dimensions")
                    mask_path = confined(mask_root, f"{mask_prefix}/{relative}.sky.png")
                    mask_path.parent.mkdir(parents=True, exist_ok=True)
                    with mask_path.open("xb") as stream:
                        Image.fromarray(mask.astype(np.uint8) * 255).save(stream, format="PNG")
                    sample = sample_for(plan, relative, digest, sha256(mask_path), mask, provenance, mask_prefix)
                    index.write(json.dumps(sample.to_dict(), ensure_ascii=False) + "\n")
                    counts["succeeded"] += 1
                    counts["zero_sky"] += int(not mask.any())
                    counts["at_least_five_percent_sky"] += int(mask.mean() >= 0.05)
                    record.update(sample.to_dict(), status="ok")
                    if number in selected_previews:
                        save_overlay(image, mask, previews / f"{object_hash(sample.sample_id)}.png")
                except (OSError, ValueError, RuntimeError) as exc:
                    # CUDA/model failures must not silently produce a shorter evaluation split.
                    if record["status"] == "ok":
                        raise  # A preview/output failure after manifest publication aborts this run.
                    counts["failed"] += 1
                    record["failure_reason"] = str(exc)
                    logging.error("%s/%s: %s", plan.dataset, relative, exc)
                log.write(json.dumps(record, ensure_ascii=False) + "\n")
                log.flush()
                index.flush()
                if (number + 1) % 25 == 0 or number + 1 == len(plan.paths):
                    logging.info("%s: %d/%d, failures=%d", plan.dataset, number + 1, len(plan.paths), counts["failed"])
        if sha256(plan.split_list) != split_digest:
            raise ValueError(f"Split list changed while segmenting: {plan.split_list}")
        if sha256(plan.dataset_config) != config_digest:
            raise ValueError(f"Dataset config changed while segmenting: {plan.dataset_config}")
        report = dict(**counts, dataset=plan.dataset, mask_source=MASK_SOURCE, limited=limited,
                      rgb_root=str(plan.source), split_list=str(plan.split_list), split_list_sha256=split_digest,
                      mask_root=str(mask_root), mask_directory=str(mask_directory),
                      dataset_config=str(plan.dataset_config), dataset_config_sha256=config_digest,
                      manifest_sha256=sha256(manifest), per_image_sha256=sha256(records),
                      run_metadata_sha256=sha256(output / "run_metadata.json"))
        write_json(folder / "segmentation_report.json", report)
        # A separate full-frame config reuses infer.py without editing benchmark configs.
        raw_lines = plan.split_list.read_text(encoding="utf-8-sig").splitlines()
        source_lines = {normalized_relative(line.split()[0]): line for line in raw_lines
                        if line.strip() and not line.lstrip().startswith("#")}
        inference_files = folder / "depth_inference_files.txt"
        with inference_files.open("x", encoding="utf-8") as stream:
            stream.write("\n".join(source_lines[path] for path in plan.paths) + "\n")
        inference_config = yaml.safe_load(plan.dataset_config.read_text(encoding="utf-8-sig"))
        # Legacy infer.py does not skip comments/blank lines; a snapshot keeps
        # the exact selected data rows while leaving the user's source untouched.
        use_snapshot = limited or any(not line.strip() or line.lstrip().startswith("#") for line in raw_lines)
        inference_config.update(dir=str(plan.source), filenames=str(inference_files if use_snapshot else plan.split_list),
                                disp_name=f"{plan.split_name}_sky_full_frame")
        if plan.dataset in ("vkitti2", "kitti"):
            inference_config.update(kitti_bm_crop=False, valid_mask_crop=None)
        with (folder / "depth_inference.yaml").open("x", encoding="utf-8") as stream:
            yaml.safe_dump(inference_config, stream, allow_unicode=True, sort_keys=False)
        summary["datasets"][plan.dataset] = {**counts, "data_split": str(plan.split_list),
                                               "mask_directory": str(mask_directory)}
        if counts["failed"]:
            summary["status"] = "failed"
        entry = dict(enabled=True, mask_format="binary_manifest", rgb_root=str(plan.source), mask_root=str(mask_root),
                     data_split=str(plan.split_list),
                     manifest=str(manifest), segmentation_report=str(folder / "segmentation_report.json"),
                     sky_values=[255], non_sky_values=[0], split_provenance=provenance,
                     mask_provenance=f"Mask2Former pseudo labels; see {output / 'run_metadata.json'}", qa_reviewed=False)
        if plan.dataset == "vkitti2":
            train_list = config["datasets"]["vkitti2"]["train_list"]
            entry.update(train_list=str(repository_path(train_list)), test_list=str(plan.split_list))
        evaluation["datasets"][plan.dataset] = entry
    write_json(output / "summary.json", summary)
    if summary["status"] != "complete":
        logging.error("Segmentation incomplete: no sky_eval.generated.yaml exported; inspect per_image.jsonl")
        return 2
    with (output / "sky_eval.generated.yaml").open("x", encoding="utf-8") as stream:
        yaml.safe_dump(evaluation, stream, allow_unicode=True, sort_keys=False)
    logging.info("Masks and SCOE diagnostic configuration written to %s", output)
    return 0
