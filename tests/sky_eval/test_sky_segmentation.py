"""Offline fixture coverage for sky-mask exports; no Mask2Former weights needed."""
from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any
import json
import tarfile
import tempfile
import unittest

import numpy as np
from PIL import Image
import yaml

from src.sky_eval.datasets import index_dataset, inspect_sample
from src.sky_eval.predictions import prediction_path
from src.sky_eval.rgb_io import RGBSource, close_rgb_sources
from src.sky_eval.segmentation import SplitPlan, generate, make_plans, mask_location, resolve_splits, sky_class_ids


def png_bytes(value: int) -> bytes:
    stream = BytesIO()
    Image.new("RGB", (12, 8), (value, 40, 80)).save(stream, format="PNG")
    return stream.getvalue()


class FixtureSegmenter:
    provenance: dict[str, Any] = {"checkpoint": "synthetic fixture", "mask_source": "mask2former_pseudo_semantic_sky"}

    def __init__(self, fail_second: bool = False) -> None:
        self.fail_second = fail_second

    def predict(self, image: Image.Image) -> np.ndarray:
        value = image.getpixel((0, 0))[0]
        if self.fail_second and value == 20:
            raise ValueError("synthetic segmentation failure")
        mask = np.zeros((image.height, image.width), dtype=bool)
        if value == 10:
            mask[:4] = True
        return mask


def make_fixture(root: Path, archive: bool = False) -> SplitPlan:
    data = root / "rgb"
    data.mkdir()
    paths = ["drive/image_02/a.png", "drive/image_02/b.png"]
    for relative, value in zip(paths, (10, 20)):
        path = data / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(png_bytes(value))
    split = root / "files.txt"
    split.write_text("\n".join(f"{path} absent_gt.png 721.5" for path in paths) + "\n", encoding="utf-8")
    ds = root / "dataset.yaml"
    ds.write_text(yaml.safe_dump(dict(name="kitti", disp_name="fixture", dir="unused.tar",
                                    filenames=str(split), kitti_bm_crop=True, valid_mask_crop="eigen")), encoding="utf-8")
    source = data
    if archive:
        source = root / "rgb.tar"
        with tarfile.open(source, "w") as stream:
            for relative in paths:
                stream.add(data / relative, arcname="./" + relative)
    return SplitPlan("kitti", source, ds, split, "fixture", paths, len(paths))


class SkySegmentationTests(unittest.TestCase):
    def tearDown(self) -> None:
        close_rgb_sources()

    def test_directory_and_tar_preserve_pixels_and_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = make_fixture(root, archive=True)
            with RGBSource(root / "rgb") as folder, RGBSource(plan.source) as archive:
                for path in plan.paths:
                    image, digest = folder.read(path)
                    other, other_digest = archive.read(path)
                    np.testing.assert_array_equal(np.asarray(image), np.asarray(other))
                    self.assertEqual(digest, other_digest)
                with self.assertRaises(ValueError):
                    folder.read("../outside.png")

    def test_duplicate_archive_names_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "duplicates.tar"
            payload = png_bytes(10)
            with tarfile.open(archive, "w") as stream:
                for name in ("a.png", "./a.png"):
                    info = tarfile.TarInfo(name)
                    info.size = len(payload)
                    stream.addfile(info, BytesIO(payload))
            with self.assertRaisesRegex(ValueError, "duplicate"):
                RGBSource(archive)

    def test_split_selection_uses_base_data_dir_and_preserves_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = make_fixture(root)
            config = {"datasets": {"kitti": {"dataset_config": str(plan.dataset_config),
                                               "directory_fallbacks": ["rgb"]}}}
            resolved = make_plans(config, root, ["kitti"], 1)[0]
            self.assertEqual(resolved.source, root / "rgb")
            self.assertEqual(resolved.paths, plan.paths[:1])
            self.assertEqual(resolved.total, 2)

    def test_class_id_is_discovered_by_label(self) -> None:
        self.assertEqual(sky_class_ids({"37": "Sky", "2": "road"}, ["sky"]), [37])
        with self.assertRaisesRegex(ValueError, "absent"):
            sky_class_ids({0: "road"}, ["sky"])

    def test_explicit_split_overrides_dataset_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = make_fixture(root)
            custom = root / "custom_subset.txt"
            custom.write_text(f"{plan.paths[1]} custom_depth.png\n", encoding="utf-8")
            config = {"datasets": {"kitti": {"dataset_config": str(plan.dataset_config),
                                               "directory_fallbacks": ["rgb"]}}}
            splits = resolve_splits([str(custom)], ["kitti"])
            selected = make_plans(config, root, ["kitti"], None, splits)[0]
            self.assertEqual(selected.paths, plan.paths[1:])
            self.assertEqual(selected.split_list, custom)
            self.assertEqual(selected.split_name, "custom_subset")
            self.assertEqual(mask_location(selected, root / "logs/run_a")[1], root / "rgb/sky_masks/run_a")

    def test_split_inference_and_dataset_cardinality(self) -> None:
        splits = resolve_splits(["data_split/kitti/eigen_test_files_with_gt.txt"], None)
        self.assertEqual(list(splits), ["kitti"])
        with self.assertRaisesRegex(ValueError, "exactly one"):
            resolve_splits(["one.txt"], ["kitti", "diode"])
        with self.assertRaisesRegex(ValueError, "Cannot infer"):
            resolve_splits(["custom_subset.txt"], None)

    def test_export_zero_sky_and_tar_scoe_integration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = make_fixture(root, archive=True)
            output = root / "output"
            self.assertEqual(generate([plan], FixtureSegmenter(), output, {}, 0, False), 0)
            cfg = yaml.safe_load((output / "sky_eval.generated.yaml").read_text(encoding="utf-8"))
            dataset = cfg["datasets"]["kitti"]
            self.assertEqual(dataset["data_split"], str(plan.split_list))
            self.assertEqual(Path(dataset["mask_root"]), root)  # tar: beside its containing archive
            samples = index_dataset("kitti", dataset)
            self.assertEqual(len(samples), 2)
            for sample in samples:
                inspected, mask = inspect_sample(sample, dataset)
                self.assertEqual(inspected, sample)
                self.assertEqual(mask.shape, (8, 12))
            self.assertEqual(samples[1].sky_ratio, 0)
            self.assertTrue((root / "sky_masks/output" / (plan.paths[0] + ".sky.png")).is_file())
            self.assertFalse((output / "kitti/masks").exists())
            inference = yaml.safe_load((output / "kitti/depth_inference.yaml").read_text(encoding="utf-8"))
            self.assertFalse(inference["kitti_bm_crop"])
            self.assertEqual(inference["filenames"], str(plan.split_list))
            predicted = prediction_path({"layout": "infer_split", "root": str(root / "pred")}, samples[0])
            self.assertEqual(predicted, root / "pred/kitti/drive/image_02/pred_a.npy")
            with self.assertRaises(FileExistsError):
                generate([plan], FixtureSegmenter(), output, {}, 0, False)
            # Close cached Windows archive handles before TemporaryDirectory cleanup.
            close_rgb_sources()

    def test_eval_rejects_changed_original_split(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = make_fixture(root)
            output = root / "metadata"
            self.assertEqual(generate([plan], FixtureSegmenter(), output, {}, 0, False), 0)
            config = yaml.safe_load((output / "sky_eval.generated.yaml").read_text(encoding="utf-8"))
            plan.split_list.write_text(plan.paths[0] + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "data_split changed"):
                index_dataset("kitti", config["datasets"]["kitti"])

    def test_failure_does_not_publish_shortened_evaluation_config(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = make_fixture(root)
            output = root / "output"
            self.assertEqual(generate([plan], FixtureSegmenter(True), output, {}, 0, False), 2)
            self.assertFalse((output / "sky_eval.generated.yaml").exists())
            rows = [json.loads(line) for line in (output / "kitti/per_image.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[1]["status"], "failed")
            self.assertIn("synthetic segmentation failure", rows[1]["failure_reason"])


if __name__ == "__main__":
    unittest.main()
