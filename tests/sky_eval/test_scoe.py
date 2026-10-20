"""SCOE regression/property tests. Synthetic fixtures are never paper results.

Run explicitly: python -B -m unittest discover -s tests/sky_eval -v
These tests were authored but NOT executed in the implementation session.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any
import csv
import json
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image
import yaml
from scipy import ndimage

from src.sky_eval.aggregate import aggregate, compare
from src.sky_eval.audit import split_audit
from src.sky_eval.datasets import inspect_sample, load_mask, read_colors
from src.sky_eval.masks import Regions, build_regions
from src.sky_eval.metrics import METRICS, score
from src.sky_eval.predictions import read_prediction, validate_model
from src.sky_eval.schemas import Protocol, Sample, confined
from src.sky_eval.workflow import formal_blockers

REPO = Path(__file__).resolve().parents[2]


def plane() -> tuple[np.ndarray, np.ndarray, Regions]:
    sky = np.zeros((120, 160), dtype=bool)
    sky[:60] = True
    d = np.broadcast_to(np.linspace(1, 4, 160), sky.shape).copy()
    d[sky] = 10
    return sky, d, build_regions(sky, Protocol())


class MetricTests(unittest.TestCase):
    def test_constant_far_near_and_ties(self) -> None:
        sky, d, regions = plane()
        for value, expected in ((10, 0), (-1, 1)):
            d[sky] = value
            result = score(d, regions, Protocol())
            self.assertEqual([result[m] for m in METRICS], [0, 0, expected])
        result = score(np.ones_like(d), regions, Protocol())
        self.assertIsNone(result["sky_iqr"])
        self.assertIn("degenerate_foreground_range", result["failure_reason"])
        self.assertEqual(result["sfoe"], 1)

    def test_positive_affine_invariance(self) -> None:
        sky, d, regions = plane()
        rng = np.random.default_rng(42)
        d[sky] = rng.uniform(0, 12, sky.sum())
        original = score(d, regions, Protocol())
        for a, b in ((0.001, 17), (2.5, -30), (1e5, 1e6)):
            transformed = score(a * d + b, regions, Protocol())
            for metric in METRICS:
                self.assertAlmostEqual(original[metric], transformed[metric], places=8)

    def test_ten_percent_collapse_and_denominator_gaming(self) -> None:
        sky, d, regions = plane()
        d[:6] = -1
        collapsed = score(d, regions, Protocol())
        self.assertEqual(collapsed["sky_iqr"], 0)
        self.assertGreater(collapsed["sky_p95_p5"], 1)
        d[~sky] *= 10
        expanded = score(d, regions, Protocol())
        self.assertAlmostEqual(expanded["sky_p95_p5"], collapsed["sky_p95_p5"] / 10)
        self.assertAlmostEqual(expanded["reference_range"], collapsed["reference_range"] * 10)

    def test_degenerate_masks_and_metric_specific_counts(self) -> None:
        sky, d, _ = plane()
        for mask, reason in ((np.zeros_like(sky), "insufficient_sky_area"),
                             (np.ones_like(sky), "no_foreground")):
            result = score(d, build_regions(mask, Protocol()), Protocol())
            self.assertIn(reason, result["failure_reason"])
            self.assertTrue(all(result[m] is None for m in METRICS))
        tiny = np.zeros_like(sky)
        tiny[0] = True
        self.assertIn("insufficient_sky_area", score(d, build_regions(tiny, Protocol()), Protocol())["failure_reason"])
        thin = np.zeros_like(sky)
        thin[::10] = True
        result = score(d, build_regions(thin, Protocol()), Protocol())
        self.assertIn("insufficient_sky_interior", result["failure_reason"])
        protocol = replace(Protocol(), min_boundary_pairs=1000)
        result = score(d, build_regions(sky, protocol), protocol)
        self.assertEqual(result["sky_iqr"], 0)
        self.assertIsNone(result["sfoe"])

    def test_finite_and_shape_errors(self) -> None:
        _, d, regions = plane()
        for value in (float("nan"), float("inf")):
            bad = d.copy()
            bad[0, 0] = value
            with self.assertRaisesRegex(ValueError, "nonfinite_prediction"):
                score(bad, regions, Protocol())
        with self.assertRaisesRegex(ValueError, "shape"):
            score(d[:-1], regions, Protocol())

    def test_macro_mean_and_shared_metric_intersection(self) -> None:
        rows = [{"sample_id": "a", "dataset": "synthetic", "sky_iqr": 0.0, "sky_p95_p5": 0.0,
                 "sfoe": 0.0, "n_sky_pixels": 10, "reference_range": 2},
                {"sample_id": "b", "dataset": "synthetic", "sky_iqr": 1.0, "sky_p95_p5": 2.0,
                 "sfoe": 1.0, "n_sky_pixels": 10000, "reference_range": 4}]
        self.assertEqual(aggregate(rows)["synthetic"]["sky_iqr"]["mean"], 0.5)
        other = [dict(r) for r in rows]
        other[0]["sfoe"] = None
        result = compare({"a": rows, "b": other}, True)
        self.assertTrue(all(r["N"] == 1 for r in result if r["metric"] == "sfoe"))
        self.assertTrue(all(r["N"] == 2 for r in result if r["metric"] == "sky_iqr"))
        result = compare({"a": rows, "b": other}, False)
        self.assertTrue(all(r["mean"] is None for r in result if r["metric"] != "sfoe"))


class MaskTests(unittest.TestCase):
    def test_normals_border_cache_and_determinism(self) -> None:
        sky, _, regions = plane()
        self.assertGreaterEqual(regions.n_pairs, 50)
        self.assertTrue(regions.interior[0].all())  # exterior is not foreground
        np.testing.assert_allclose(regions.normals[:, 0], 1)
        np.testing.assert_allclose(regions.normals[:, 1], 0)
        self.assertTrue(sky.ravel()[regions.sky_indices].all())
        self.assertFalse(sky.ravel()[regions.foreground_indices].any())
        for mask, indices in ((sky, regions.sky_indices), (~sky, regions.foreground_indices)):
            distances = ndimage.distance_transform_edt(mask).ravel()[indices]
            self.assertTrue(((distances >= 2) & (distances <= 3)).all())
        repeated = build_regions(sky, Protocol())
        for key in regions.__dict__:
            np.testing.assert_array_equal(getattr(regions, key), getattr(repeated, key))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pairs.npz"
            regions.save(path)
            loaded = Regions.load(path)
            np.testing.assert_array_equal(loaded.sky_indices, regions.sky_indices)

    def test_holes_branches_and_all_sky(self) -> None:
        sky, _, _ = plane()
        sky[20:50, 70:90] = False
        sky[10:60, 110] = False
        sky[45:48, 100:140] = False
        regions = build_regions(sky, Protocol())
        self.assertTrue(sky.ravel()[regions.sky_indices].all())
        self.assertFalse(sky.ravel()[regions.foreground_indices].any())
        self.assertEqual(build_regions(np.ones_like(sky), Protocol()).n_pairs, 0)


class DatasetTests(unittest.TestCase):
    def test_vkitti_palette_rgb_bgr_and_unknown(self) -> None:
        # Official palette values, synthetic geometry; not an actual downloaded VKITTI sample.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "colors.txt").write_text("Category r g b\nSky 90 200 255\nRoad 100 60 100\n", encoding="utf-8")
            values = np.full((8, 12, 3), (100, 60, 100), dtype=np.uint8)
            values[:4] = (90, 200, 255)
            Image.fromarray(values).save(root / "mask.png")
            sample = Sample("v", "vkitti2", "fixture", "rgb.png", "mask.png",
                            "official_palette_fixture", "synthetic", colors_path="colors.txt")
            cfg = {"root": directory}
            mask = load_mask(sample, cfg)
            self.assertEqual(mask.sum(), 48)
            Image.fromarray(values[..., ::-1]).save(root / "mask.png")
            with self.assertRaisesRegex(ValueError, "unknown_semantic_colors"):
                load_mask(sample, cfg)
            (root / "colors.txt").write_text("Sky 255 200 90\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                read_colors(root / "colors.txt")

    def test_split_leakage_at_scene_level(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train, test = root / "train.txt", root / "test.txt"
            train.write_text("rgb/Scene01/clone/frames/rgb/Camera_0/rgb_00000.jpg depth.png\n", encoding="utf-8")
            test.write_text("rgb/Scene01/rain/frames/rgb/Camera_1/rgb_00003.jpg depth.png\n", encoding="utf-8")
            result = split_audit(train, test)
            self.assertEqual(result["exact_overlap_count"], 0)
            self.assertEqual(result["scene_overlap"], ["Scene01"])
            self.assertEqual(result["status"], "BLOCKED")

    def test_formal_report_preserves_scene_leakage_blocker(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "dataset_audit.json").write_text(json.dumps({
                "datasets": {"hrwsi": {}, "vkitti2": {}},
                "blockers": ["vkitti2: seen_scene ['Scene01']"]}), encoding="utf-8")
            cfg = {"models": {}, "datasets": {}, "cross_dataset_duplicate_reviewed": True}
            self.assertIn("vkitti2: seen_scene ['Scene01']", formal_blockers(cfg, root, {}))

    def test_prediction_contracts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "d.npy"
            model = {"distance_direction": "smaller_is_farther"}
            values = np.arange(12, dtype=np.float32).reshape(3, 4)
            np.save(path, values)
            read, _ = read_prediction(path, model, (3, 4))
            np.testing.assert_array_equal(read, -values)
            np.save(path, values[None])
            with self.assertRaisesRegex(ValueError, "shape"):
                read_prediction(path, model, (3, 4))
            model["squeeze_leading_channel"] = True
            self.assertEqual(read_prediction(path, model, (3, 4))[0].shape, (3, 4))
            np.save(path, values.astype(np.uint8))
            with self.assertRaisesRegex(ValueError, "dtype"):
                read_prediction(path, model, (3, 4))
            with self.assertRaisesRegex(ValueError, "missing_prediction"):
                read_prediction(path.with_name("missing.npy"), model, (3, 4))
            with self.assertRaises(ValueError):
                confined(Path(directory), "../outside")


def fixture(root: Path) -> tuple[Path, dict[str, Any]]:
    data = root / "data"
    data.mkdir()
    records = []
    models: dict[str, Any] = {}
    for name in ("far", "near"):
        model_root = root / name
        model_root.mkdir()
        models[name] = dict(root=str(model_root), layout="explicit", files={}, representation="synthetic_depth",
                            distance_direction="larger_is_farther", semantics_source="synthetic analytical fixture",
                            amplitude_comparability_verified=True, checkpoint="synthetic", inference_command="fixture",
                            git_commit="synthetic", seed=0, resolution="original", denoise_steps=None,
                            ensemble_size=None, scheduler="not applicable", inference_config="synthetic",
                            native_postprocessing="none", geometry="full_frame")
    for index in range(4):
        sky, depth, _ = plane()
        rgb = np.zeros((*sky.shape, 3), dtype=np.uint8)
        rgb[:] = (30 + index * 20, 100, 150)
        Image.fromarray(rgb).save(data / f"{index}.png")
        Image.fromarray(sky.astype(np.uint8) * 255).save(data / f"{index}_sky.png")
        sample = Sample(f"synthetic/{index}", "synthetic", "synthetic_fixture", f"{index}.png",
                        f"{index}_sky.png", "synthetic_semantic", "analytical fixture")
        records.append(sample.to_dict())
        for name in models:
            prediction = depth.copy()
            prediction[sky] = 10 if name == "far" else -1
            np.save(root / name / f"{index}.npy", prediction.astype(np.float32))
            models[name]["files"][sample.sample_id] = f"{index}.npy"
    manifest = root / "input.jsonl"
    manifest.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    cfg = dict(datasets={"synthetic": dict(root=str(data), manifest=str(manifest), sky_values=[255],
                non_sky_values=[0], split_provenance="synthetic", mask_provenance="synthetic", qa_reviewed=False)},
               models=models, output=dict(root=str(root / "output"), run_id="fixture"))
    path = root / "config.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path, cfg


class EndToEndTests(unittest.TestCase):
    def cli(self, command: str, config: Path, *args: str, expected: int = 0) -> subprocess.CompletedProcess[str]:
        result = subprocess.run([sys.executable, "-B", str(REPO / "script/sky_eval.py"), command,
                                 "--config", str(config), *args], cwd=REPO, capture_output=True, text=True)
        self.assertEqual(result.returncode, expected, result.stdout + result.stderr)
        return result

    def test_four_images_two_models_csv_json_tex_and_no_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config, _ = fixture(Path(directory))
            self.cli("audit", config, expected=2)  # synthetic is never a passed real-data audit
            self.cli("prepare", config)
            for model in ("far", "near"):
                self.cli("evaluate", config, "--model", model)
            self.cli("report", config)
            run = Path(directory) / "output/fixture"
            table = json.loads((run / "diagnostic_report/comparison.json").read_text(encoding="utf-8"))
            with (run / "diagnostic_report/comparison.csv").open(encoding="utf-8") as stream:
                csv_rows = list(csv.DictReader(stream))
            self.assertEqual(len(table), len(csv_rows))
            for row, csv_row in zip(table, csv_rows):
                self.assertEqual(row["mean"], float(csv_row["mean"]))
                self.assertEqual(row["N"], 4)
            self.assertEqual(next(r["mean"] for r in table if r["model_id"] == "near" and r["metric"] == "sfoe"), 1)
            tex = (run / "diagnostic_report/diagnostic_table.tex").read_text(encoding="utf-8")
            self.assertIn("DIAGNOSTIC ONLY", tex)
            self.assertIn("near & 0 & 0 & 1", tex)
            self.assertFalse((run / "diagnostic_report/paper_table.tex").exists())
            self.cli("audit", config, expected=2)
            self.cli("evaluate", config, "--model", "far", expected=2)
            self.cli("report", config, "--formal", expected=2)
            self.assertFalse((run / "formal_report/paper_table.tex").exists())

    def test_missing_prediction_record_and_shared_intersection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, _ = fixture(root)
            (root / "near/0.npy").unlink()
            self.cli("audit", config, expected=2)
            self.cli("prepare", config)
            self.cli("evaluate", config, "--model", "far")
            self.cli("evaluate", config, "--model", "near", expected=2)
            self.cli("report", config)
            report = json.loads((root / "output/fixture/diagnostic_report/comparison.json").read_text())
            self.assertTrue(all(r["N"] == 3 for r in report))
            rows = json.loads((root / "output/fixture/models/near/per_image.json").read_text())
            self.assertEqual(len(rows), 4)
            self.assertIn("missing_prediction", rows[0]["failure_reason"])

    def test_lock_rejects_cache_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, _ = fixture(root)
            self.cli("audit", config, expected=2)
            self.cli("prepare", config)
            cache = next((root / "output/fixture/sample_pairs").glob("*.npz"))
            with cache.open("ab") as stream:
                stream.write(b"tamper")
            result = self.cli("evaluate", config, "--model", "far", expected=2)
            self.assertIn("Frozen artifact changed", result.stderr)

    def test_changed_config_and_incomplete_model_set_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, cfg = fixture(root)
            self.cli("audit", config, expected=2)
            self.cli("prepare", config)
            self.cli("evaluate", config, "--model", "far")
            self.cli("report", config, expected=2)
            cfg["protocol"] = {"min_boundary_pairs": 1}
            config.write_text(yaml.safe_dump(cfg), encoding="utf-8")
            result = self.cli("evaluate", config, "--model", "near", expected=2)
            self.assertIn("Configuration changed", result.stderr)

    def test_dry_run_writes_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, _ = fixture(root)
            self.cli("audit", config, "--dry-run")
            self.assertFalse((root / "output").exists())


if __name__ == "__main__":
    unittest.main()
