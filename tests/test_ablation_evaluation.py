import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import subprocess
import sys

import numpy as np
from PIL import Image


spec = importlib.util.spec_from_file_location("eval_ablations", Path("script/eval_ablations.py"))
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class AblationEvaluationTest(unittest.TestCase):
    def test_main_uses_five_datasets_and_ablations_only_two(self):
        ids = sorted(p.stem for p in Path("config/ablation").glob("*.yaml"))
        plan = list(runner.evaluation_plan(ids, main_extra=True))
        self.assertEqual(len(ids), 11)
        self.assertEqual(len(plan), 25)
        self.assertEqual({ds for name, ds in plan if name == "00_full"}, set(runner.DATASETS))
        for name in ids[1:]:
            self.assertEqual({ds for experiment, ds in plan if experiment == name}, {"nyu", "kitti"})

    def test_csv_scales_metrics_once_and_missing_results_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "00_full" / "nyu" / "eval"
            path.mkdir(parents=True)
            (path / "metrics.json").write_text(json.dumps({
                "alignment": "least_square", "samples": 654,
                "metrics": {"abs_relative_difference": .056, "delta1_acc": .965},
            }), encoding="utf-8")
            destination = runner.collect_results([("00_full", "nyu")], root)
            with destination.open(newline="", encoding="utf-8") as handle:
                row = next(csv.DictReader(handle))
            self.assertAlmostEqual(float(row["AbsRel_x100"]), 5.6)
            self.assertAlmostEqual(float(row["delta1_percent"]), 96.5)
            with self.assertRaises(FileNotFoundError):
                runner.collect_results([("00_full", "kitti")], root)

    def test_eval_cli_runs_on_cpu_and_writes_metrics_json(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "nyu"
            data.mkdir()
            Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(data / "rgb_00000.png")
            depth = np.linspace(1000, 4000, 64).reshape(8, 8).astype(np.uint16)
            Image.fromarray(depth).save(data / "depth.png")
            filenames = root / "files.txt"
            filenames.write_text("rgb_00000.png depth.png depth.png\n", encoding="utf-8")
            config = root / "dataset.yaml"
            config.write_text("name: nyu_v2\ndisp_name: tiny_nyu\ndir: nyu\n"
                              f"filenames: {filenames.as_posix()}\neigen_valid_mask: false\n", encoding="utf-8")
            prediction = root / "prediction"
            prediction.mkdir()
            np.save(prediction / "pred_00000.npy", depth.astype(np.float32) / 5000)
            output = root / "eval"
            result = subprocess.run([
                sys.executable, "-B", "eval.py", "--prediction_dir", str(prediction),
                "--output_dir", str(output), "--base_data_dir", str(root),
                "--dataset_config", str(config), "--alignment", "least_square", "--no_cuda",
            ], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(metrics["samples"], 1)
            self.assertLess(metrics["metrics"]["abs_relative_difference"], 1e-5)
            self.assertEqual(metrics["metrics"]["delta1_acc"], 1.)


if __name__ == "__main__":
    unittest.main()
