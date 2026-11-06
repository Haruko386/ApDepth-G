import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image
from scipy import ndimage

from src.util.diode_evaluation import refine_diode_valid_mask


class DIODEEvaluationTest(unittest.TestCase):
    def test_l1_response_removes_diagonal_gradients_and_preserves_inputs(self):
        y, x = np.mgrid[:8, :8]
        depth = 2.0 + .025 * (x + y)
        mask = np.ones_like(depth, dtype=bool)
        mask[0, 0] = False
        depth_before, mask_before = depth.copy(), mask.copy()
        dx, dy = ndimage.sobel(depth, 1), ndimage.sobel(depth, 0)
        # At this pixel L2 < 0.3 but L1 > 0.3: the distinction matters.
        self.assertLess(np.hypot(dx[3, 3], dy[3, 3]), .3)
        self.assertFalse(refine_diode_valid_mask(depth, mask)[3, 3])
        self.assertFalse(refine_diode_valid_mask(depth, mask)[0, 0])
        np.testing.assert_array_equal(depth, depth_before)
        np.testing.assert_array_equal(mask, mask_before)

    def test_depth_range_and_finite_values(self):
        for value, keep in [(.59, False), (.6, True), (350., True),
                            (350.1, False), (float("nan"), False), (float("inf"), False)]:
            with self.subTest(value=value):
                depth = np.full((5, 5), value)
                with np.errstate(invalid="ignore"):
                    mask = refine_diode_valid_mask(depth, np.ones((5, 5), dtype=bool))
                self.assertEqual(bool(mask.any()), keep)

    def test_eval_cli_filters_before_alignment_and_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "diode"
            data.mkdir()
            y, x = np.mgrid[:8, :8]
            depth = (2.0 + .004 * x + .003 * y).astype(np.float32)
            depth[:, 6:] += 3
            mask = np.ones((8, 8), dtype=np.uint8)
            mask[0, 0] = 0
            Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(data / "image.png")
            np.save(data / "depth.npy", depth[..., None])
            np.save(data / "mask.npy", mask)
            split = root / "files.txt"
            split.write_text("image.png depth.npy mask.npy\n", encoding="utf-8")
            config = root / "data.yaml"
            config.write_text(f"name: diode\ndisp_name: tiny_diode\ndir: diode\nfilenames: {split.as_posix()}\n", encoding="utf-8")
            prediction = root / "prediction"
            prediction.mkdir()
            predicted = depth / 20
            predicted[:, 5:7] += 20  # Only Sobel-excluded pixels are corrupted.
            np.save(prediction / "pred_image.npy", predicted)
            output = root / "evaluation"
            run = subprocess.run([
                sys.executable, "-B", "eval.py", "--dataset_config", str(config),
                "--base_data_dir", str(root), "--prediction_dir", str(prediction),
                "--output_dir", str(output), "--alignment", "least_square", "--no_cuda",
            ], capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            result = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
            self.assertLess(result["metrics"]["abs_relative_difference"], 1e-5)
            self.assertEqual(result["metrics"]["delta1_acc"], 1.)
            self.assertEqual(result["evaluation_mask"]["protocol"], "ppd_diode")
            self.assertEqual(result["evaluation_mask"]["sobel_threshold"], .3)


if __name__ == "__main__":
    unittest.main()
