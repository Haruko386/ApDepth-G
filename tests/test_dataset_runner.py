"""Server workflow commands must select SDXL and its matching checkpoint."""

import subprocess
import sys
import unittest


class DatasetRunnerTest(unittest.TestCase):
    def run_cli(self, script, *args):
        result = subprocess.run([sys.executable, "-B", script, *args],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result.stdout.splitlines()

    def test_all_five_in_order_and_matching_checkpoint(self):
        lines = self.run_cli("script/eval_datasets.py", "--checkpoint", "/base",
                             "--unet_checkpoint", "/trained", "--output_dir", "/results",
                             "--da2_config", "config/da2.yaml", "--dry_run")
        self.assertEqual(len(lines), 10)
        for infer, evaluate, name in zip(lines[::2], lines[1::2],
                                        ("nyu", "kitti", "eth3d", "scannet", "diode")):
            self.assertIn("--backbone sdxl", infer)
            self.assertIn("--unet_checkpoint", infer)
            self.assertIn("--da2_config config/da2.yaml", infer)
            self.assertIn("--processing_res 0", infer)
            self.assertNotIn("--fp16", infer)
            self.assertIn(name, infer)
            self.assertIn(name, evaluate)
            self.assertIn("--alignment least_square", evaluate)

    def test_eval_only_needs_no_weights(self):
        lines = self.run_cli("script/eval_datasets.py", "--phase", "eval",
                             "--output_dir", "/results", "--datasets", "diode", "nyu", "--dry_run")
        self.assertEqual(len(lines), 2)
        self.assertIn("diode", lines[0])
        self.assertIn("nyu", lines[1])
        self.assertNotIn("--checkpoint", " ".join(lines))

    def test_sdxl_ablation_runner_forwards_backbone(self):
        lines = self.run_cli("script/eval_ablations.py", "--backbone", "sdxl",
                             "--base_checkpoint", "/base", "--base_data_dir", "/data",
                             "--ids", "00_full", "--main_extra", "--dry_run")
        self.assertEqual(len(lines), 10)
        self.assertTrue(all("--backbone sdxl" in line for line in lines[::2]))

    def test_fp16_is_forwarded_only_when_explicit(self):
        lines = self.run_cli("script/eval_datasets.py", "--checkpoint", "/base",
                             "--unet_checkpoint", "/trained", "--output_dir", "/results",
                             "--datasets", "nyu", "--fp16", "--dry_run")
        self.assertIn("--fp16", lines[0])
        self.assertNotIn("--fp16", lines[1])


if __name__ == "__main__":
    unittest.main()
