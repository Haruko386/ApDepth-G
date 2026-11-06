"""Host RAM savings must preserve FP32 samples, worker loading and resume."""

import pickle
import tarfile
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import DataLoader

from src.dataset.base_depth_dataset import BaseDepthDataset, DatasetMode, DepthFileNameMode
from src.util.data_loader import skip_first_batches, training_loader_kwargs
from src.util.depth_transform import ScaleShiftDepthNormalizer
from src.util.host_memory import host_memory_snapshot


class HostMemoryOptimizationTest(unittest.TestCase):
    def make_dataset(self, root, fields=None, mode=DatasetMode.TRAIN, archive=False):
        rgb = np.arange(12 * 16 * 3, dtype=np.uint8).reshape(12, 16, 3)
        depth = np.arange(12 * 16, dtype=np.uint16).reshape(12, 16)
        Image.fromarray(rgb).save(root / "rgb.png")
        Image.fromarray(depth).save(root / "depth.png")
        (root / "files.txt").write_text("rgb.png depth.png\n" * 3)
        dataset_dir = root
        if archive:
            dataset_dir = root / "data.tar"
            with tarfile.open(dataset_dir, "w") as tar:
                for name in ("rgb.png", "depth.png"):
                    tar.add(root / name, arcname="./" + name)
        return BaseDepthDataset(
            mode=mode, filename_ls_path=root / "files.txt", dataset_dir=str(dataset_dir),
            disp_name="test", min_depth=0, max_depth=150, has_filled_depth=False,
            name_mode=DepthFileNameMode.id, depth_transform=ScaleShiftDepthNormalizer(),
            augmentation_args=OmegaConf.create({"lr_flip_p": 1.0}),
            resize_to_hw=(8, 8), training_output_fields=fields,
        )

    def test_compact_samples_match_full_samples_exactly_and_reduce_bytes(self):
        for depth_key, mask_key in (("depth_raw_norm", "valid_mask_raw"),
                                    ("depth_filled_norm", "valid_mask_filled")):
            for archive in (False, True):
                with self.subTest(depth=depth_key, tar=archive), tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    full_dataset = self.make_dataset(root, archive=archive)
                    full = full_dataset[0]
                    decoded = np.arange(12 * 16 * 3, dtype=np.uint8).reshape(12, 16, 3)
                    legacy_rgb = decoded.transpose(2, 0, 1).astype(int) / 255.0 * 2.0 - 1.0
                    torch.testing.assert_close(
                        full_dataset._load_rgb_data("rgb.png")["rgb_norm"],
                        torch.from_numpy(legacy_rgb).float(), rtol=0, atol=0,
                    )
                    # Close the reader before rebuilding the fixture archive on Windows.
                    full_dataset.__del__()
                    fields = ["rgb_norm", depth_key, mask_key]
                    compact_dataset = self.make_dataset(root, fields, archive=archive)
                    compact = compact_dataset[0]
                    for key in fields:
                        torch.testing.assert_close(compact[key], full[key], rtol=0, atol=0)
                    self.assertEqual(compact["rgb_norm"].dtype, torch.float32)
                    self.assertEqual(compact[depth_key].dtype, torch.float32)
                    self.assertEqual(set(compact), set(fields) | {"index", "rgb_relative_path", "dataset_name"})
                    size = lambda sample: sum(v.numel() * v.element_size() for v in sample.values()
                                              if isinstance(v, torch.Tensor))
                    self.assertEqual(size(compact), 17 * 8 * 8)
                    self.assertEqual(size(full), 42 * 8 * 8)
                    compact_dataset.__del__()

    def test_eval_retains_all_rasters(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = self.make_dataset(Path(directory), ["rgb_norm"], DatasetMode.EVAL)
            sample = dataset[0]
            self.assertIn("rgb_int", sample)
            self.assertIn("depth_raw_linear", sample)
            self.assertIn("valid_mask_raw", sample)

    def test_zero_and_spawned_workers_preserve_skip_and_prefetch(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = self.make_dataset(Path(directory), ["rgb_norm", "depth_raw_norm", "valid_mask_raw"])
            pickle.dumps(dataset)  # Default RGB transform must be spawn-safe.
            for workers in (0, 1):
                with self.subTest(workers=workers):
                    kwargs = training_loader_kwargs({"num_workers": workers})
                    if workers:
                        kwargs["multiprocessing_context"] = "spawn"
                    loader = DataLoader(dataset, batch_size=1, **kwargs)
                    resumed = skip_first_batches(loader, 1)
                    self.assertEqual(resumed.num_workers, workers)
                    self.assertEqual(resumed.prefetch_factor, 1 if workers else None)
                    self.assertEqual([batch["index"].item() for batch in resumed], [1, 2])

    def test_invalid_worker_settings_fail_before_training(self):
        for cfg in ({"num_workers": -1}, {"prefetch_factor": 0}):
            with self.assertRaises(ValueError):
                training_loader_kwargs(cfg)

    def test_linux_memory_reports_host_and_container_limits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "self").mkdir()
            (root / "self/status").write_text("VmRSS: 1048576 kB\nVmHWM: 2097152 kB\n")
            (root / "meminfo").write_text("MemAvailable: 4194304 kB\nMemTotal: 8388608 kB\n")
            (root / "memory.current").write_text(str(3 * 1024 ** 3))
            (root / "memory.max").write_text(str(6 * 1024 ** 3))
            snapshot = host_memory_snapshot(root, root)
            self.assertEqual(snapshot["process_rss_gib"], 1)
            self.assertEqual(snapshot["process_peak_rss_gib"], 2)
            self.assertEqual(snapshot["host_available_gib"], 4)
            self.assertEqual(snapshot["container_used_gib"], 3)
            self.assertEqual(snapshot["container_limit_gib"], 6)
            (root / "memory.max").write_text("max")
            self.assertNotIn("container_limit_gib", host_memory_snapshot(root, root))
            self.assertEqual(host_memory_snapshot(root / "missing", root / "missing"), {})


if __name__ == "__main__":
    unittest.main()
