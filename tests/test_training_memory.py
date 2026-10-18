"""CUDA memory reporting must resolve implicit devices before querying PyTorch."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from omegaconf import OmegaConf
import torch

from src.trainer.marigold_trainer import MarigoldTrainer


class TrainingMemoryTest(unittest.TestCase):
    def test_memory_report_uses_explicit_or_current_device_and_writes_profile(self):
        for device, expected in [("cuda", 3), ("cuda:0", 0), ("cuda:1", 1)]:
            with self.subTest(device=device), tempfile.TemporaryDirectory() as directory:
                trainer = object.__new__(MarigoldTrainer)
                trainer.device = torch.device(device)
                trainer.effective_iter = 2
                trainer.accumulation_steps = 42
                trainer.model = MagicMock(unet=torch.nn.Linear(2, 1))
                trainer.cfg = OmegaConf.create({
                    "model": {"backbone": "sdxl"}, "optimizer": {"name": "Adam8bit"},
                    "trainer": {"mixed_precision": "no", "gradient_checkpointing": True},
                    "dataloader": {"max_train_batch_size": 1},
                })
                checkpoint_dir = Path(directory) / "checkpoint"
                checkpoint_dir.mkdir()
                trainer.out_dir_ckpt = str(checkpoint_dir)

                def memory_info(index):
                    # Enforce the argument contract that failed on the server.
                    self.assertIsInstance(index, int)
                    self.assertEqual(index, expected)
                    return 5 * 1024 ** 3, 48 * 1024 ** 3

                mocks = {
                    "current_device": MagicMock(return_value=3),
                    "synchronize": MagicMock(),
                    "mem_get_info": MagicMock(side_effect=memory_info),
                    "get_device_name": MagicMock(return_value="test GPU"),
                    "memory_allocated": MagicMock(return_value=2 * 1024 ** 3),
                    "max_memory_allocated": MagicMock(return_value=4 * 1024 ** 3),
                    "max_memory_reserved": MagicMock(return_value=6 * 1024 ** 3),
                }
                with patch.multiple(torch.cuda, **mocks):
                    trainer._report_memory()
                for name, mock in mocks.items():
                    if name != "current_device":
                        mock.assert_called_once_with(expected)
                if device == "cuda":
                    mocks["current_device"].assert_called_once_with()
                else:
                    mocks["current_device"].assert_not_called()
                record = json.loads((Path(directory) / "memory_profile.json").read_text())
                self.assertEqual(record["effective_iter"], 2)
                self.assertEqual(record["peak_allocated_gib"], 4.)
                self.assertEqual(record["device_free_gib"], 5.)


if __name__ == "__main__":
    unittest.main()
