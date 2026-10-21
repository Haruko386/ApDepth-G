"""Gradient-scaler overflow must not consume an extra scheduled update."""

import unittest
from unittest.mock import MagicMock

import torch

from src.trainer.marigold_trainer import MarigoldTrainer


class TrainingPrecisionTest(unittest.TestCase):
    def test_overflow_retries_same_batches_then_steps_optimizer_and_lr_once(self):
        trainer = object.__new__(MarigoldTrainer)
        weight = torch.nn.Parameter(torch.tensor(1.0))
        trainer.optimizer = torch.optim.SGD([weight], lr=0.1)
        trainer.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(trainer.optimizer, lambda _: 1.)
        trainer.scaler = torch.amp.GradScaler("cpu", init_scale=128)
        trainer.train_metrics = MagicMock()
        trainer.scaler.scale(weight * torch.tensor(float("inf"))).backward()
        replayed = []

        def backward(batch, seed):
            self.assertEqual(weight.item(), 1.)  # overflow did not corrupt weights
            replayed.append((batch, seed))
            trainer.scaler.scale(weight.square() / 2).backward()

        trainer._backward_batch = backward
        pending = [("first", 123), ("second", 456)]
        last_epoch = trainer.lr_scheduler.last_epoch
        trainer._step_accumulated_gradients(pending)
        self.assertEqual(replayed, pending)
        self.assertEqual(trainer.lr_scheduler.last_epoch, last_epoch + 1)
        self.assertAlmostEqual(weight.item(), .8, places=6)
        self.assertEqual(trainer.scaler.get_scale(), 64)
        self.assertIsNone(weight.grad)
        trainer.train_metrics.reset.assert_called_once()


if __name__ == "__main__":
    unittest.main()
