import unittest
from unittest.mock import patch

import torch
from torch.utils.data import ConcatDataset, DataLoader, TensorDataset

from src.dataset.mixed_sampler import MixedBatchSampler
from src.util.config_util import recursive_load_config
from src.util.data_loader import skip_first_batches


class MixedSamplerTest(unittest.TestCase):
    def make_sampler(self, lengths=(18, 14), accumulation=2, total_steps=8, schedule=None):
        datasets = [TensorDataset(torch.full((n,), i)) for i, n in enumerate(lengths)]
        sampler = MixedBatchSampler(
            datasets, batch_size=2, drop_last=True, shuffle=True,
            prob=[0.9, 0.1], generator=torch.Generator().manual_seed(17),
            probability_schedule=schedule if schedule is not None else {
                "enabled": True, "start_fraction": .2, "end_fraction": .8,
                "final_prob_ls": [.8, .2],
            },
            total_steps=total_steps, accumulation_steps=accumulation,
        )
        return sampler, ConcatDataset(datasets)

    def collect(self, sampler, dataset, workers=0, skip=0):
        probabilities = []
        original = torch.multinomial

        def record(prob, *args, **kwargs):
            probabilities.append(prob.clone())
            return original(prob, *args, **kwargs)

        loader = DataLoader(dataset, batch_sampler=sampler, num_workers=workers)
        with patch("torch.multinomial", side_effect=record):
            batches = [batch[0].tolist() for batch in skip_first_batches(loader, skip)]
        return probabilities, batches

    def test_default_config_boundaries_and_fixed_ablation(self):
        cfg = recursive_load_config("config/train_marigold.yaml")
        sampler, _ = self.make_sampler(total_steps=cfg.max_iter,
                                      schedule=cfg.dataset.train.probability_schedule)
        for step, expected in [(0, .1), (4600, .1), (11500, .15),
                               (18400, .2), (23000, .2), (30000, .2)]:
            self.assertAlmostEqual(sampler.probabilities_at(step)[1].item(), expected)
        fixed, _ = self.make_sampler(schedule={"enabled": False})
        self.assertAlmostEqual(fixed.probabilities_at(23000)[1].item(), .1)
        # A legacy config without a schedule preserves fixed sampling.
        legacy = MixedBatchSampler([range(10), range(10)], 2, True, False, prob=[.9, .1])
        self.assertFalse(legacy.schedule_enabled)

    def test_prefetch_does_not_advance_schedule(self):
        baseline, dataset = self.make_sampler()
        expected, expected_batches = self.collect(baseline, dataset)
        prefetched, dataset = self.make_sampler()
        actual, batches = self.collect(prefetched, dataset, workers=2)
        torch.testing.assert_close(torch.stack(actual), torch.stack(expected))
        self.assertEqual(batches, expected_batches)
        for i, probability in enumerate(actual):
            torch.testing.assert_close(probability, prefetched.probabilities_at(i // 2))

    def test_resume_replays_skipped_positions_without_resetting_progress(self):
        full, dataset = self.make_sampler()
        expected, expected_batches = self.collect(full, dataset)
        resumed, dataset = self.make_sampler()
        resumed.set_epoch_start(completed_steps=3, skipped_batches=6)
        actual, batches = self.collect(resumed, dataset, workers=2, skip=6)
        torch.testing.assert_close(torch.stack(actual[6:]), torch.stack(expected[6:]))
        self.assertEqual(batches, expected_batches[6:])

    def test_accumulation_crosses_epoch_boundary(self):
        sampler, dataset = self.make_sampler(lengths=(8, 6), accumulation=3)
        first, _ = self.collect(sampler, dataset)
        sampler.set_epoch_start(completed_steps=2, accumulated_batches=1)
        second, _ = self.collect(sampler, dataset)
        for i, probability in enumerate(first + second):
            torch.testing.assert_close(probability, sampler.probabilities_at(i // 3))
        # Resume in a later epoch, whose first batch has global position seven.
        sampler.set_epoch_start(completed_steps=3, skipped_batches=2)
        self.assertEqual(sampler.epoch_start_batch, 7)

    def test_invalid_configs_fail_early(self):
        for final in ([1], [-1, 2], [0, 0], [float("nan"), 1]):
            with self.subTest(final=final), self.assertRaises(ValueError):
                self.make_sampler(schedule={"enabled": True, "final_prob_ls": final})
        with self.assertRaises(ValueError):
            self.make_sampler(total_steps=0)
        with self.assertRaises(ValueError):
            self.make_sampler(schedule={"enabled": True, "final_prob_ls": [.8, .2],
                                        "start_fraction": .8, "end_fraction": .2})


if __name__ == "__main__":
    unittest.main()
