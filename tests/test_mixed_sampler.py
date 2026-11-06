import unittest
from unittest.mock import patch

import torch
from torch.utils.data import ConcatDataset, DataLoader, TensorDataset

from src.dataset.mixed_sampler import MixedBatchSampler
from src.util.config_util import recursive_load_config
from src.util.data_loader import skip_first_batches


class MixedSamplerTest(unittest.TestCase):
    def make_sampler(self, lengths=(18, 14), prob=(.9, .1)):
        datasets = [TensorDataset(torch.full((n,), i)) for i, n in enumerate(lengths)]
        sampler = MixedBatchSampler(
            datasets, batch_size=2, drop_last=True, shuffle=True,
            prob=prob, generator=torch.Generator().manual_seed(17),
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

    def test_default_config_is_fixed_nine_to_one(self):
        cfg = recursive_load_config("config/train_marigold.yaml")
        self.assertEqual(list(cfg.dataset.train.prob_ls), [.9, .1])
        self.assertNotIn("probability_schedule", cfg.dataset.train)
        sampler, dataset = self.make_sampler(prob=cfg.dataset.train.prob_ls)
        for _ in range(3):
            probabilities, _ = self.collect(sampler, dataset)
            for probability in probabilities:
                torch.testing.assert_close(probability, torch.tensor([.9, .1], dtype=probability.dtype))

    def test_prefetch_preserves_fixed_sampling(self):
        baseline, dataset = self.make_sampler()
        expected, expected_batches = self.collect(baseline, dataset)
        prefetched, dataset = self.make_sampler()
        actual, batches = self.collect(prefetched, dataset, workers=2)
        torch.testing.assert_close(torch.stack(actual), torch.stack(expected))
        self.assertEqual(batches, expected_batches)
        for probability in actual:
            torch.testing.assert_close(probability, prefetched.prob)

    def test_resume_replays_skipped_batches_with_fixed_probabilities(self):
        full, dataset = self.make_sampler()
        expected, expected_batches = self.collect(full, dataset)
        resumed, dataset = self.make_sampler()
        actual, batches = self.collect(resumed, dataset, workers=2, skip=6)
        torch.testing.assert_close(torch.stack(actual[6:]), torch.stack(expected[6:]))
        self.assertEqual(batches, expected_batches[6:])

    def test_sampling_stays_fixed_across_epoch_boundaries(self):
        sampler, dataset = self.make_sampler(lengths=(8, 6))
        first, _ = self.collect(sampler, dataset)
        second, _ = self.collect(sampler, dataset)
        for probability in first + second:
            torch.testing.assert_close(probability, sampler.prob)

    def test_observed_ratio_and_single_dataset_batches(self):
        sampler, dataset = self.make_sampler(lengths=(10000, 10000))
        outdoor = 0
        for indices in sampler:
            sources = [dataset[index][0].item() for index in indices]
            self.assertEqual(len(set(sources)), 1)
            outdoor += sources[0]
        self.assertAlmostEqual(outdoor / len(sampler), .1, delta=.02)

    def test_invalid_configs_fail_early(self):
        for probabilities in ([1], [-1, 2], [0, 0], [float("nan"), 1]):
            with self.subTest(probabilities=probabilities), self.assertRaises(ValueError):
                self.make_sampler(prob=probabilities)


if __name__ == "__main__":
    unittest.main()
