# Last modified: 2024-04-18
#
# Copyright 2023 Bingxin Ke, ETH Zurich. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# --------------------------------------------------------------------------
# If you find this code useful, we kindly ask you to cite our paper in your work.
# Please find bibtex at: https://github.com/prs-eth/Marigold#-citation
# If you use or adapt this code, please attribute to https://github.com/prs-eth/marigold.
# More information about the method can be found at https://marigoldmonodepth.github.io
# --------------------------------------------------------------------------

import torch
from torch.utils.data import (
    BatchSampler,
    RandomSampler,
    SequentialSampler,
)


class MixedBatchSampler(BatchSampler):
    """Sample one batch from a selected dataset with given probability.
    Compatible with datasets at different resolution
    """

    def __init__(
        self, src_dataset_ls, batch_size, drop_last, shuffle, prob=None, generator=None,
        probability_schedule=None, total_steps=None, accumulation_steps=1,
    ):
        self.base_sampler = None
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.generator = generator

        self.src_dataset_ls = src_dataset_ls
        self.n_dataset = len(self.src_dataset_ls)

        # Dataset length
        self.dataset_length = [len(ds) for ds in self.src_dataset_ls]
        self.cum_dataset_length = [
            sum(self.dataset_length[:i]) for i in range(self.n_dataset)
        ]  # cumulative dataset length

        # BatchSamplers for each source dataset
        if self.shuffle:
            self.src_batch_samplers = [
                BatchSampler(
                    sampler=RandomSampler(
                        ds, replacement=False, generator=self.generator
                    ),
                    batch_size=self.batch_size,
                    drop_last=self.drop_last,
                )
                for ds in self.src_dataset_ls
            ]
        else:
            self.src_batch_samplers = [
                BatchSampler(
                    sampler=SequentialSampler(ds),
                    batch_size=self.batch_size,
                    drop_last=self.drop_last,
                )
                for ds in self.src_dataset_ls
            ]
        self.raw_batches = [
            list(bs) for bs in self.src_batch_samplers
        ]  # index in original dataset
        self.n_batches = [len(b) for b in self.raw_batches]
        self.n_total_batch = sum(self.n_batches)
        if any(n == 0 for n in self.n_batches):
            raise ValueError("Each source dataset must provide at least one batch")

        # sampling probability
        if prob is None:
            # if not given, decide by dataset length
            self.prob = torch.tensor(self.n_batches) / self.n_total_batch
        else:
            self.prob = torch.as_tensor(prob)

        self.initial_prob = self._normalize_probabilities(self.prob)
        self.accumulation_steps = int(accumulation_steps)
        if self.accumulation_steps < 1 or self.accumulation_steps != accumulation_steps:
            raise ValueError("accumulation_steps must be a positive integer")
        self.epoch_start_batch = 0
        schedule = probability_schedule or {}
        self.schedule_enabled = bool(schedule.get("enabled", False))
        if self.schedule_enabled:
            if total_steps is None or total_steps <= 0:
                raise ValueError("Dynamic sampling requires a positive max_iter")
            self.final_prob = self._normalize_probabilities(schedule["final_prob_ls"])
            start = float(schedule.get("start_fraction", 0.2))
            end = float(schedule.get("end_fraction", 0.8))
            if not 0 <= start < end <= 1:
                raise ValueError("Require 0 <= start_fraction < end_fraction <= 1")
            self.start_step = start * total_steps
            self.end_step = end * total_steps

    def _normalize_probabilities(self, values):
        values = torch.as_tensor(list(values), dtype=torch.float64)
        if (values.ndim != 1 or len(values) != self.n_dataset
                or not torch.isfinite(values).all() or (values < 0).any()
                or values.sum() <= 0):
            raise ValueError("Dataset probabilities must be finite, nonnegative, "
                             "match the dataset count, and have a positive sum")
        return values / values.sum()

    def probabilities_at(self, completed_steps):
        """Probabilities for the update following completed_steps optimizer steps."""
        if not self.schedule_enabled:
            return self.initial_prob.clone()
        fraction = min(1.0, max(0.0, (completed_steps - self.start_step)
                               / (self.end_step - self.start_step)))
        return self.initial_prob.lerp(self.final_prob, fraction)

    def set_epoch_start(self, completed_steps, accumulated_batches=0, skipped_batches=0):
        # Assign probabilities by a batch's logical training position, not by
        # when DataLoader happens to prefetch it. On resume, SkipBatchSampler
        # replays skipped batches, so their positions must be subtracted here.
        start = (completed_steps * self.accumulation_steps
                 + accumulated_batches - skipped_batches)
        if start < 0:
            raise ValueError("Epoch start precedes the beginning of training")
        self.epoch_start_batch = start

    def __iter__(self):
        """_summary_

        Yields:
            list(int): a batch of indics, corresponding to ConcatDataset of src_dataset_ls
        """
        epoch_start_batch = self.epoch_start_batch
        for batch_index in range(self.n_total_batch):
            completed_steps = (epoch_start_batch + batch_index) // self.accumulation_steps
            # Preserve the original fixed sampler when scheduling is disabled.
            prob = self.probabilities_at(completed_steps) if self.schedule_enabled else self.prob
            idx_ds = torch.multinomial(
                prob, 1, replacement=True, generator=self.generator
            ).item()
            # if batch list is empty, generate new list
            if 0 == len(self.raw_batches[idx_ds]):
                self.raw_batches[idx_ds] = list(self.src_batch_samplers[idx_ds])
            # get a batch from list
            batch_raw = self.raw_batches[idx_ds].pop()
            # shift by cumulative dataset length
            shift = self.cum_dataset_length[idx_ds]
            batch = [n + shift for n in batch_raw]

            yield batch

    def __len__(self):
        return self.n_total_batch


# Unit test
if "__main__" == __name__:
    from torch.utils.data import ConcatDataset, DataLoader, Dataset

    class SimpleDataset(Dataset):
        def __init__(self, start, len) -> None:
            super().__init__()
            self.start = start
            self.len = len

        def __len__(self):
            return self.len

        def __getitem__(self, index):
            return self.start + index

    dataset_1 = SimpleDataset(0, 10)
    dataset_2 = SimpleDataset(200, 20)
    dataset_3 = SimpleDataset(1000, 50)

    concat_dataset = ConcatDataset(
        [dataset_1, dataset_2, dataset_3]
    )  # will directly concatenate

    mixed_sampler = MixedBatchSampler(
        src_dataset_ls=[dataset_1, dataset_2, dataset_3],
        batch_size=4,
        drop_last=True,
        shuffle=False,
        prob=[0.6, 0.3, 0.1],
        generator=torch.Generator().manual_seed(0),
    )

    loader = DataLoader(concat_dataset, batch_sampler=mixed_sampler)

    for d in loader:
        print(d)
