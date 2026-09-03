import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch.utils.data import ConcatDataset, Dataset

import train_vla
from utils.load_training_dataset import (
    EpochGroupedDistributedBatchSampler,
    EpochGroupedSampler,
)


class _IndexedDataset(Dataset):
    def __init__(self, size):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return index



class _GroupedIndexedDataset(_IndexedDataset):
    def __init__(self, size, groups):
        super().__init__(size)
        self.groups = groups

    def sampling_groups(self):
        return self.groups


class TrainResumeTest(unittest.TestCase):
    def test_epoch_sampler_is_reproducible_and_keeps_episode_groups_contiguous(self):
        grouped = _GroupedIndexedDataset(5, groups=[[0, 1, 2], [3, 4]])
        ungrouped = _IndexedDataset(3)
        concat = ConcatDataset([grouped, ungrouped])
        sampler = EpochGroupedSampler(concat, seed=23)

        sampler.set_epoch(4)
        first = list(sampler)
        sampler.set_epoch(4)
        resumed = list(sampler)
        sampler.set_epoch(5)
        next_epoch = list(sampler)

        self.assertEqual(first, resumed)
        self.assertNotEqual(first, next_epoch)
        self.assertEqual(sorted(first), list(range(8)))
        for episode in ([0, 1, 2], [3, 4]):
            positions = [first.index(index) for index in episode]
            self.assertEqual(positions, list(range(min(positions), max(positions) + 1)))

    def test_distributed_batch_sampler_preserves_natural_tail_for_all_batch_pairs(self):
        concat = ConcatDataset([_IndexedDataset(310743)])

        for micro_batch, gas in ((1, 32), (2, 16), (4, 8), (8, 4), (16, 2), (32, 1)):
            sampler = EpochGroupedDistributedBatchSampler(
                concat,
                batch_size_per_device=micro_batch,
                num_processes=4,
                seed=42,
            )
            batches_by_rank = [[], [], [], []]
            for batch_index, batch in enumerate(sampler):
                batches_by_rank[batch_index % 4].append(batch)

            flattened = [
                index
                for rank_batches in batches_by_rank
                for batch in rank_batches
                for index in batch
            ]
            self.assertEqual(len(flattened), 310744)
            self.assertEqual(len(set(flattened)), 310743)
            self.assertEqual([len(indices) for indices in batches_by_rank], [
                math.ceil(77686 / micro_batch)
            ] * 4)
            final_microbatches = len(batches_by_rank[0]) % gas or gas
            self.assertEqual(
                sum(
                    len(batch)
                    for rank_batches in batches_by_rank
                    for batch in rank_batches[-final_microbatches:]
                ),
                88,
            )

    def test_integrity_check_accepts_action_only_batch_without_labels(self):
        processor = SimpleNamespace(
            tokenizer=SimpleNamespace(decode=lambda token: f"token-{token}")
        )
        batch = {
            "input_ids": torch.tensor([[1, 2]]),
            "attention_mask": torch.tensor([[1, 1]]),
        }
        train_vla.integrity_check(batch, processor)

    def test_resume_data_position_uses_completed_optimizer_steps(self):
        self.assertEqual(
            train_vla.resume_data_position(
                global_completed_steps=3,
                dataloader_length=5,
                gradient_accumulation_steps=2,
            ),
            (1, 0),
        )
        self.assertEqual(
            train_vla.resume_data_position(
                global_completed_steps=2,
                dataloader_length=5,
                gradient_accumulation_steps=2,
            ),
            (0, 4),
        )

    def test_resume_batch_prefix_is_skipped_only_in_resume_epoch(self):
        visited = []
        global_completed_steps = 6
        for epoch in range(1, 3):
            for batch_idx in range(4):
                if train_vla.should_skip_resumed_batch(
                    epoch=epoch,
                    batch_idx=batch_idx,
                    resume_epoch=1,
                    resume_batch_idx=2,
                ):
                    continue
                visited.append((epoch, batch_idx))
                global_completed_steps += 1

        self.assertEqual(
            visited,
            [(1, 2), (1, 3), (2, 0), (2, 1), (2, 2), (2, 3)],
        )
        self.assertEqual(global_completed_steps, 12)

    def test_single_process_step_sync_does_not_require_initialized_distributed(self):
        accelerator = SimpleNamespace(num_processes=1, device=torch.device("cpu"))

        with patch.object(
            train_vla.dist,
            "broadcast",
            side_effect=AssertionError("broadcast must not be called"),
        ):
            synchronized = train_vla.synchronize_global_step(7, accelerator)

        self.assertEqual(synchronized, 7)

    def test_multi_process_step_sync_requires_initialized_distributed(self):
        accelerator = SimpleNamespace(num_processes=2, device=torch.device("cpu"))

        with patch.object(train_vla.dist, "is_available", return_value=True), \
             patch.object(train_vla.dist, "is_initialized", return_value=False), \
             patch.object(
                 train_vla.dist,
                 "broadcast",
                 side_effect=AssertionError("broadcast must not run"),
             ):
            with self.assertRaisesRegex(RuntimeError, "process group"):
                train_vla.synchronize_global_step(7, accelerator)

    def test_multi_process_step_sync_broadcasts_when_initialized(self):
        accelerator = SimpleNamespace(num_processes=2, device=torch.device("cpu"))

        def set_rank_zero_value(step_tensor, src):
            self.assertEqual(src, 0)
            step_tensor.fill_(11)

        with patch.object(train_vla.dist, "is_available", return_value=True), \
             patch.object(train_vla.dist, "is_initialized", return_value=True), \
             patch.object(
                 train_vla.dist, "broadcast", side_effect=set_rank_zero_value
             ) as broadcast:
            synchronized = train_vla.synchronize_global_step(7, accelerator)

        self.assertEqual(synchronized, 11)
        broadcast.assert_called_once()


if __name__ == "__main__":
    unittest.main()
