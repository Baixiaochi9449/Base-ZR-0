import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

import train_vla


class TrainResumeTest(unittest.TestCase):
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
