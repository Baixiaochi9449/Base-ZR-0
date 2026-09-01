import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.training_numerics import assert_all_finite


class FakeAccelerator:
    def __init__(self, finite_count: int):
        self.device = torch.device("cpu")
        self.num_processes = 4
        self.finite_count = finite_count
        self.reduce_calls = []

    def reduce(self, tensor: torch.Tensor, reduction: str) -> torch.Tensor:
        self.reduce_calls.append((int(tensor.item()), reduction))
        return torch.tensor(self.finite_count, device=tensor.device)


class TrainingNumericsTest(unittest.TestCase):
    def test_all_finite_values_pass_on_every_rank(self):
        accelerator = FakeAccelerator(finite_count=4)

        assert_all_finite(
            accelerator, torch.tensor(1.25), "loss", step=7
        )

        self.assertEqual(accelerator.reduce_calls, [(1, "sum")])

    def test_local_nan_raises_with_step_and_metric(self):
        accelerator = FakeAccelerator(finite_count=0)

        with self.assertRaisesRegex(FloatingPointError, r"loss.*step 7"):
            assert_all_finite(
                accelerator, torch.tensor(float("nan")), "loss", step=7
            )

    def test_non_finite_value_on_another_rank_raises_everywhere(self):
        accelerator = FakeAccelerator(finite_count=3)

        with self.assertRaisesRegex(
            FloatingPointError, r"global gradient norm.*step 11"
        ):
            assert_all_finite(
                accelerator, 2.0, "global gradient norm", step=11
            )


if __name__ == "__main__":
    unittest.main()
