import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.difference_query import (
    DifferenceQuery,
    build_difference_query_sequence,
)


class DifferenceQueryParameterTest(unittest.TestCase):
    def test_parameter_shapes_and_counts(self):
        for num_queries, expected_count in ((8, 16_384), (32, 65_536), (64, 131_072)):
            with self.subTest(num_queries=num_queries):
                query = DifferenceQuery(num_queries, hidden_size=2048, initializer_std=0.02)
                self.assertEqual(query.weight.shape, (num_queries, 2048))
                self.assertEqual(query.weight.dtype, torch.float32)
                self.assertEqual(sum(p.numel() for p in query.parameters()), expected_count)

    def test_initialization_does_not_advance_global_rng(self):
        torch.manual_seed(1234)
        expected_action_layer = nn.Linear(5, 7)

        torch.manual_seed(1234)
        before = torch.random.get_rng_state().clone()
        DifferenceQuery(32, hidden_size=16, initializer_std=0.02)
        after = torch.random.get_rng_state().clone()
        actual_action_layer = nn.Linear(5, 7)

        self.assertTrue(torch.equal(before, after))
        torch.testing.assert_close(actual_action_layer.weight, expected_action_layer.weight)
        torch.testing.assert_close(actual_action_layer.bias, expected_action_layer.bias)

    def test_batch_values_follow_reference_dtype_and_device(self):
        query = DifferenceQuery(2, hidden_size=3, initializer_std=0.02)
        reference = torch.zeros(4, 7, 3, dtype=torch.float64)

        values = query.for_batch(batch_size=4, reference=reference)

        self.assertEqual(values.shape, (4, 2, 3))
        self.assertEqual(values.dtype, reference.dtype)
        self.assertEqual(values.device, reference.device)


class DifferenceQuerySequenceTest(unittest.TestCase):
    def setUp(self):
        self.input_ids = torch.tensor(
            [
                [10, 11, 31, 32, 99, 99],
                [20, 21, 22, 99, 99, 99],
            ]
        )
        self.attention_mask = torch.tensor(
            [
                [1, 1, 1, 1, 0, 0],
                [1, 1, 1, 0, 0, 0],
            ]
        )
        self.labels = torch.tensor(
            [
                [-100, -100, 31, 32, -100, -100],
                [-100, -100, -100, -100, -100, -100],
            ]
        )

    def test_training_reorders_each_sample_to_context_query_target_padding(self):
        result = build_difference_query_sequence(
            self.input_ids,
            self.attention_mask,
            labels=self.labels,
            num_queries=2,
            placeholder_token_id=0,
        )

        torch.testing.assert_close(
            result.auxiliary_input_ids,
            torch.tensor(
                [
                    [10, 11, 0, 0, 31, 32, 99, 99],
                    [20, 21, 22, 0, 0, 99, 99, 99],
                ]
            ),
        )
        torch.testing.assert_close(
            result.labels,
            torch.tensor(
                [
                    [-100, -100, -100, -100, 31, 32, -100, -100],
                    [-100, -100, -100, -100, -100, -100, -100, -100],
                ]
            ),
        )
        torch.testing.assert_close(
            result.valid_attention_mask,
            torch.tensor(
                [
                    [True, True, True, True, True, True, False, False],
                    [True, True, True, True, True, False, False, False],
                ]
            ),
        )
        torch.testing.assert_close(
            result.query_positions,
            torch.tensor([[2, 3], [3, 4]]),
        )
        torch.testing.assert_close(result.context_lengths, torch.tensor([2, 3]))
        torch.testing.assert_close(result.target_lengths, torch.tensor([2, 0]))

    def test_direct_inference_reorders_to_context_query_padding(self):
        result = build_difference_query_sequence(
            self.input_ids,
            self.attention_mask,
            labels=None,
            num_queries=2,
            placeholder_token_id=0,
        )

        torch.testing.assert_close(
            result.auxiliary_input_ids,
            torch.tensor(
                [
                    [10, 11, 31, 32, 0, 0, 99, 99],
                    [20, 21, 22, 0, 0, 99, 99, 99],
                ]
            ),
        )
        self.assertIsNone(result.labels)
        torch.testing.assert_close(result.query_positions, torch.tensor([[4, 5], [3, 4]]))

    def test_attention_mask_matches_every_context_query_target_padding_block(self):
        result = build_difference_query_sequence(
            self.input_ids[:1],
            self.attention_mask[:1],
            labels=self.labels[:1],
            num_queries=2,
            placeholder_token_id=0,
        )
        mask = result.attention_mask[0, 0]

        expected = torch.tensor(
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 0, 0, 0, 0],
                [1, 1, 1, 1, 0, 0, 0, 0],
                [0, 0, 1, 1, 1, 0, 0, 0],
                [0, 0, 1, 1, 1, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
            ],
            dtype=torch.bool,
        )
        torch.testing.assert_close(mask, expected)
        self.assertEqual(result.attention_mask.shape, (1, 1, 8, 8))
        self.assertEqual(result.attention_mask.dtype, torch.bool)
        self.assertEqual(result.attention_mask.device, self.input_ids.device)
        self.assertFalse(mask[:, 6:].any())
        self.assertFalse(mask[6:, :].any())

    def test_padding_must_be_right_aligned_and_labels_cannot_supervise_padding(self):
        bad_padding = torch.tensor([[1, 0, 1, 0, 0, 0]])
        with self.assertRaisesRegex(ValueError, "right padding"):
            build_difference_query_sequence(
                self.input_ids[:1],
                bad_padding,
                labels=self.labels[:1],
                num_queries=2,
                placeholder_token_id=0,
            )

        bad_labels = self.labels[:1].clone()
        bad_labels[0, 5] = 42
        with self.assertRaisesRegex(ValueError, "padding"):
            build_difference_query_sequence(
                self.input_ids[:1],
                self.attention_mask[:1],
                labels=bad_labels,
                num_queries=2,
                placeholder_token_id=0,
            )

    def test_sequence_build_does_not_extract_per_sample_python_scalars(self):
        with patch.object(
            torch.Tensor,
            "item",
            side_effect=AssertionError("per-sample Tensor.item() is forbidden"),
        ):
            result = build_difference_query_sequence(
                self.input_ids,
                self.attention_mask,
                labels=self.labels,
                num_queries=32,
                placeholder_token_id=0,
            )

        self.assertEqual(result.attention_mask.shape, (2, 1, 38, 38))

    def test_max_length_cpu_structure_and_parameterized_query_counts(self):
        batch_size = 4
        original_length = 1200
        valid_lengths = torch.tensor([1200, 911, 513, 1])
        positions = torch.arange(original_length).unsqueeze(0)
        attention_mask = positions < valid_lengths.unsqueeze(1)
        input_ids = positions.expand(batch_size, -1).clone() + 10
        input_ids.masked_fill_(~attention_mask, 99)
        labels = torch.full_like(input_ids, -100)
        labels[0, 1100:1200] = input_ids[0, 1100:1200]
        labels[1, 900:911] = input_ids[1, 900:911]

        for num_queries in (8, 32, 64):
            with self.subTest(num_queries=num_queries):
                result = build_difference_query_sequence(
                    input_ids,
                    attention_mask,
                    labels=labels,
                    num_queries=num_queries,
                    placeholder_token_id=0,
                )
                extended_length = original_length + num_queries
                self.assertEqual(
                    result.attention_mask.shape,
                    (batch_size, 1, extended_length, extended_length),
                )
                self.assertEqual(
                    result.query_positions.shape, (batch_size, num_queries)
                )
                torch.testing.assert_close(
                    result.valid_attention_mask.sum(dim=1),
                    valid_lengths + num_queries,
                )
                for batch_index, valid_length in enumerate(valid_lengths.tolist()):
                    padding_start = valid_length + num_queries
                    mask = result.attention_mask[batch_index, 0]
                    self.assertFalse(mask[padding_start:, :].any())
                    self.assertFalse(mask[:, padding_start:].any())

if __name__ == "__main__":
    unittest.main()
