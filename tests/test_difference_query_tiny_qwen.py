import sys
import unittest
from pathlib import Path

import torch
from torch import nn
from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration
from transformers.feature_extraction_utils import BatchFeature


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.difference_query import DifferenceQuery
from model.qwen_vl_backbone import QwenVLBackbone


class TinyQwenDifferenceQueryTest(unittest.TestCase):
    @staticmethod
    def make_backbone(num_queries: int = 2) -> QwenVLBackbone:
        config = Qwen3VLConfig(
            text_config={
                "vocab_size": 128,
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 8,
                "max_position_embeddings": 128,
                "attention_dropout": 0.0,
                "rope_scaling": {
                    "mrope_interleaved": True,
                    "mrope_section": [2, 1, 1],
                    "rope_type": "default",
                },
            },
            vision_config={
                "depth": 1,
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_heads": 4,
                "out_hidden_size": 32,
                "deepstack_visual_indexes": [],
            },
            image_token_id=120,
            video_token_id=121,
            vision_start_token_id=122,
            vision_end_token_id=123,
        )
        config._attn_implementation = "sdpa"
        torch.manual_seed(7)
        model = Qwen3VLForConditionalGeneration(config)

        backbone = QwenVLBackbone.__new__(QwenVLBackbone)
        nn.Module.__init__(backbone)
        backbone.tune_vlm = False
        backbone.model = model
        backbone.use_difference_query = True
        backbone.num_difference_queries = num_queries
        backbone.query_placeholder_token_id = 0
        backbone.difference_query = DifferenceQuery(
            num_queries, hidden_size=32, initializer_std=0.02
        )
        model.requires_grad_(False)
        backbone.eval()
        return backbone

    @staticmethod
    def inputs(context_first_token: int, target_tokens: tuple[int, int]) -> BatchFeature:
        first_target, second_target = target_tokens
        return BatchFeature(
            {
                "input_ids": torch.tensor(
                    [[context_first_token, 6, first_target, second_target, 127]]
                ),
                "attention_mask": torch.tensor([[1, 1, 1, 1, 0]]),
                "labels": torch.tensor(
                    [[-100, -100, first_target, second_target, -100]]
                ),
            }
        )

    def test_target_is_isolated_context_changes_queries_and_frozen_qwen_backpropagates(self):
        backbone = self.make_backbone()

        attention_outputs = []
        hook = backbone.model.model.language_model.layers[0].self_attn.register_forward_hook(
            lambda _module, _args, output: attention_outputs.append(output[0].detach())
        )
        original = backbone(self.inputs(5, (7, 8))).backbone_embeddings
        hook.remove()
        padding_position = 6  # original L=5, Nq=2, and the final position is P.
        self.assertTrue(torch.isfinite(attention_outputs[0][:, padding_position]).all())
        changed_target = backbone(self.inputs(5, (9, 10))).backbone_embeddings
        changed_context = backbone(self.inputs(11, (7, 8))).backbone_embeddings

        torch.testing.assert_close(original, changed_target, rtol=0.0, atol=0.0)
        self.assertFalse(torch.allclose(original, changed_context))
        self.assertTrue(torch.isfinite(original).all())

        loss = original.square().mean()
        loss.backward()
        gradient = backbone.difference_query.weight.grad
        self.assertIsNotNone(gradient)
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(gradient.abs().sum().item(), 0.0)
        self.assertTrue(all(parameter.grad is None for parameter in backbone.model.parameters()))

    def test_real_qwen_generate_is_blocked_before_generation(self):
        backbone = self.make_backbone()
        backbone._install_generation_guard()

        with self.assertRaisesRegex(NotImplementedError, "Difference Query"):
            backbone.model.generate(input_ids=torch.tensor([[5, 6]]))

    def test_action_only_bypasses_lm_head_and_matches_vlm_hidden_states(self):
        backbone = self.make_backbone()
        inputs = self.inputs(5, (7, 8))

        with torch.no_grad():
            with_vlm_loss = backbone(inputs, compute_vlm_loss=True)
            action_only = backbone(inputs, compute_vlm_loss=False)
        torch.testing.assert_close(
            action_only.backbone_embeddings,
            with_vlm_loss.backbone_embeddings,
            rtol=1e-5,
            atol=1e-6,
        )
        self.assertIsNone(action_only.vlm_loss)
        self.assertIsNotNone(with_vlm_loss.vlm_loss)

        class FailingLMHead(nn.Module):
            def forward(self, _hidden_states):
                raise AssertionError("LM head must not run for action-only")

        backbone.model.lm_head = FailingLMHead()
        with torch.no_grad():
            outputs = backbone(inputs, compute_vlm_loss=False)
        self.assertTrue(torch.isfinite(outputs.backbone_embeddings).all())
        with self.assertRaisesRegex(AssertionError, "LM head"):
            backbone(inputs, compute_vlm_loss=True)

    def test_disabled_action_only_preserves_conditional_lm_head_path(self):
        backbone = self.make_backbone()
        backbone.use_difference_query = False
        backbone.num_difference_queries = None
        backbone.difference_query = None
        inputs = self.inputs(5, (7, 8))
        inputs["sub_task_flag"] = torch.tensor([0])

        with torch.no_grad():
            with_vlm_loss = backbone(inputs, compute_vlm_loss=True)
            action_only = backbone(inputs, compute_vlm_loss=False)
        torch.testing.assert_close(
            action_only.backbone_embeddings,
            with_vlm_loss.backbone_embeddings,
            rtol=1e-5,
            atol=1e-6,
        )

        original_lm_head = backbone.model.lm_head

        class CountingLMHead(nn.Module):
            def __init__(self, wrapped):
                super().__init__()
                self.wrapped = wrapped
                self.calls = 0

            def forward(self, hidden_states):
                self.calls += 1
                return self.wrapped(hidden_states)

        counting_lm_head = CountingLMHead(original_lm_head)
        backbone.model.lm_head = counting_lm_head
        with torch.no_grad():
            outputs = backbone(inputs, compute_vlm_loss=False)
        self.assertEqual(counting_lm_head.calls, 1)
        self.assertIsNotNone(outputs.vlm_loss)

    def test_real_qwen_forward_supports_nq_8_32_64(self):
        inputs = self.inputs(5, (7, 8))
        inputs.pop("labels")

        for num_queries in (8, 32, 64):
            with self.subTest(num_queries=num_queries):
                backbone = self.make_backbone(num_queries=num_queries)
                with torch.no_grad():
                    outputs = backbone(inputs, compute_vlm_loss=False)
                self.assertEqual(
                    outputs.backbone_embeddings.shape,
                    (1, num_queries, 32),
                )
                self.assertEqual(
                    outputs.action_expert_cross_attn_mask.shape,
                    (1, num_queries),
                )
                self.assertTrue(torch.isfinite(outputs.backbone_embeddings).all())


if __name__ == "__main__":
    unittest.main()
