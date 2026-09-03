import sys
import unittest
from pathlib import Path
from types import MethodType

import torch
from torch import nn
from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration
from transformers.feature_extraction_utils import BatchFeature


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.difference_query import DifferenceQuery
from model.flow_matching_action_head import (
    FlowmatchingActionHead,
    FlowmatchingActionHeadConfig,
)
from model.qwen_vl_backbone import QwenVLBackbone
from model.reasoning_vla_model import ZR0Model


class TinyVlaDifferenceQueryTest(unittest.TestCase):
    @staticmethod
    def make_backbone(num_queries: int = 4) -> QwenVLBackbone:
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
        model = Qwen3VLForConditionalGeneration(config)
        model.requires_grad_(False)

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
        return backbone

    @staticmethod
    def make_action_head() -> FlowmatchingActionHead:
        config = FlowmatchingActionHeadConfig(
            vlm_output_embedding_dim=32,
            action_or_state_token_embedding_dim=32,
            mlp_hidden_size=16,
            max_seq_len=8,
            action_dim=4,
            state_dim=4,
            action_horizon=3,
            diffusion_transformer_cfg={
                "num_attention_heads": 4,
                "attention_head_dim": 8,
                "output_dim": 16,
                "num_layers": 2,
                "dropout": 0.0,
                "attention_bias": True,
                "activation_fn": "gelu-approximate",
                "upcast_attention": False,
                "norm_type": "ada_norm",
                "norm_elementwise_affine": False,
                "norm_eps": 1e-5,
                "max_num_positional_embeddings": 8,
                "positional_embeddings": None,
                "final_dropout": False,
                "interleave_self_attention": True,
                "causal_mask_in_self_attn": False,
            },
        )
        return FlowmatchingActionHead(config, tune_action_expert=True)

    def test_action_only_forward_backward_and_direct_denoising_are_finite(self):
        torch.manual_seed(19)
        backbone = self.make_backbone()
        action_head = self.make_action_head()
        vlm_inputs = BatchFeature(
            {
                "input_ids": torch.tensor([[5, 6, 7, 8, 127]]),
                "attention_mask": torch.tensor([[1, 1, 1, 1, 0]]),
                "labels": torch.tensor([[-100, -100, 7, 8, -100]]),
            }
        )
        backbone_outputs = backbone(vlm_inputs)
        train_inputs = BatchFeature(
            {
                "observation.state": torch.randn(1, 1, 4),
                "state_mask": torch.ones(1, 4, dtype=torch.bool),
                "action": torch.randn(1, 3, 4),
                "action_mask": torch.ones(1, 3, 4, dtype=torch.bool),
            }
        )

        train_outputs = action_head(
            backbone_outputs, train_inputs, training_progress=0.25
        )
        loss = train_outputs.action_expert_loss
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        query_gradient = backbone.difference_query.weight.grad
        self.assertIsNotNone(query_gradient)
        self.assertTrue(torch.isfinite(query_gradient).all())
        self.assertGreater(query_gradient.abs().sum().item(), 0.0)

        infer_inputs = BatchFeature(
            {
                "observation.state": train_inputs["observation.state"],
                "state_mask": train_inputs["state_mask"],
                "infer_action_mask": torch.ones(1, 3, 4, dtype=torch.bool),
            }
        )
        torch.manual_seed(23)
        action_outputs = action_head.get_action(
            backbone_outputs, infer_inputs, num_denoised_steps=2
        )
        self.assertEqual(action_outputs.action_pred.shape, (1, 3, 4))
        self.assertTrue(torch.isfinite(action_outputs.action_pred).all())

    def test_direct_action_supports_nq_8_32_64(self):
        infer_inputs = BatchFeature(
            {
                "observation.state": torch.randn(1, 1, 4),
                "state_mask": torch.ones(1, 4, dtype=torch.bool),
                "infer_action_mask": torch.ones(1, 3, 4, dtype=torch.bool),
            }
        )
        vlm_inputs = BatchFeature(
            {
                "input_ids": torch.tensor([[5, 6, 7, 8, 127]]),
                "attention_mask": torch.tensor([[1, 1, 1, 1, 0]]),
            }
        )

        for num_queries in (8, 32, 64):
            with self.subTest(num_queries=num_queries):
                backbone = self.make_backbone(num_queries=num_queries)
                action_head = self.make_action_head()
                with torch.no_grad():
                    backbone_outputs = backbone(
                        vlm_inputs, compute_vlm_loss=False
                    )
                    torch.manual_seed(23)
                    action_outputs = action_head.get_action(
                        backbone_outputs, infer_inputs, num_denoised_steps=2
                    )
                self.assertEqual(action_outputs.action_pred.shape, (1, 3, 4))
                self.assertTrue(torch.isfinite(action_outputs.action_pred).all())

    def test_disabled_action_only_and_direct_preserve_head_path_and_outputs(self):
        torch.manual_seed(31)
        backbone = self.make_backbone()
        backbone.use_difference_query = False
        backbone.num_difference_queries = None
        backbone.difference_query = None
        action_head = self.make_action_head()

        def prepare_text_inputs(_backbone, batch):
            values = {
                "input_ids": batch["input_ids"],
                "attention_mask": batch["attention_mask"],
                "sub_task_flag": batch["sub_task_flag"],
            }
            if "labels" in batch:
                values["labels"] = batch["labels"]
            return BatchFeature(values)

        backbone.prepare_inputs = MethodType(prepare_text_inputs, backbone)

        class CountingLMHead(nn.Module):
            def __init__(self, wrapped):
                super().__init__()
                self.wrapped = wrapped
                self.calls = 0
                self.sequence_lengths = []

            def forward(self, hidden_states):
                self.calls += 1
                self.sequence_lengths.append(hidden_states.shape[1])
                return self.wrapped(hidden_states)

        counting_lm_head = CountingLMHead(backbone.model.lm_head)
        backbone.model.lm_head = counting_lm_head

        model = ZR0Model.__new__(ZR0Model)
        nn.Module.__init__(model)
        model.backbone = backbone
        model.action_expert = action_head
        model.action_expert_config = action_head.config
        model.loss_type = "action"
        model.use_difference_query = False
        model.num_difference_queries = None
        model.detach_vlm_outputs_for_action_expert = False

        common = {
            "input_ids": torch.tensor([[5, 6, 7, 8, 127]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 0]]),
            "sub_task_flag": torch.tensor([0]),
            "observation.state": torch.randn(1, 1, 4),
            "state_mask": torch.ones(1, 1, 4, dtype=torch.bool),
        }
        train_batch = BatchFeature(
            {
                **common,
                "labels": torch.tensor([[-100, -100, 7, 8, -100]]),
                "action": torch.randn(1, 3, 4),
                "action_mask": torch.ones(1, 3, 4, dtype=torch.bool),
            }
        )
        model.train()
        torch.manual_seed(37)
        train_outputs = model(
            train_batch,
            training_progress=0.25,
        )
        self.assertTrue(torch.isfinite(train_outputs.action_expert_loss))
        self.assertEqual(counting_lm_head.calls, 1)
        self.assertEqual(counting_lm_head.sequence_lengths, [0])

        direct_batch = BatchFeature(
            {
                **common,
                "infer_action_mask": torch.ones(1, 3, 4, dtype=torch.bool),
            }
        )
        model.eval()
        prepared = backbone.prepare_inputs(direct_batch)
        with torch.no_grad():
            reference_backbone = backbone(prepared, compute_vlm_loss=True)
            compatibility_backbone = backbone(prepared, compute_vlm_loss=False)
        torch.testing.assert_close(
            compatibility_backbone.backbone_embeddings,
            reference_backbone.backbone_embeddings,
            rtol=1e-5,
            atol=1e-6,
        )
        torch.testing.assert_close(
            compatibility_backbone.action_expert_cross_attn_mask,
            reference_backbone.action_expert_cross_attn_mask,
        )
        self.assertEqual(
            compatibility_backbone.action_expert_cross_attn_mask.ndim,
            2,
        )

        action_inputs = action_head.prepare_inputs(direct_batch)
        torch.manual_seed(41)
        with torch.no_grad():
            reference_actions = action_head.get_action(
                reference_backbone,
                action_inputs,
                num_denoised_steps=2,
            ).action_pred
        torch.manual_seed(41)
        actual_actions = model.get_action_direct(
            direct_batch,
            num_denoised_steps=2,
        ).action_pred
        torch.testing.assert_close(actual_actions, reference_actions)
        self.assertEqual(counting_lm_head.calls, 4)
        self.assertEqual(counting_lm_head.sequence_lengths, [0, 5, 0, 0])


if __name__ == "__main__":
    unittest.main()
