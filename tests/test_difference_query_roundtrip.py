import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import AutoProcessor, Qwen3VLConfig, Qwen3VLForConditionalGeneration
from transformers.feature_extraction_utils import BatchFeature


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.difference_query import save_difference_query_artifacts
from model.flow_matching_action_head import (
    FlowmatchingActionHead,
    FlowmatchingActionHeadConfig,
)
from model.reasoning_vla_model import ZR0Model


PROCESSOR_SOURCE = Path("/opt/data/private/lq/models/Qwen3-VL-2B-Instruct")


@unittest.skipUnless(
    (PROCESSOR_SOURCE / "preprocessor_config.json").is_file(),
    "local Qwen3-VL processor is unavailable",
)
class DifferenceQueryRealRoundTripTest(unittest.TestCase):
    @staticmethod
    def qwen_config() -> Qwen3VLConfig:
        return Qwen3VLConfig(
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

    @staticmethod
    def action_config() -> FlowmatchingActionHeadConfig:
        return FlowmatchingActionHeadConfig(
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

    @classmethod
    def make_complete_checkpoint(cls, directory: Path) -> torch.Tensor:
        torch.manual_seed(31)
        qwen = Qwen3VLForConditionalGeneration(cls.qwen_config())
        qwen.save_pretrained(directory)
        AutoProcessor.from_pretrained(PROCESSOR_SOURCE).save_pretrained(directory)

        action_config = cls.action_config()
        action_expert = FlowmatchingActionHead(
            action_config, tune_action_expert=False
        )
        save_file(
            action_expert.state_dict(), directory / "action_expert.safetensors"
        )
        (directory / "action_expert_config.json").write_text(
            json.dumps(action_config.to_dict(), indent=4) + "\n",
            encoding="utf-8",
        )

        expected_query = torch.linspace(-0.25, 0.25, 8 * 32).reshape(8, 32)
        save_difference_query_artifacts(
            directory,
            enabled=True,
            hidden_size=32,
            difference_query=expected_query,
        )
        return expected_query

    @staticmethod
    def batch() -> BatchFeature:
        return BatchFeature(
            {
                "input_ids": torch.tensor([[5, 6, 7, 8, 127]]),
                "attention_mask": torch.tensor([[1, 1, 1, 1, 0]]),
                "pixel_values": None,
                "image_grid_thw": None,
                "sub_task_flag": torch.tensor([0]),
                "observation.state": torch.randn(1, 1, 4),
                "state_mask": torch.ones(1, 4, dtype=torch.bool),
                "infer_action_mask": torch.ones(1, 3, 4, dtype=torch.bool),
            }
        )

    def test_complete_real_qwen_checkpoint_round_trip_without_process_group(self):
        self.assertFalse(torch.distributed.is_initialized())
        with tempfile.TemporaryDirectory() as temp_directory:
            root = Path(temp_directory)
            source = root / "source"
            saved = root / "saved"
            source.mkdir()
            expected_query = self.make_complete_checkpoint(source)

            first = ZR0Model.from_pretrained(source)
            first.float().eval()
            batch = self.batch()
            backbone_inputs = first.backbone.prepare_inputs(batch)
            with torch.no_grad():
                backbone_before = first.backbone(
                    backbone_inputs, compute_vlm_loss=False
                ).backbone_embeddings
                torch.manual_seed(97)
                action_before = first.get_action_direct(
                    batch, num_denoised_steps=2
                ).action_pred
            first.save_pretrained(saved)

            second = ZR0Model.from_pretrained(saved)
            second.float().eval()
            with torch.no_grad():
                backbone_after = second.backbone(
                    second.backbone.prepare_inputs(batch), compute_vlm_loss=False
                ).backbone_embeddings
                torch.manual_seed(97)
                action_after = second.get_action_direct(
                    batch, num_denoised_steps=2
                ).action_pred

            self.assertEqual(second.num_difference_queries, 8)
            torch.testing.assert_close(
                second.backbone.difference_query.weight, expected_query
            )
            torch.testing.assert_close(backbone_after, backbone_before)
            torch.testing.assert_close(action_after, action_before)
            with self.assertRaisesRegex(NotImplementedError, "Difference Query"):
                second.backbone.model.generate(input_ids=torch.tensor([[5, 6]]))


if __name__ == "__main__":
    unittest.main()
