import sys
import unittest
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from transformers import AutoProcessor, Qwen3VLConfig, Qwen3VLForConditionalGeneration
from transformers.feature_extraction_utils import BatchFeature


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = Path("/opt/data/private/lq/models/Qwen3-VL-2B-Instruct")
sys.path.insert(0, str(ROOT))

from model.difference_query import DifferenceQuery, build_difference_query_sequence
from model.qwen_vl_backbone import QwenVLBackbone


@unittest.skipUnless(MODEL_DIR.is_dir(), "local Qwen3-VL processor is unavailable")
class DifferenceQueryImageTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.processor = AutoProcessor.from_pretrained(MODEL_DIR)

    @staticmethod
    def make_model() -> Qwen3VLForConditionalGeneration:
        config = Qwen3VLConfig(
            text_config={
                "vocab_size": 151936,
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 8,
                "max_position_embeddings": 256,
                "attention_dropout": 0.0,
                "tie_word_embeddings": True,
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
                "deepstack_visual_indexes": [0],
            },
            image_token_id=151655,
            video_token_id=151656,
            vision_start_token_id=151652,
            vision_end_token_id=151653,
            tie_word_embeddings=True,
        )
        config._attn_implementation = "sdpa"
        torch.manual_seed(13)
        model = Qwen3VLForConditionalGeneration(config)
        model.eval()
        return model

    def make_inputs(self) -> BatchFeature:
        image = Image.new("RGB", (32, 32), color=(90, 30, 210))
        text = (
            "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|> "
            "move the block<|im_end|>\n<|im_start|>assistant\nanswer"
        )
        processed = self.processor(
            text=[text], images=[image], return_tensors="pt"
        )
        pad_id = self.processor.tokenizer.pad_token_id
        input_ids = torch.cat(
            [processed.input_ids, torch.tensor([[pad_id]])], dim=1
        )
        attention_mask = torch.cat(
            [processed.attention_mask, torch.zeros(1, 1, dtype=torch.long)], dim=1
        )
        labels = torch.full_like(input_ids, -100)
        valid_length = int(attention_mask.sum().item())
        labels[0, valid_length - 1] = input_ids[0, valid_length - 1]
        return BatchFeature(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "pixel_values": processed.pixel_values,
                "image_grid_thw": processed.image_grid_thw,
                "sub_task_flag": torch.tensor([0]),
            }
        )

    def test_processor_vision_scatter_deepstack_and_mrope_match_inputs_embeds_path(self):
        model = self.make_model()
        inputs = self.make_inputs()
        image_token_id = model.config.image_token_id
        original_image_token_count = int((inputs.input_ids == image_token_id).sum())

        placeholder_records = []
        deepstack_records = []
        original_placeholder = model.model.get_placeholder_mask
        original_deepstack = model.model.language_model._deepstack_process
        original_lm_head = model.lm_head

        class CountingLMHead(nn.Module):
            def __init__(self, wrapped):
                super().__init__()
                self.wrapped = wrapped
                self.sequence_lengths = []

            def forward(self, hidden_states):
                self.sequence_lengths.append(hidden_states.shape[1])
                return self.wrapped(hidden_states)

        counting_lm_head = CountingLMHead(original_lm_head)

        def record_placeholder(
            input_ids, inputs_embeds, image_features=None, video_features=None
        ):
            result = original_placeholder(
                input_ids,
                inputs_embeds,
                image_features=image_features,
                video_features=video_features,
            )
            placeholder_records.append(
                {
                    "input_ids_is_none": input_ids is None,
                    "image_mask": result[0][..., 0].detach().clone(),
                    "feature_rows": image_features.shape[0],
                }
            )
            return result

        def record_deepstack(hidden_states, visual_pos_masks, visual_embeds):
            deepstack_records.append(
                {
                    "positions": torch.nonzero(
                        visual_pos_masks, as_tuple=False
                    ).detach().clone(),
                    "embeds": visual_embeds.detach().clone(),
                }
            )
            return original_deepstack(
                hidden_states, visual_pos_masks, visual_embeds
            )

        model.model.get_placeholder_mask = record_placeholder
        model.model.language_model._deepstack_process = record_deepstack
        try:
            model(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                pixel_values=inputs.pixel_values,
                image_grid_thw=inputs.image_grid_thw,
                output_hidden_states=True,
                use_cache=False,
            )
            original_placeholder_record = placeholder_records.pop()
            original_deepstack_record = deepstack_records.pop()

            direct_backbone = QwenVLBackbone.__new__(QwenVLBackbone)
            nn.Module.__init__(direct_backbone)
            direct_backbone.tune_vlm = False
            direct_backbone.model = model
            direct_backbone.use_difference_query = False
            direct_backbone.num_difference_queries = None
            direct_backbone.query_placeholder_token_id = 0
            direct_backbone.difference_query = None
            direct_backbone.eval()
            model.lm_head = counting_lm_head
            direct_outputs = direct_backbone(inputs, compute_vlm_loss=False)
            direct_placeholder_record = placeholder_records.pop()
            direct_deepstack_record = deepstack_records.pop()

            backbone = QwenVLBackbone.__new__(QwenVLBackbone)
            nn.Module.__init__(backbone)
            backbone.tune_vlm = False
            backbone.model = model
            backbone.use_difference_query = True
            backbone.num_difference_queries = 2
            backbone.query_placeholder_token_id = 0
            backbone.difference_query = DifferenceQuery(
                2, hidden_size=32, initializer_std=0.02
            )
            backbone.eval()
            outputs = backbone(inputs)
            query_placeholder_record = placeholder_records.pop()
            query_deepstack_record = deepstack_records.pop()
        finally:
            model.model.get_placeholder_mask = original_placeholder
            model.model.language_model._deepstack_process = original_deepstack
            model.lm_head = original_lm_head

        sequence = build_difference_query_sequence(
            inputs.input_ids,
            inputs.attention_mask,
            labels=inputs.labels,
            num_queries=2,
            placeholder_token_id=0,
        )
        self.assertEqual(
            int((sequence.auxiliary_input_ids == image_token_id).sum()),
            original_image_token_count,
        )
        self.assertFalse(original_placeholder_record["input_ids_is_none"])
        self.assertFalse(direct_placeholder_record["input_ids_is_none"])
        self.assertTrue(query_placeholder_record["input_ids_is_none"])
        self.assertEqual(
            int(original_placeholder_record["image_mask"].sum()),
            original_image_token_count,
        )
        self.assertEqual(
            int(query_placeholder_record["image_mask"].sum()),
            original_image_token_count,
        )
        self.assertEqual(
            original_placeholder_record["feature_rows"], original_image_token_count
        )
        self.assertEqual(
            direct_placeholder_record["feature_rows"], original_image_token_count
        )
        self.assertEqual(
            query_placeholder_record["feature_rows"], original_image_token_count
        )
        torch.testing.assert_close(
            query_deepstack_record["positions"],
            original_deepstack_record["positions"],
        )
        torch.testing.assert_close(
            direct_deepstack_record["positions"],
            original_deepstack_record["positions"],
        )
        torch.testing.assert_close(
            query_deepstack_record["embeds"],
            original_deepstack_record["embeds"],
        )
        torch.testing.assert_close(
            direct_deepstack_record["embeds"],
            original_deepstack_record["embeds"],
        )

        original_positions, _ = model.model.get_rope_index(
            inputs.input_ids,
            image_grid_thw=inputs.image_grid_thw,
            attention_mask=inputs.attention_mask.bool(),
        )
        query_positions, _ = model.model.get_rope_index(
            sequence.auxiliary_input_ids,
            image_grid_thw=inputs.image_grid_thw,
            attention_mask=sequence.valid_attention_mask,
        )
        original_visual_mask = inputs.input_ids == image_token_id
        query_visual_mask = sequence.auxiliary_input_ids == image_token_id
        torch.testing.assert_close(
            query_positions[:, query_visual_mask],
            original_positions[:, original_visual_mask],
        )
        self.assertEqual(outputs.backbone_embeddings.shape, (1, 2, 32))
        self.assertEqual(direct_outputs.backbone_embeddings.shape[:2], inputs.input_ids.shape)
        self.assertEqual(counting_lm_head.sequence_lengths[0], 0)
        self.assertEqual(len(counting_lm_head.sequence_lengths), 2)
        self.assertGreater(counting_lm_head.sequence_lengths[1], 0)
        self.assertTrue(torch.isfinite(outputs.backbone_embeddings).all())


if __name__ == "__main__":
    unittest.main()
