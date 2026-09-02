import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn
from transformers.feature_extraction_utils import BatchFeature


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.difference_query import DifferenceQuery, save_difference_query_artifacts
from model.qwen_vl_backbone import QwenVLBackbone


class FakeLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.RMSNorm(2)


class FakeQwenBody(nn.Module):
    def __init__(self):
        super().__init__()
        self.language_model = FakeLanguageModel()
        self.rope_call = None

    def get_rope_index(
        self,
        input_ids,
        image_grid_thw=None,
        video_grid_thw=None,
        attention_mask=None,
    ):
        self.rope_call = {
            "input_ids": input_ids,
            "image_grid_thw": image_grid_thw,
            "video_grid_thw": video_grid_thw,
            "attention_mask": attention_mask,
        }
        positions = attention_mask.long().cumsum(-1) - 1
        positions.masked_fill_(~attention_mask, 1)
        return positions.unsqueeze(0).expand(3, -1, -1), torch.zeros(
            input_ids.shape[0], 1, dtype=input_ids.dtype
        )


class FakeConditionalModel(nn.Module):
    def __init__(self, hidden_state: torch.Tensor | None):
        super().__init__()
        self.model = FakeQwenBody()
        self.hidden_state = hidden_state
        self.embedding = nn.Embedding(200, 2)
        self.config = SimpleNamespace(
            text_config=SimpleNamespace(hidden_size=2, initializer_range=0.02),
        )
        self.forward_call = None
        self.generate_call = None

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, **kwargs):
        self.forward_call = kwargs
        hidden_state = (
            kwargs["inputs_embeds"]
            if self.hidden_state is None
            else self.hidden_state
        )
        return {
            "hidden_states": (hidden_state,),
            "loss": torch.tensor(0.25) if "labels" in kwargs else None,
        }

    def generate(self, **kwargs):
        self.generate_call = kwargs
        return torch.tensor([[42]])


class FakePeftModel(nn.Module):
    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base_model_for_test = base_model
        self.forward_call_count = 0

    def get_base_model(self):
        return self.base_model_for_test

    def forward(self, **kwargs):
        self.forward_call_count += 1
        return self.base_model_for_test(**kwargs)

    def generate(self, **kwargs):
        return self.base_model_for_test.generate(**kwargs)


class QwenVLBackboneTest(unittest.TestCase):
    @staticmethod
    def make_backbone(model: nn.Module) -> QwenVLBackbone:
        backbone = QwenVLBackbone.__new__(QwenVLBackbone)
        nn.Module.__init__(backbone)
        backbone.tune_vlm = True
        backbone.model = model
        backbone.use_difference_query = False
        backbone.num_difference_queries = None
        backbone.difference_query = None
        backbone.query_placeholder_token_id = 0
        return backbone

    def test_forward_returns_final_rmsnorm_for_plain_and_peft_models(self):
        raw = torch.tensor([[[3.0, 4.0], [0.0, 2.0]]])

        for wrap_in_peft in (False, True):
            with self.subTest(wrap_in_peft=wrap_in_peft):
                base_model = FakeConditionalModel(raw)
                model = FakePeftModel(base_model) if wrap_in_peft else base_model
                backbone = self.make_backbone(model)
                inputs = BatchFeature(
                    {
                        "input_ids": torch.tensor([[1, 2]]),
                        "sub_task_flag": torch.tensor([0]),
                    }
                )

                outputs = backbone(inputs)

                expected = base_model.model.language_model.norm(raw)
                torch.testing.assert_close(outputs.backbone_embeddings, expected)
                self.assertFalse(torch.equal(outputs.backbone_embeddings, raw))

    def test_disabled_eval_path_preserves_original_qwen_call_and_output_shapes(self):
        raw = torch.tensor([[[3.0, 4.0], [0.0, 2.0], [5.0, 1.0]]])
        model = FakeConditionalModel(raw)
        backbone = self.make_backbone(model)
        backbone.eval()
        inputs = BatchFeature(
            {
                "input_ids": torch.tensor([[1, 2, 3]]),
                "attention_mask": torch.tensor([[1, 1, 1]]),
                "pixel_values": torch.ones(2, 3),
                "image_grid_thw": torch.tensor([[1, 2, 2]]),
                "sub_task_flag": torch.tensor([0]),
            }
        )

        outputs = backbone(inputs)

        call = model.forward_call
        self.assertIs(call["input_ids"], inputs["input_ids"])
        self.assertIs(call["attention_mask"], inputs["attention_mask"])
        self.assertNotIn("inputs_embeds", call)
        self.assertNotIn("position_ids", call)
        self.assertEqual(call["attention_mask"].ndim, 2)
        self.assertTrue(call["use_cache"])
        self.assertTrue(call["return_dict"])
        self.assertTrue(call["output_hidden_states"])
        self.assertEqual(outputs.backbone_embeddings.shape, raw.shape)
        torch.testing.assert_close(
            outputs.action_expert_cross_attn_mask,
            torch.ones(1, 3, dtype=torch.bool),
        )

    def test_disabled_action_only_uses_conditional_wrapper_with_head_arguments(self):
        raw = torch.tensor([[[3.0, 4.0], [0.0, 2.0], [5.0, 1.0]]])
        labels = torch.tensor([[-100, 2, 3]])

        for training, wrap_in_peft in ((True, False), (False, True)):
            with self.subTest(training=training, wrap_in_peft=wrap_in_peft):
                base_model = FakeConditionalModel(raw)
                model = FakePeftModel(base_model) if wrap_in_peft else base_model
                backbone = self.make_backbone(model)
                backbone.train(training)
                inputs = BatchFeature(
                    {
                        "input_ids": torch.tensor([[1, 2, 3]]),
                        "attention_mask": torch.tensor([[1, 1, 1]]),
                        "labels": labels,
                        "pixel_values": torch.ones(2, 3),
                        "image_grid_thw": torch.tensor([[1, 2, 2]]),
                        "sub_task_flag": torch.tensor([0]),
                    }
                )

                outputs = backbone(inputs, compute_vlm_loss=False)

                call = base_model.forward_call
                self.assertIs(call["input_ids"], inputs["input_ids"])
                self.assertIs(call["attention_mask"], inputs["attention_mask"])
                self.assertIs(call["labels"], labels)
                self.assertEqual(call["attention_mask"].ndim, 2)
                self.assertNotIn("inputs_embeds", call)
                self.assertNotIn("position_ids", call)
                self.assertTrue(call["return_dict"])
                self.assertTrue(call["output_hidden_states"])
                self.assertEqual(call["use_cache"], not training)
                self.assertEqual(
                    getattr(model, "forward_call_count", 1),
                    1,
                    "the PEFT wrapper and its forward hooks must not be bypassed",
                )
                self.assertEqual(outputs.vlm_loss.item(), 0.25)
                expected = base_model.model.language_model.norm(raw)
                torch.testing.assert_close(outputs.backbone_embeddings, expected)

    def test_enabled_path_injects_queries_and_gathers_only_query_hidden_states(self):
        model = FakeConditionalModel(hidden_state=None)
        with torch.no_grad():
            model.embedding.weight.copy_(
                torch.arange(400, dtype=torch.float32).reshape(200, 2)
            )
        backbone = self.make_backbone(model)
        backbone.use_difference_query = True
        backbone.num_difference_queries = 2
        backbone.difference_query = DifferenceQuery(2, 2, initializer_std=0.0)
        with torch.no_grad():
            backbone.difference_query.weight.copy_(
                torch.tensor([[3.0, 4.0], [0.0, 2.0]])
            )
        inputs = BatchFeature(
            {
                "input_ids": torch.tensor([[10, 11, 31, 99]]),
                "attention_mask": torch.tensor([[1, 1, 1, 0]]),
                "labels": torch.tensor([[-100, -100, 31, -100]]),
                "pixel_values": torch.ones(2, 3),
                "image_grid_thw": torch.tensor([[1, 2, 2]]),
                "sub_task_flag": torch.tensor([0]),
            }
        )

        outputs = backbone(inputs)

        call = model.forward_call
        self.assertNotIn("input_ids", call)
        self.assertEqual(call["inputs_embeds"].shape, (1, 6, 2))
        self.assertEqual(call["attention_mask"].shape, (1, 1, 6, 6))
        self.assertEqual(call["attention_mask"].dtype, torch.bool)
        self.assertEqual(call["position_ids"].shape, (3, 1, 6))
        self.assertFalse(call["use_cache"])
        torch.testing.assert_close(
            call["labels"],
            torch.tensor([[-100, -100, -100, -100, 31, -100]]),
        )
        torch.testing.assert_close(
            model.model.rope_call["input_ids"],
            torch.tensor([[10, 11, 0, 0, 31, 99]]),
        )
        torch.testing.assert_close(
            model.model.rope_call["attention_mask"],
            torch.tensor([[True, True, True, True, True, False]]),
        )
        expected = model.model.language_model.norm(
            torch.tensor([[[3.0, 4.0], [0.0, 2.0]]])
        )
        torch.testing.assert_close(outputs.backbone_embeddings, expected)
        torch.testing.assert_close(
            outputs.action_expert_cross_attn_mask,
            torch.ones(1, 2, dtype=torch.bool),
        )
        self.assertEqual(outputs.vlm_loss.item(), 0.25)

    def test_disabled_checkpoint_hidden_size_must_match_loaded_vlm(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            save_difference_query_artifacts(
                temp_dir,
                enabled=False,
                hidden_size=4,
                difference_query=None,
            )
            model = FakeConditionalModel(hidden_state=None)
            model.config._attn_implementation = "eager"
            processor = SimpleNamespace(tokenizer=SimpleNamespace())

            with (
                patch(
                    "model.qwen_vl_backbone.Qwen3VLForConditionalGeneration.from_pretrained",
                    return_value=model,
                ),
                patch(
                    "model.qwen_vl_backbone.AutoProcessor.from_pretrained",
                    return_value=processor,
                ),
                self.assertRaisesRegex(ValueError, "hidden size"),
            ):
                QwenVLBackbone(
                    temp_dir,
                    tune_vlm=False,
                    lora_args=None,
                )

    def test_generation_guard_blocks_query_mode_and_preserves_disabled_mode(self):
        disabled_model = FakeConditionalModel(hidden_state=None)
        disabled = self.make_backbone(disabled_model)
        generated = disabled.generate(input_ids=torch.tensor([[1]]))
        torch.testing.assert_close(generated, torch.tensor([[42]]))
        self.assertIsNotNone(disabled_model.generate_call)

        query_model = FakeConditionalModel(hidden_state=None)
        query = self.make_backbone(query_model)
        query.use_difference_query = True
        query._install_generation_guard()

        with self.assertRaisesRegex(NotImplementedError, "Difference Query"):
            query.generate(input_ids=torch.tensor([[1]]))
        with self.assertRaisesRegex(NotImplementedError, "Difference Query"):
            query.model.generate(input_ids=torch.tensor([[1]]))

    def test_generation_guard_also_blocks_peft_base_model(self):
        base_model = FakeConditionalModel(hidden_state=None)
        peft_model = FakePeftModel(base_model)
        query = self.make_backbone(peft_model)
        query.use_difference_query = True
        query._install_generation_guard()

        with self.assertRaisesRegex(NotImplementedError, "Difference Query"):
            query.generate(input_ids=torch.tensor([[1]]))
        with self.assertRaisesRegex(NotImplementedError, "Difference Query"):
            query.model.generate(input_ids=torch.tensor([[1]]))
        with self.assertRaisesRegex(NotImplementedError, "Difference Query"):
            query.model.get_base_model().generate(input_ids=torch.tensor([[1]]))


if __name__ == "__main__":
    unittest.main()
