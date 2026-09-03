import json
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

from model.difference_query import DifferenceQuery
from model.flow_matching_action_head import FlowmatchingActionHeadConfig
from model.reasoning_vla_model import ZR0Model


class FakeSaveableModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.ones(1), requires_grad=False)
        self.config = SimpleNamespace(
            text_config=SimpleNamespace(hidden_size=4, initializer_range=0.02)
        )

    def save_pretrained(self, directory):
        Path(directory, "config.json").write_text("{}\n", encoding="utf-8")


class FakeProcessor:
    def save_pretrained(self, directory):
        Path(directory, "processor_config.json").write_text("{}\n", encoding="utf-8")


class FakeBackbone(nn.Module):
    def __init__(
        self,
        _model_name,
        _tune_vlm,
        _lora_args,
        *,
        resolved_difference_query_config,
    ):
        super().__init__()
        self.model = FakeSaveableModel()
        self.processor = FakeProcessor()
        self.use_difference_query = resolved_difference_query_config.enabled
        self.num_difference_queries = resolved_difference_query_config.num_difference_queries
        self.compute_vlm_loss_calls = []
        self.difference_query = None
        if self.use_difference_query:
            self.difference_query = DifferenceQuery(
                self.num_difference_queries, hidden_size=4, initializer_std=0.0
            )
            if resolved_difference_query_config.checkpoint_tensor is not None:
                with torch.no_grad():
                    self.difference_query.weight.copy_(
                        resolved_difference_query_config.checkpoint_tensor
                    )

    def prepare_inputs(self, batch):
        return batch

    def forward(self, batch, compute_vlm_loss=True):
        self.compute_vlm_loss_calls.append(compute_vlm_loss)
        batch_size = batch["input_ids"].shape[0]
        embeddings = self.difference_query.weight.unsqueeze(0).expand(
            batch_size, -1, -1
        )
        return BatchFeature(
            {
                "backbone_embeddings": embeddings,
                "action_expert_cross_attn_mask": torch.ones(
                    batch_size,
                    self.num_difference_queries,
                    dtype=torch.bool,
                    device=embeddings.device,
                ),
                "vlm_loss": embeddings.square().mean() if compute_vlm_loss else None,
            }
        )


class FakeActionExpert(nn.Module):
    def __init__(self, config, _tune_action_expert):
        super().__init__()
        self.config = config
        self.scale = nn.Parameter(torch.tensor(2.0))

    def prepare_inputs(self, batch):
        return batch

    def forward(self, backbone_outputs, _action_inputs, _training_progress):
        loss = (backbone_outputs["backbone_embeddings"] * self.scale).square().mean()
        return BatchFeature({"action_expert_loss": loss})

    def get_action(self, backbone_outputs, _action_inputs, _num_denoised_steps):
        batch_size = backbone_outputs["backbone_embeddings"].shape[0]
        value = backbone_outputs["backbone_embeddings"].mean() * self.scale
        return BatchFeature(
            {
                "action_pred": value.expand(
                    batch_size, self.config.action_horizon, self.config.action_dim
                )
            }
        )


class ZR0DifferenceQueryTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.config = FlowmatchingActionHeadConfig(
            action_dim=2,
            state_dim=2,
            action_horizon=3,
        )
        self.patches = (
            patch("model.reasoning_vla_model.QwenVLBackbone", FakeBackbone),
            patch("model.reasoning_vla_model.FlowmatchingActionHead", FakeActionExpert),
        )
        for active_patch in self.patches:
            active_patch.start()

    def tearDown(self):
        for active_patch in reversed(self.patches):
            active_patch.stop()
        self.temp_dir.cleanup()

    def make_model(self, **kwargs):
        loss_type = kwargs.get("loss_type", "vlm_and_action")
        return ZR0Model(
            vlm_name_or_path="base-vlm",
            action_expert_name_or_path=None,
            action_expert_config=self.config,
            tune_vlm=loss_type != "action",
            tune_action_expert=loss_type != "vlm",
            use_difference_query=True,
            num_difference_queries=8,
            **kwargs,
        )

    @staticmethod
    def batch(batch_size=2):
        return BatchFeature(
            {
                "input_ids": torch.ones(batch_size, 3, dtype=torch.long),
                "attention_mask": torch.ones(batch_size, 3, dtype=torch.long),
                "labels": torch.tensor(
                    [[-100, 1, -100]] * batch_size, dtype=torch.long
                ),
                "observation.state": torch.ones(batch_size, 2),
                "state_mask": torch.ones(batch_size, 2, dtype=torch.bool),
                "action": torch.ones(batch_size, 3, 2),
                "action_mask": torch.ones(batch_size, 3, 2, dtype=torch.bool),
            }
        )

    def test_query_is_optimized_and_action_loss_produces_nonzero_finite_gradient(self):
        model = self.make_model(loss_type="action")
        with torch.no_grad():
            model.backbone.difference_query.weight.fill_(0.5)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        query_parameter = model.backbone.difference_query.weight
        self.assertTrue(
            any(query_parameter is parameter for group in optimizer.param_groups for parameter in group["params"])
        )
        outputs = model(self.batch(), training_progress=0.0)
        outputs.loss.backward()

        self.assertIsNotNone(query_parameter.grad)
        self.assertTrue(torch.isfinite(query_parameter.grad).all())
        self.assertGreater(query_parameter.grad.abs().sum().item(), 0.0)
        self.assertEqual(model.backbone.compute_vlm_loss_calls, [False])

    def test_loss_modes_and_direct_action_route_vlm_loss_explicitly(self):
        batch = self.batch(batch_size=1)
        ar_model = self.make_model(loss_type="vlm")
        joint_model = self.make_model(loss_type="vlm_and_action")

        ar_model(batch, training_progress=0.0)
        joint_model(batch, training_progress=0.0)
        joint_model.get_action_direct(batch)
        joint_model.get_n_actions_direct(batch, n=1)

        self.assertEqual(ar_model.backbone.compute_vlm_loss_calls, [True])
        self.assertEqual(joint_model.backbone.compute_vlm_loss_calls, [True, False, False])

    def test_action_conditioning_boundary_rejects_wrong_length_hidden_and_mask(self):
        model = self.make_model()
        valid = BatchFeature(
            {
                "backbone_embeddings": torch.zeros(2, 8, 4),
                "action_expert_cross_attn_mask": torch.ones(2, 8, dtype=torch.bool),
            }
        )
        model._validate_action_conditioning(valid, expected_batch_size=2)

        invalid_cases = (
            ({**valid, "backbone_embeddings": torch.zeros(2, 7, 4)}, "Nq"),
            ({**valid, "backbone_embeddings": torch.zeros(2, 8, 5)}, "hidden"),
            ({**valid, "action_expert_cross_attn_mask": torch.ones(2, 7, dtype=torch.bool)}, "mask shape"),
            ({**valid, "action_expert_cross_attn_mask": torch.ones(2, 8)}, "bool"),
            ({**valid, "action_expert_cross_attn_mask": torch.zeros(2, 8, dtype=torch.bool)}, "all True"),
        )
        for values, message in invalid_cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    model._validate_action_conditioning(
                        BatchFeature(values), expected_batch_size=2
                    )

    def test_action_only_detach_conflict_and_generate_are_rejected(self):
        model = self.make_model(
            loss_type="action", detach_vlm_outputs_for_action_expert=True
        )
        with self.assertRaisesRegex(ValueError, "detach_vlm_outputs"):
            model(self.batch(), training_progress=0.0)
        with self.assertRaises(NotImplementedError):
            model.get_action_subtask(self.batch())

    def test_save_and_from_pretrained_round_trip_query_values_and_disabled_config(self):
        save_directory = Path(self.temp_dir.name) / "checkpoint"
        model = self.make_model()
        expected = torch.arange(32, dtype=torch.float32).reshape(8, 4)
        with torch.no_grad():
            model.backbone.difference_query.weight.copy_(expected)
        fixed_batch = self.batch(batch_size=1)
        backbone_before_save = model.backbone(fixed_batch).backbone_embeddings
        torch.manual_seed(97)
        action_before_save = model.get_action_direct(fixed_batch).action_pred
        model.save_pretrained(save_directory)

        saved_config = json.loads(
            (save_directory / "difference_query_config.json").read_text(encoding="utf-8")
        )
        self.assertTrue(saved_config["enabled"])
        loaded = ZR0Model.from_pretrained(save_directory)
        self.assertTrue(loaded.use_difference_query)
        torch.testing.assert_close(loaded.backbone.difference_query.weight, expected)
        backbone_after_load = loaded.backbone(fixed_batch).backbone_embeddings
        torch.testing.assert_close(backbone_after_load, backbone_before_save)
        torch.manual_seed(97)
        action_after_load = loaded.get_action_direct(fixed_batch).action_pred
        torch.testing.assert_close(action_after_load, action_before_save)

        disabled_directory = Path(self.temp_dir.name) / "disabled"
        disabled = ZR0Model(
            vlm_name_or_path="base-vlm",
            action_expert_name_or_path=None,
            action_expert_config=self.config,
            tune_vlm=False,
            tune_action_expert=False,
        )
        disabled.save_pretrained(disabled_directory)
        disabled_config = json.loads(
            (disabled_directory / "difference_query_config.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertFalse(disabled_config["enabled"])
        self.assertFalse((disabled_directory / "difference_query.safetensors").exists())


if __name__ == "__main__":
    unittest.main()
