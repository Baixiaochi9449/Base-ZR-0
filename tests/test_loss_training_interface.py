import math
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn
from transformers.feature_extraction_utils import BatchFeature


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import train_vla
from model.flow_matching_action_head import FlowmatchingActionHeadConfig
from model.reasoning_vla_model import ZR0Model
from utils.cli_options import parse_train_options
from utils.wandb_training_logger import WandbTrainingLogger


class _FakeSaveableModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(
            text_config=SimpleNamespace(hidden_size=4, initializer_range=0.02)
        )


class _FakeBackbone(nn.Module):
    def __init__(self, *_args, resolved_difference_query_config, **_kwargs):
        super().__init__()
        self.model = _FakeSaveableModel()
        self.processor = SimpleNamespace()
        self.use_difference_query = resolved_difference_query_config.enabled
        self.num_difference_queries = resolved_difference_query_config.num_difference_queries
        self.embedding = nn.Parameter(torch.ones(1, 4))
        self.prepare_calls = 0
        self.forward_calls = []

    def prepare_inputs(self, batch):
        self.prepare_calls += 1
        return batch

    def forward(self, batch, compute_vlm_loss=True):
        self.forward_calls.append(compute_vlm_loss)
        embeddings = self.embedding.unsqueeze(0).expand(batch["input_ids"].shape[0], -1, -1)
        return BatchFeature(
            {
                "backbone_embeddings": embeddings,
                "action_expert_cross_attn_mask": torch.ones(
                    embeddings.shape[:2], dtype=torch.bool
                ),
                "vlm_loss": embeddings.square().mean() if compute_vlm_loss else None,
            }
        )


class _FakeActionExpert(nn.Module):
    def __init__(self, config, _tune_action_expert):
        super().__init__()
        self.config = config
        self.scale = nn.Parameter(torch.tensor(2.0))
        self.prepare_calls = 0
        self.forward_calls = 0
        self.requires_grad_(_tune_action_expert)

    def prepare_inputs(self, batch):
        self.prepare_calls += 1
        return batch

    def forward(self, backbone_outputs, _action_inputs, _training_progress):
        self.forward_calls += 1
        return BatchFeature(
            {"action_expert_loss": (backbone_outputs["backbone_embeddings"] * self.scale).square().mean()}
        )


class ModelLossContractTest(unittest.TestCase):
    def setUp(self):
        self.config = FlowmatchingActionHeadConfig(action_dim=2, state_dim=2, action_horizon=3)
        self.patches = (
            patch("model.reasoning_vla_model.QwenVLBackbone", _FakeBackbone),
            patch("model.reasoning_vla_model.FlowmatchingActionHead", _FakeActionExpert),
        )
        for active_patch in self.patches:
            active_patch.start()

    def tearDown(self):
        for active_patch in reversed(self.patches):
            active_patch.stop()

    def make_model(self, *, tune_vlm=None, tune_action_expert=None, **kwargs):
        loss_type = kwargs.get("loss_type", "vlm_and_action")
        if tune_vlm is None:
            tune_vlm = loss_type != "action"
        if tune_action_expert is None:
            tune_action_expert = loss_type != "vlm"
        return ZR0Model(
            vlm_name_or_path="base-vlm",
            action_expert_name_or_path=None,
            action_expert_config=self.config,
            tune_vlm=tune_vlm,
            tune_action_expert=tune_action_expert,
            **kwargs,
        )

    @staticmethod
    def batch(**overrides):
        batch = {
            "input_ids": torch.ones(2, 3, dtype=torch.long),
            "attention_mask": torch.ones(2, 3, dtype=torch.long),
            "labels": torch.tensor([[-100, 1, -100], [-100, 1, -100]]),
            "observation.state": torch.ones(2, 2),
            "state_mask": torch.ones(2, 2, dtype=torch.bool),
            "action": torch.ones(2, 3, 2),
            "action_mask": torch.ones(2, 3, 2, dtype=torch.bool),
        }
        batch.update(overrides)
        return BatchFeature(batch)

    def test_loss_modes_use_weighted_ar_and_flow_matching_contract(self):
        vlm = self.make_model(loss_type="vlm")(
            self.batch(), 0.0, vlm_loss_weight=2.0
        )
        action = self.make_model(loss_type="action")(
            self.batch(), 0.0, action_expert_loss_weight=3.0
        )
        joint = self.make_model(loss_type="vlm_and_action")(
            self.batch(),
            0.0,
            vlm_loss_weight=2.0,
            action_expert_loss_weight=3.0,
        )

        for outputs, raw_key, weighted_key, weight in (
            (vlm, "ar_loss", "weighted_ar_loss", 2.0),
            (action, "flow_matching_loss", "weighted_flow_matching_loss", 3.0),
        ):
            self.assertIn("loss", outputs)
            self.assertIn("total_loss", outputs)
            self.assertIn(raw_key, outputs)
            self.assertIn(weighted_key, outputs)
            torch.testing.assert_close(outputs.loss, outputs.total_loss)
            torch.testing.assert_close(outputs.loss, outputs[weighted_key])
            torch.testing.assert_close(outputs[weighted_key], outputs[raw_key] * weight)

        self.assertIn("vlm_loss", joint)
        self.assertIn("action_expert_loss", joint)
        self.assertIn("weighted_vlm_loss", joint)
        self.assertIn("weighted_action_expert_loss", joint)
        self.assertIn("vlm_loss weight", joint)
        self.assertIn("action_expert_loss weight", joint)
        torch.testing.assert_close(joint.weighted_vlm_loss, joint.weighted_ar_loss)
        torch.testing.assert_close(joint.weighted_action_expert_loss, joint.weighted_flow_matching_loss)
        torch.testing.assert_close(joint.loss, joint.weighted_ar_loss + joint.weighted_flow_matching_loss)
        torch.testing.assert_close(joint.loss, joint.total_loss)
        self.assertEqual(joint.vlm_loss_weight, 2.0)
        self.assertEqual(joint.action_expert_loss_weight, 3.0)

    def test_forward_exposes_differentiable_loss_numerators_and_exact_counts(self):
        ar_model = self.make_model(loss_type="vlm")
        ar_batch = self.batch(
            labels=torch.tensor([[-100, 1, 2], [-100, 3, -100]])
        )
        ar_outputs = ar_model(ar_batch, 0.0)
        self.assertEqual(ar_outputs.ar_loss_count.item(), 3)
        torch.testing.assert_close(
            ar_outputs.ar_loss_sum,
            ar_outputs.ar_loss * ar_outputs.ar_loss_count,
        )
        self.assertTrue(ar_outputs.ar_loss_sum.requires_grad)

        action_model = self.make_model(loss_type="action")
        action_batch = self.batch(
            action_mask=torch.tensor(
                [
                    [[1, 0], [1, 0], [0, 0]],
                    [[1, 1], [0, 0], [0, 0]],
                ],
                dtype=torch.bool,
            )
        )
        action_outputs = action_model(action_batch, 0.0)
        self.assertEqual(action_outputs.flow_matching_loss_count.item(), 4)
        torch.testing.assert_close(
            action_outputs.flow_matching_loss_sum,
            action_outputs.flow_matching_loss
            * action_outputs.flow_matching_loss_count,
        )
        self.assertTrue(action_outputs.flow_matching_loss_sum.requires_grad)

    def test_forward_loss_mode_is_locked_to_constructed_mode(self):
        joint_model = self.make_model(loss_type="vlm_and_action")
        with self.assertRaisesRegex(
            ValueError,
            "constructed mode='vlm_and_action'.*requested mode='action'",
        ):
            joint_model(self.batch(), 0.0, loss_type="action")
        self.assertEqual(joint_model.backbone.forward_calls, [])

        ar_model = self.make_model(
            tune_vlm=True,
            tune_action_expert=False,
            loss_type="vlm",
        )
        with self.assertRaisesRegex(
            ValueError,
            "constructed mode='vlm'.*requested mode='vlm_and_action'",
        ):
            ar_model(self.batch(), 0.0, dynamic_loss_type="vlm_and_action")
        self.assertEqual(ar_model.backbone.forward_calls, [])

        outputs = joint_model(self.batch(), 0.0)
        self.assertIn("ar_loss", outputs)
        self.assertIn("flow_matching_loss", outputs)

    def test_ar_only_does_not_prepare_or_run_action_expert(self):
        model = self.make_model(loss_type="vlm")
        model(self.batch(), 0.0)
        self.assertIsNone(model.action_expert)

    def test_ar_only_construction_allocates_no_action_expert_parameters(self):
        model = self.make_model(
            tune_vlm=True,
            tune_action_expert=False,
            loss_type="vlm",
        )
        self.assertIsNone(model.action_expert)
        self.assertFalse(
            any(name.startswith("action_expert.") for name, _ in model.named_parameters())
        )
        outputs = model(self.batch(), 0.0, loss_type="vlm")
        outputs.loss.backward()
        self.assertIsNotNone(model.backbone.embedding.grad)

    def test_ar_only_construction_rejects_conflicting_training_options(self):
        for kwargs, message in (
            ({"tune_vlm": False, "tune_action_expert": False}, "tune_vlm"),
            ({"tune_vlm": True, "tune_action_expert": True}, "tune_action_expert"),
            (
                {
                    "tune_vlm": True,
                    "tune_action_expert": False,
                    "action_expert_name_or_path": "forbidden",
                },
                "action_expert_name_or_path",
            ),
        ):
            with self.subTest(kwargs=kwargs):
                constructor = {
                    "vlm_name_or_path": "base-vlm",
                    "action_expert_name_or_path": None,
                    "action_expert_config": self.config,
                    "loss_type": "vlm",
                    **kwargs,
                }
                with self.assertRaisesRegex(ValueError, message):
                    ZR0Model(**constructor)

    def test_branch_gradients_and_ar_optimizer_has_no_action_expert(self):
        ar_model = self.make_model(loss_type="vlm")
        optimizer = train_vla.build_adamw_optimizer(
            ar_model, learning_rate=1e-2, beta1=0.9, beta2=0.95, epsilon=1e-8
        )
        ar_outputs = ar_model(self.batch(), 0.0)
        self.assertTrue(torch.isfinite(ar_outputs.ar_loss))
        self.assertGreater(ar_outputs.ar_loss.item(), 0)
        ar_outputs.loss.backward()
        self.assertIsNotNone(ar_model.backbone.embedding.grad)
        self.assertTrue(torch.isfinite(ar_model.backbone.embedding.grad).all())
        self.assertGreater(ar_model.backbone.embedding.grad.abs().sum().item(), 0)
        self.assertFalse(
            any(
                name.startswith("action_expert.")
                for name, _ in ar_model.named_parameters()
            )
        )
        optimizer.step()

        joint_model = self.make_model()
        joint_outputs = joint_model(
            self.batch(),
            0.0,
            loss_type="vlm_and_action",
            vlm_loss_weight=1.0,
            action_expert_loss_weight=5.0,
        )
        joint_outputs.loss.backward()
        for gradient in (
            joint_model.backbone.embedding.grad,
            joint_model.action_expert.scale.grad,
        ):
            self.assertIsNotNone(gradient)
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(gradient.abs().sum().item(), 0)

    def test_invalid_modes_weights_and_supervision_fail_before_backbone(self):
        invalid_calls = (
            ("vlm_and_action", {"loss_type": "unknown"}, "loss_type"),
            ("vlm", {"vlm_loss_weight": 0.0}, "vlm_loss_weight"),
            ("action", {"action_expert_loss_weight": float("nan")}, "action_expert_loss_weight"),
            ("vlm_and_action", {"action_expert_loss_weight": 0.0}, "action_expert_loss_weight"),
        )
        for constructed_mode, kwargs, message in invalid_calls:
            with self.subTest(kwargs=kwargs):
                model = self.make_model(loss_type=constructed_mode)
                with self.assertRaisesRegex(ValueError, message):
                    model(self.batch(), 0.0, **kwargs)
                self.assertEqual(model.backbone.forward_calls, [])

        for batch, message in (
            (self.batch(labels=None), "labels"),
            (self.batch(labels=torch.full((2, 3), -100)), "-100"),
        ):
            with self.subTest(message=message):
                model = self.make_model(loss_type="vlm")
                with self.assertRaisesRegex(ValueError, message):
                    model(batch, 0.0)
                self.assertEqual(model.backbone.forward_calls, [])

    def test_action_inputs_fail_fast_and_action_only_skips_vlm_loss(self):
        for key, value in (
            ("observation.state", None),
            ("state_mask", None),
            ("action", None),
            ("action_mask", torch.zeros(2, 3, 2, dtype=torch.bool)),
        ):
            with self.subTest(key=key):
                model = self.make_model(loss_type="action")
                batch = self.batch()
                if value is None:
                    del batch[key]
                else:
                    batch[key] = value
                with self.assertRaisesRegex(ValueError, key):
                    model(batch, 0.0)
                self.assertEqual(model.backbone.forward_calls, [])

        model = self.make_model(loss_type="action")
        model(self.batch(), 0.0)
        self.assertEqual(model.backbone.forward_calls, [False])

    def test_training_inputs_require_nonempty_tensors_matching_masks_and_batches(self):
        invalid_action_batches = (
            (self.batch(**{"observation.state": []}), "observation.state.*Tensor"),
            (self.batch(**{"state_mask": torch.ones(2, 3, dtype=torch.bool)}), "state_mask.*shape"),
            (self.batch(**{"action": torch.ones(1, 3, 2)}), "action.*batch"),
            (self.batch(**{"action_mask": torch.ones(2, 3, 1, dtype=torch.bool)}), "action_mask.*shape"),
            (
                self.batch(
                    **{
                        "action_mask": torch.tensor(
                            [
                                [[1, 1], [1, 1], [1, 1]],
                                [[0, 0], [0, 0], [0, 0]],
                            ],
                            dtype=torch.bool,
                        )
                    }
                ),
                "each sample",
            ),
        )
        for batch, message in invalid_action_batches:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    self.make_model(loss_type="action")(batch, 0.0)

        invalid_vlm_batches = (
            (self.batch(labels=torch.ones(2, 2, dtype=torch.long)), "labels.*shape"),
            (
                self.batch(labels=torch.tensor([[-100, 1, -100], [-100, -100, -100]])),
                "each sample",
            ),
        )
        for batch, message in invalid_vlm_batches:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    self.make_model(loss_type="vlm")(batch, 0.0)

    def test_state_mask_requires_a_valid_element_for_each_action_sample(self):
        model = self.make_model(loss_type="action")
        batch = self.batch(
            **{
                "state_mask": torch.tensor(
                    [[1, 1], [0, 0]], dtype=torch.bool
                )
            }
        )
        with self.assertRaisesRegex(ValueError, "state_mask.*each sample"):
            model(batch, 0.0)
        self.assertEqual(model.backbone.forward_calls, [])

    def test_action_horizon_must_match_model_config(self):
        model = self.make_model(loss_type="action")
        batch = self.batch(
            action=torch.ones(2, 2, 2),
            action_mask=torch.ones(2, 2, 2, dtype=torch.bool),
        )
        with self.assertRaisesRegex(ValueError, "action horizon.*3.*2"):
            model(batch, 0.0)
        self.assertEqual(model.backbone.forward_calls, [])

    def test_action_horizon_must_fit_action_expert_position_table(self):
        self.config.action_horizon = 4
        self.config.max_seq_len = 3
        with self.assertRaisesRegex(ValueError, "action_horizon.*max_seq_len"):
            self.make_model()

    def test_difference_query_detach_rejects_all_action_training_modes(self):
        for mode in ("action", "vlm_and_action"):
            with self.subTest(mode=mode):
                model = self.make_model(
                    loss_type=mode,
                    use_difference_query=True,
                    num_difference_queries=1,
                    detach_vlm_outputs_for_action_expert=True,
                )
                with self.assertRaisesRegex(ValueError, "detach_vlm_outputs"):
                    model(self.batch(), 0.0)


class TrainInterfaceHelperTest(unittest.TestCase):
    def test_action_config_reuses_checkpoint_architecture_and_overrides_io_shape(self):
        source = FlowmatchingActionHeadConfig(
            action_dim=8,
            state_dim=8,
            action_horizon=20,
            noise_s=0.75,
        )
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "action_expert_config.json").write_text(
                json.dumps(source.to_dict()), encoding="utf-8"
            )
            options = SimpleNamespace(
                action_expert_name_or_path=None,
                vlm_name_or_path=directory,
                max_pad_state_and_action_length=64,
                action_horizon=16,
            )
            resolved = train_vla.resolve_action_expert_config(options)
        self.assertEqual(resolved.action_dim, 64)
        self.assertEqual(resolved.state_dim, 64)
        self.assertEqual(resolved.action_horizon, 16)
        self.assertEqual(resolved.noise_s, 0.75)

    def test_optimizer_uses_only_trainable_parameters_and_rejects_empty(self):
        model = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
        for parameter in model[1].parameters():
            parameter.requires_grad_(False)
        parameters = train_vla.get_trainable_parameters(model)
        self.assertEqual(set(parameters), set(model[0].parameters()))
        optimizer = train_vla.build_adamw_optimizer(
            model, learning_rate=1e-3, beta1=0.8, beta2=0.9, epsilon=1e-5
        )
        self.assertEqual(set(optimizer.param_groups[0]["params"]), set(parameters))
        frozen = nn.Linear(2, 2)
        frozen.requires_grad_(False)
        with self.assertRaisesRegex(ValueError, "No trainable"):
            train_vla.get_trainable_parameters(frozen)

    def test_effective_horizon_warmup_and_output_metrics_are_pure(self):
        self.assertEqual(train_vla.effective_total_optimizer_steps(12, None), 12)
        self.assertEqual(train_vla.effective_total_optimizer_steps(12, 5), 5)
        self.assertEqual(train_vla.calculate_warmup_steps(100, 2, None), 16)
        self.assertEqual(train_vla.calculate_warmup_steps(100, 2, 0.25), 50)
        outputs = BatchFeature(
            {
                "loss": torch.tensor(5.0),
                "total_loss": torch.tensor(5.0),
                "ar_loss": torch.tensor(2.0),
                "weighted_ar_loss": torch.tensor(3.0),
                "vlm_loss_weight": 1.5,
            }
        )
        metrics = train_vla.loss_output_metrics(outputs)
        self.assertIsInstance(metrics["loss"], torch.Tensor)
        self.assertIsInstance(metrics["vlm_loss_weight"], torch.Tensor)
        self.assertEqual(metrics["loss"].item(), 5.0)
        self.assertEqual(metrics["total_loss"].item(), 5.0)
        self.assertEqual(metrics["ar_loss"].item(), 2.0)
        self.assertEqual(metrics["weighted_ar_loss"].item(), 3.0)
        self.assertEqual(metrics["vlm_loss_weight"].item(), 1.5)
        self.assertEqual(train_vla.tensorboard_loss_value(metrics["loss"]), 5.0)
        self.assertEqual(train_vla.tensorboard_loss_metric_name("loss"), "loss")
        self.assertEqual(train_vla.tensorboard_loss_metric_name("vlm_loss"), "vlm-loss")
        self.assertEqual(
            train_vla.tensorboard_loss_metric_name("action_expert_loss"),
            "action-expert-loss",
        )

    def test_batch_token_metrics_records_boundaries_and_truncation_ratios(self):
        batch = {
            "context_token_count": torch.tensor([10, 14]),
            "json_content_token_count": torch.tensor([5, 7]),
            "chat_termination_token_count": torch.tensor([2, 2]),
            "target_region_token_count": torch.tensor([7, 9]),
            "padding_token_count": torch.tensor([20, 18]),
            "original_target_token_count": torch.tensor([5, 7]),
            "kept_target_token_count": torch.tensor([5, 6]),
            "supervised_token_count": torch.tensor([5, 6]),
            "truncated_token_count": torch.tensor([0, 1]),
            "input_truncated": torch.tensor([False, True]),
            "target_truncated": torch.tensor([False, True]),
        }

        for key in tuple(batch):
            batch[f"{key}_valid"] = torch.tensor([True, True])

        metrics = train_vla.batch_token_metrics(batch)

        self.assertEqual(metrics["context_token_count"].item(), 12.0)
        self.assertEqual(metrics["json_content_token_count"].item(), 6.0)
        self.assertEqual(metrics["chat_termination_token_count"].item(), 2.0)
        self.assertEqual(metrics["target_region_token_count"].item(), 8.0)
        self.assertEqual(metrics["padding_token_count"].item(), 19.0)
        self.assertEqual(metrics["original_target_token_count"].item(), 6.0)
        self.assertEqual(metrics["kept_target_token_count"].item(), 5.5)
        self.assertEqual(metrics["supervised_token_count"].item(), 5.5)
        self.assertEqual(metrics["truncated_token_count"].item(), 0.5)
        self.assertEqual(metrics["input_truncated_ratio"].item(), 0.5)
        self.assertEqual(metrics["target_truncated_ratio"].item(), 0.5)
        self.assertEqual(train_vla.batch_token_metrics({}), {})

    def test_batch_token_metrics_excludes_invalid_placeholders_and_is_order_invariant(self):
        from utils.training_tokenization import token_metric_validity_key

        def make_batch(order):
            values = torch.tensor([999, 0, 8])[order]
            validity = torch.tensor([False, True, True])[order]
            return {
                "json_content_token_count": values,
                token_metric_validity_key("json_content_token_count"): validity,
                "target_truncated": torch.tensor([True, False, True])[order],
                token_metric_validity_key("target_truncated"): validity,
            }

        first = train_vla.batch_token_metrics(make_batch(torch.tensor([0, 1, 2])))
        second = train_vla.batch_token_metrics(make_batch(torch.tensor([2, 0, 1])))
        self.assertEqual(first.keys(), second.keys())
        for key in first:
            torch.testing.assert_close(first[key], second[key])
        self.assertEqual(first["json_content_token_count"].item(), 4.0)
        self.assertEqual(first["json_content_token_count_min"].item(), 0.0)
        self.assertEqual(first["json_content_token_count_max"].item(), 8.0)
        self.assertEqual(first["target_truncated_ratio"].item(), 0.5)

    def test_train_cli_validates_adam_and_warmup_values(self):
        defaults = parse_train_options([])
        self.assertEqual((defaults.adam_beta1, defaults.adam_beta2, defaults.adam_epsilon), (0.9, 0.95, 1e-6))
        self.assertIsNone(defaults.warmup_ratio)
        for args, flag in (
            (["--adam_beta1", "1"], "adam_beta1"),
            (["--adam_beta2", "-0.1"], "adam_beta2"),
            (["--adam_epsilon", "0"], "adam_epsilon"),
            (["--warmup_ratio", "1.1"], "warmup_ratio"),
        ):
            with self.subTest(args=args):
                with self.assertRaisesRegex(SystemExit, "2"):
                    parse_train_options(args)

        for value in ("0", "-1"):
            with self.subTest(action_horizon=value):
                with self.assertRaisesRegex(SystemExit, "2"):
                    parse_train_options(["--action_horizon", value])

    def test_train_cli_validates_loss_mode_and_weights(self):
        invalid = (
            (["--loss_type", "not-a-mode"],),
            (["--loss_type", "vlm", "--vlm_loss_weight", "0"],),
            (["--loss_type", "action", "--action_expert_loss_weight", "0"],),
            (["--loss_type", "vlm_and_action", "--vlm_loss_weight", "0"],),
            (["--loss_type", "vlm_and_action", "--action_expert_loss_weight", "0"],),
            (["--loss_type", "vlm", "--vlm_loss_weight", "nan"],),
            (["--loss_type", "action", "--action_expert_loss_weight", "inf"],),
            (["--loss_type", "action", "--vlm_loss_weight", "-1"],),
            (["--loss_type", "vlm", "--action_expert_loss_weight", "-1"],),
        )
        for (args,) in invalid:
            with self.subTest(args=args):
                with self.assertRaisesRegex(SystemExit, "2"):
                    parse_train_options(args)

        self.assertEqual(
            parse_train_options(
                [
                    "--loss_type", "vlm", "--tune_vlm",
                    "--action_expert_loss_weight", "0",
                ]
            ).action_expert_loss_weight,
            0.0,
        )
        self.assertEqual(
            parse_train_options(
                ["--loss_type", "action", "--vlm_loss_weight", "0"]
            ).vlm_loss_weight,
            0.0,
        )

    def test_train_cli_rejects_ar_only_model_construction_conflicts(self):
        for args in (
            ["--loss_type", "vlm"],
            ["--loss_type", "vlm", "--tune_vlm", "--tune_action_expert"],
            [
                "--loss_type", "vlm", "--tune_vlm",
                "--action_expert_name_or_path", "/tmp/action",
            ],
        ):
            with self.subTest(args=args):
                with self.assertRaises(SystemExit):
                    parse_train_options(args)


class _WandbRun:
    def log(self, _metrics, step):
        self.step = step


class _WandbModule(types.ModuleType):
    def __init__(self):
        super().__init__("wandb")
        self.run = _WandbRun()

    def init(self, **_kwargs):
        return self.run


class _TensorReducingAccelerator:
    is_main_process = True

    def __init__(self):
        self.reduced_values = []

    def reduce(self, value, reduction):
        self.reduced_values.append((value, reduction))
        return value


class TrainWandbIntegrationTest(unittest.TestCase):
    def test_output_metrics_remain_tensors_for_real_wandb_logger_reduction(self):
        metrics = train_vla.loss_output_metrics(
            BatchFeature({"loss": torch.tensor(3.0), "vlm_loss_weight": 1.0})
        )
        accelerator = _TensorReducingAccelerator()
        fake_wandb = _WandbModule()
        with patch.dict(sys.modules, {"wandb": fake_wandb}):
            logger = WandbTrainingLogger(
                accelerator,
                project="test-project",
                run_name="test-run",
                run_id=None,
                resume="never",
                log_dir="/tmp/wandb",
            )
            logger.log(
                step=1,
                mean_metrics={f"train/{key}": value for key, value in metrics.items()},
                scalar_metrics={},
            )
        self.assertTrue(accelerator.reduced_values)
        self.assertTrue(
            all(isinstance(value, torch.Tensor) for value, _ in accelerator.reduced_values)
        )


if __name__ == "__main__":
    unittest.main()
