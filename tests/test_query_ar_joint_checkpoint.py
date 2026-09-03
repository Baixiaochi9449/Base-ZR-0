import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from safetensors.torch import load_file, save_file
from torch import nn

from model.difference_query import DifferenceQuery
from model.flow_matching_action_head import FlowmatchingActionHeadConfig
from model.reasoning_vla_model import ZR0Model
from utils.training_checkpoint import (
    CHECKPOINT_TAG,
    checkpoint_model_optimizer_scheduler,
    resume_model_optimizer_scheduler,
)


class _TinySaveableVlm(nn.Module):
    filename = "tiny_vlm.safetensors"

    def __init__(self, source):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(2, 3))
        self.config = SimpleNamespace(
            text_config=SimpleNamespace(hidden_size=3, initializer_range=0.02)
        )
        checkpoint = Path(source) / self.filename
        if checkpoint.is_file():
            self.load_state_dict(load_file(checkpoint))

    def save_pretrained(self, directory):
        save_file(self.state_dict(), Path(directory) / self.filename)


class _TinyProcessor:
    def save_pretrained(self, directory):
        Path(directory, "processor_config.json").write_text("{}\n", encoding="utf-8")


class _TinyBackbone(nn.Module):
    def __init__(
        self,
        model_name,
        _tune_vlm,
        _lora_args,
        *,
        resolved_difference_query_config,
    ):
        super().__init__()
        self.model = _TinySaveableVlm(model_name)
        self.processor = _TinyProcessor()
        self.use_difference_query = resolved_difference_query_config.enabled
        self.num_difference_queries = resolved_difference_query_config.num_difference_queries
        self.difference_query = None
        if self.use_difference_query:
            self.difference_query = DifferenceQuery(
                self.num_difference_queries,
                hidden_size=3,
                initializer_std=0.02,
            )
            checkpoint_tensor = resolved_difference_query_config.checkpoint_tensor
            if checkpoint_tensor is not None:
                with torch.no_grad():
                    self.difference_query.weight.copy_(checkpoint_tensor)


class _TinyActionExpert(nn.Module):
    def __init__(self, _config, _tune_action_expert):
        super().__init__()
        self.projection = nn.Linear(3, 2)


class QueryArWarmStartCheckpointTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.patches = (
            patch("model.reasoning_vla_model.QwenVLBackbone", _TinyBackbone),
            patch("model.reasoning_vla_model.FlowmatchingActionHead", _TinyActionExpert),
        )
        for active_patch in self.patches:
            active_patch.start()

    def tearDown(self):
        for active_patch in reversed(self.patches):
            active_patch.stop()
        self.temporary_directory.cleanup()

    @staticmethod
    def action_config():
        return FlowmatchingActionHeadConfig(
            action_dim=2,
            state_dim=2,
            action_horizon=3,
        )

    def model(
        self,
        vlm_path,
        *,
        action_path=None,
        explicit_query=False,
        loss_type="vlm_and_action",
    ):
        return ZR0Model(
            vlm_name_or_path=str(vlm_path),
            action_expert_name_or_path=(
                str(action_path) if action_path is not None else None
            ),
            action_expert_config=self.action_config(),
            tune_vlm=True,
            tune_action_expert=loss_type != "vlm",
            loss_type=loss_type,
            use_difference_query=True if explicit_query else None,
            num_difference_queries=4 if explicit_query else None,
            vlm_attention_backend="sdpa" if explicit_query else None,
        )

    def test_ar_checkpoint_restores_vlm_and_query_exactly(self):
        source = self.model(
            self.root / "base", explicit_query=True, loss_type="vlm"
        )
        with torch.no_grad():
            source.backbone.model.weight.copy_(
                torch.arange(6, dtype=torch.float32).reshape(2, 3)
            )
            source.backbone.difference_query.weight.copy_(
                torch.arange(12, dtype=torch.float32).reshape(4, 3) / 10
            )
        checkpoint = self.root / "ar-checkpoint"
        source.resolved_dataset_manifest = {
            "format_version": 1,
            "loss_type": "vlm",
            "entries": [{"dataset_entry": "fixture", "action_horizon": 3}],
        }
        source.save_pretrained(checkpoint)

        self.assertIsNone(source.action_expert)
        self.assertFalse((checkpoint / "action_expert.safetensors").exists())
        self.assertTrue((checkpoint / "action_expert_config.json").is_file())
        from utils.dataset_manifest import load_resolved_dataset_manifest

        self.assertEqual(
            load_resolved_dataset_manifest(checkpoint)["entries"][0]["dataset_entry"],
            "fixture",
        )
        metadata = json.loads(
            (checkpoint / "zr0_checkpoint_metadata.json").read_text(encoding="utf-8")
        )
        self.assertEqual(metadata["checkpoint_kind"], "ar_only")
        restored = ZR0Model.from_pretrained(
            checkpoint,
            tune_vlm=True,
            tune_action_expert=False,
            loss_type="vlm",
        )

        for name, expected in source.backbone.model.state_dict().items():
            torch.testing.assert_close(restored.backbone.model.state_dict()[name], expected)
        torch.testing.assert_close(
            restored.backbone.difference_query.weight,
            source.backbone.difference_query.weight,
            rtol=0,
            atol=0,
        )

    def test_ar_training_state_resume_has_no_action_expert_payload(self):
        torch.manual_seed(73)
        source = self.model(
            self.root / "base", explicit_query=True, loss_type="vlm"
        )
        optimizer = torch.optim.AdamW(source.parameters(), lr=2e-4)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda step: 1.0 / (step + 1)
        )
        loss = sum(parameter.square().mean() for parameter in source.parameters())
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        expected_model = copy.deepcopy(source.state_dict())
        expected_optimizer = copy.deepcopy(optimizer.state_dict())
        expected_scheduler = copy.deepcopy(scheduler.state_dict())

        output = self.root / "ar-run"
        checkpoint_model_optimizer_scheduler(
            _TinyEngine(source, optimizer),
            output,
            global_completed_steps=1,
            lr_scheduler=scheduler,
            accelerator=_TinyAccelerator(),
        )
        checkpoint = output / CHECKPOINT_TAG
        self.assertFalse((checkpoint / "action_expert.safetensors").exists())

        restored = self.model(checkpoint, loss_type="vlm")
        restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=2e-4)
        restored_scheduler = torch.optim.lr_scheduler.LambdaLR(
            restored_optimizer, lr_lambda=lambda step: 1.0 / (step + 1)
        )
        with patch(
            "utils.training_checkpoint.register_deepspeed_checkpoint_safe_globals"
        ):
            restored_step = resume_model_optimizer_scheduler(
                _TinyEngine(restored, restored_optimizer), output, restored_scheduler
            )

        self.assertEqual(restored_step, 1)
        FullTrainingStateCheckpointTest.assert_nested_equal(
            restored.state_dict(), expected_model
        )
        FullTrainingStateCheckpointTest.assert_nested_equal(
            restored_optimizer.state_dict(), expected_optimizer
        )
        FullTrainingStateCheckpointTest.assert_nested_equal(
            restored_scheduler.state_dict(), expected_scheduler
        )

    def test_joint_warm_start_leaves_action_expert_at_same_seed_fresh_init(self):
        ar_model = self.model(
            self.root / "base", explicit_query=True, loss_type="vlm"
        )
        checkpoint = self.root / "ar-checkpoint"
        ar_model.save_pretrained(checkpoint)

        torch.manual_seed(917)
        expected_fresh = self.model(checkpoint, action_path=None)
        torch.manual_seed(917)
        joint = ZR0Model.from_pretrained(
            checkpoint,
            tune_vlm=True,
            tune_action_expert=True,
            loss_type="vlm_and_action",
            allow_ar_warm_start=True,
        )

        for name, expected in expected_fresh.action_expert.state_dict().items():
            torch.testing.assert_close(
                joint.action_expert.state_dict()[name], expected, rtol=0, atol=0
            )
        torch.testing.assert_close(
            joint.backbone.difference_query.weight,
            ar_model.backbone.difference_query.weight,
            rtol=0,
            atol=0,
        )
        with self.assertRaisesRegex(ValueError, "allow_ar_warm_start"):
            ZR0Model.from_pretrained(
                checkpoint,
                tune_vlm=False,
                tune_action_expert=False,
                loss_type="vlm_and_action",
            )

    def test_joint_checkpoint_kind_restores_action_and_rejects_ar_resume(self):
        source = self.model(self.root / "base", explicit_query=True)
        with torch.no_grad():
            for parameter in source.action_expert.parameters():
                parameter.fill_(0.375)
        checkpoint = self.root / "joint-checkpoint"
        source.save_pretrained(checkpoint)
        self.assertTrue((checkpoint / "action_expert.safetensors").is_file())
        metadata = json.loads(
            (checkpoint / "zr0_checkpoint_metadata.json").read_text(encoding="utf-8")
        )
        self.assertEqual(metadata["checkpoint_kind"], "joint")

        restored = ZR0Model.from_pretrained(
            checkpoint,
            tune_vlm=True,
            tune_action_expert=True,
            loss_type="vlm_and_action",
        )
        with self.assertRaisesRegex(
            ValueError,
            "constructed mode='vlm_and_action'.*requested mode='action'",
        ):
            restored({}, training_progress=0.0, loss_type="action")
        for name, expected in source.action_expert.state_dict().items():
            torch.testing.assert_close(
                restored.action_expert.state_dict()[name], expected, rtol=0, atol=0
            )
        with self.assertRaisesRegex(ValueError, "checkpoint kind.*joint.*vlm"):
            ZR0Model.from_pretrained(
                checkpoint,
                tune_vlm=True,
                tune_action_expert=False,
                loss_type="vlm",
            )

        with self.assertRaisesRegex(
            ValueError, "joint.*requires the same checkpoint.*Action Expert"
        ):
            self.model(checkpoint, action_path=None, loss_type="vlm_and_action")

    def test_action_inference_loads_action_only_and_joint_but_rejects_ar_only(self):
        checkpoints = {}
        for loss_type in ("action", "vlm_and_action", "vlm"):
            source = self.model(
                self.root / f"base-{loss_type}",
                explicit_query=True,
                loss_type=loss_type,
            )
            checkpoint = self.root / f"{loss_type}-checkpoint"
            source.save_pretrained(checkpoint)
            checkpoints[loss_type] = (source, checkpoint)

        for loss_type in ("action", "vlm_and_action"):
            source, checkpoint = checkpoints[loss_type]
            restored = ZR0Model.from_pretrained(
                checkpoint,
                for_action_inference=True,
            )
            self.assertEqual(restored.loss_type, "action")
            self.assertIsNotNone(restored.action_expert)
            for name, expected in source.action_expert.state_dict().items():
                torch.testing.assert_close(
                    restored.action_expert.state_dict()[name], expected, rtol=0, atol=0
                )

        with self.assertRaisesRegex(ValueError, "ar_only.*action inference"):
            ZR0Model.from_pretrained(
                checkpoints["vlm"][1],
                for_action_inference=True,
            )

    def test_policy_loads_action_only_and_joint_and_rejects_ar_only(self):
        import policies.reasoning_vla_policy as policy_module
        from utils.dataset_manifest import (
            build_resolved_dataset_manifest,
        )
        from utils.dataset_spec import resolve_dataset_spec

        dataset_root = self.root / "dataset"
        entry = {
            "dataset_path": str(dataset_root),
            "dataset_type": "vla",
            "dataset_adapter": "lerobot_v2",
            "target_text_field": "precomputed_target",
            "sample_ratio": 1.0,
            "use_quantile": True,
        }
        metadata = SimpleNamespace(
            root=dataset_root,
            camera_keys=["front"],
            grounding_camera_keys=None,
            features={
                "observation.state": {"shape": [2]},
                "action": {"shape": [2]},
                "precomputed_target": {"shape": [1]},
            },
            stats={
                "observation.state": {"q01": [0.0, 0.0], "q99": [1.0, 1.0]},
                "action": {"q01": [0.0, 0.0], "q99": [1.0, 1.0]},
            },
            codebase_version="v2-test",
        )
        spec = resolve_dataset_spec(
            "fixture",
            entry,
            action_horizon=3,
            require_action=True,
            v2_metadata=metadata,
        )
        checkpoints = {}
        for loss_type in ("action", "vlm_and_action", "vlm"):
            source = self.model(
                self.root / f"policy-base-{loss_type}",
                explicit_query=True,
                loss_type=loss_type,
            )
            source.resolved_dataset_manifest = build_resolved_dataset_manifest(
                [spec], loss_type
            )
            checkpoint = self.root / f"policy-{loss_type}"
            source.save_pretrained(checkpoint)
            checkpoints[loss_type] = checkpoint

        with (
            patch.dict(policy_module.DATASET2FEATURE, {"fixture": entry}, clear=True),
            patch.object(
                policy_module.AutoProcessor, "from_pretrained", return_value=object()
            ),
            patch.object(
                policy_module,
                "LeRobotDatasetMetadata",
                return_value=metadata,
            ),
            patch.object(torch, "compile", side_effect=lambda model, **_: model),
        ):
            for loss_type in ("action", "vlm_and_action"):
                policy = policy_module.ZR0Policy(
                    "fixture",
                    str(checkpoints[loss_type]),
                    "direct_action",
                    window_size=1,
                    max_pad_state_and_action_length=2,
                    device="cpu",
                )
                self.assertIsNotNone(policy.model.action_expert)
            with self.assertRaisesRegex(ValueError, "ar_only.*action inference"):
                policy_module.ZR0Policy(
                    "fixture",
                    str(checkpoints["vlm"]),
                    "direct_action",
                    window_size=1,
                    max_pad_state_and_action_length=2,
                    device="cpu",
                )


class _TinyEngine:
    def __init__(self, module, optimizer):
        self.module = module
        self.optimizer = optimizer

    def save_checkpoint(self, output_dir, *, tag, client_state, save_latest):
        self.assert_false(save_latest)
        directory = Path(output_dir) / tag
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": self.module.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "client_state": client_state,
            },
            directory / "tiny_engine.pt",
        )

    @staticmethod
    def assert_false(value):
        if value:
            raise AssertionError("save_latest must remain disabled")

    def load_checkpoint(self, output_dir, *, tag):
        path = Path(output_dir) / tag
        state = torch.load(path / "tiny_engine.pt", map_location="cpu", weights_only=True)
        self.module.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        return str(path), state["client_state"]


class _TinyCheckpointModel(nn.Linear):
    def save_pretrained(self, directory):
        save_file(self.state_dict(), Path(directory) / "model.safetensors")


class _TinyAccelerator:
    is_main_process = True

    @staticmethod
    def print(*_args, **_kwargs):
        return None

    @staticmethod
    def save(value, path):
        torch.save(value, path)

    @staticmethod
    def unwrap_model(engine):
        return engine.module


class FullTrainingStateCheckpointTest(unittest.TestCase):
    @staticmethod
    def assert_nested_equal(actual, expected):
        if isinstance(expected, torch.Tensor):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        elif isinstance(expected, dict):
            if actual.keys() != expected.keys():
                raise AssertionError(f"keys differ: {actual.keys()} != {expected.keys()}")
            for key in expected:
                FullTrainingStateCheckpointTest.assert_nested_equal(
                    actual[key], expected[key]
                )
        elif isinstance(expected, list):
            if len(actual) != len(expected):
                raise AssertionError(f"lengths differ: {len(actual)} != {len(expected)}")
            for actual_value, expected_value in zip(actual, expected):
                FullTrainingStateCheckpointTest.assert_nested_equal(
                    actual_value, expected_value
                )
        else:
            if actual != expected:
                raise AssertionError(f"values differ: {actual!r} != {expected!r}")

    def test_joint_resume_restores_model_optimizer_scheduler_and_global_step(self):
        torch.manual_seed(31)
        model = _TinyCheckpointModel(3, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda step: 1.0 / (step + 1)
        )
        loss = model(torch.ones(2, 3)).square().mean()
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        expected_model = copy.deepcopy(model.state_dict())
        expected_optimizer = copy.deepcopy(optimizer.state_dict())
        expected_scheduler = copy.deepcopy(scheduler.state_dict())

        with tempfile.TemporaryDirectory() as temporary_directory:
            engine = _TinyEngine(model, optimizer)
            checkpoint_model_optimizer_scheduler(
                engine,
                temporary_directory,
                global_completed_steps=7,
                lr_scheduler=scheduler,
                accelerator=_TinyAccelerator(),
            )

            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.add_(10)
            optimizer.param_groups[0]["lr"] = 9.0
            scheduler.last_epoch = 99

            with patch(
                "utils.training_checkpoint.register_deepspeed_checkpoint_safe_globals"
            ):
                restored_step = resume_model_optimizer_scheduler(
                    engine, temporary_directory, scheduler
                )

            self.assertEqual(restored_step, 7)
            self.assertTrue(
                (Path(temporary_directory) / CHECKPOINT_TAG / "model.safetensors").is_file()
            )
            self.assert_nested_equal(model.state_dict(), expected_model)
            self.assert_nested_equal(optimizer.state_dict(), expected_optimizer)
            self.assert_nested_equal(scheduler.state_dict(), expected_scheduler)


if __name__ == "__main__":
    unittest.main()
