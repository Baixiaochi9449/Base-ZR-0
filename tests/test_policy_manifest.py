import json
import tempfile
import unittest
import warnings
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from utils.dataset_manifest import (
    build_resolved_dataset_manifest,
    write_resolved_dataset_manifest,
)
from utils.dataset_spec import resolve_dataset_spec


class _FakeModel:
    action_expert_config = SimpleNamespace(
        action_horizon=32,
        action_dim=64,
        state_dim=64,
    )

    def to(self, *args, **kwargs):
        return self

    def eval(self):
        return self


class PolicyManifestTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        root = Path(self.temporary_directory.name) / "dataset"
        meta = root / "meta"
        meta.mkdir(parents=True)
        info = {
            "codebase_version": "v3.0-test",
            "features": {
                "first": {"dtype": "image", "shape": [4, 4, 3]},
                "second": {"dtype": "image", "shape": [4, 4, 3]},
                "wrist": {"dtype": "image", "shape": [4, 4, 3]},
                "state": {"dtype": "float32", "shape": [7]},
                "actions": {"dtype": "float32", "shape": [7]},
                "train_data": {"dtype": "string", "shape": [1]},
            },
        }
        (meta / "info.json").write_text(json.dumps(info), encoding="utf-8")
        self.stats_path = meta / "stats_gr00t.json"
        self._write_stats(1.0)
        self.root = root
        self.entry = {
            "dataset_path": str(root),
            "dataset_type": "vla",
            "dataset_adapter": "lerobot_v3_future_difference",
            "target_text_field": "train_data",
            "camera_keys": ["first", "second", "wrist"],
            "state_field": "state",
            "action_field": "actions",
            "stats_path": "meta/stats_gr00t.json",
            "use_quantile": True,
            "sample_ratio": 1.0,
        }
        self.checkpoint = Path(self.temporary_directory.name) / "checkpoint"

    def tearDown(self):
        self.temporary_directory.cleanup()

    def _write_stats(self, upper):
        self.stats_path.write_text(
            json.dumps(
                {
                    "statistics": {
                        "state": {"q01": [0.0] * 7, "q99": [upper] * 7},
                        "actions": {"q01": [-upper] * 7, "q99": [upper] * 7},
                    }
                }
            ),
            encoding="utf-8",
        )

    def _manifest(self, *extra_specs):
        spec = resolve_dataset_spec(
            "future", self.entry, action_horizon=32, require_action=True
        )
        manifest = build_resolved_dataset_manifest(
            [*extra_specs, spec], "vlm_and_action"
        )
        write_resolved_dataset_manifest(self.checkpoint, manifest)
        return spec

    def _construct(self, **kwargs):
        import policies.reasoning_vla_policy as policy_module

        load_calls = []

        def load_model(*args, **load_kwargs):
            load_calls.append((args, load_kwargs))
            return _FakeModel()

        with (
            patch.dict(policy_module.DATASET2FEATURE, {"future": self.entry}, clear=True),
            patch.object(policy_module.AutoProcessor, "from_pretrained", return_value=object()),
            patch.object(policy_module.ZR0Model, "from_pretrained", side_effect=load_model),
            patch.object(torch, "compile", side_effect=lambda model, **_: model),
        ):
            policy = policy_module.ZR0Policy(
                "future",
                str(self.checkpoint),
                "direct_action",
                window_size=1,
                device="cpu",
                **kwargs,
            )
        policy._test_model_load_calls = load_calls
        return policy

    def test_matching_checkpoint_manifest_constructs_policy(self):
        expected = self._manifest()
        policy = self._construct()
        self.assertEqual(policy.dataset_spec.stats_sha256, expected.stats_sha256)
        self.assertTrue(policy._test_model_load_calls[0][1]["for_action_inference"])

    def test_camera_order_and_stats_mutation_fail_policy_validation(self):
        self._manifest()
        reversed_entry = dict(self.entry)
        reversed_entry["camera_keys"] = ["second", "first", "wrist"]
        self.entry = reversed_entry
        with self.assertRaisesRegex(ValueError, "camera_keys"):
            self._construct()

        self.entry = dict(self.entry)
        self.entry["camera_keys"] = ["first", "second", "wrist"]
        self._write_stats(2.0)
        with self.assertRaisesRegex(ValueError, "stats_sha256|state_q99|action_q"):
            self._construct()

    def test_mixed_manifest_selects_unique_entry_and_rejects_missing_or_ambiguous(self):
        future = self._manifest()
        legacy = replace(future, dataset_entry="legacy", adapter="lerobot_v2")
        write_resolved_dataset_manifest(
            self.checkpoint,
            build_resolved_dataset_manifest([legacy, future], "vlm_and_action"),
        )
        self.assertEqual(self._construct().dataset_spec.dataset_entry, "future")

        missing = replace(future, dataset_entry="other")
        write_resolved_dataset_manifest(
            self.checkpoint,
            build_resolved_dataset_manifest([missing], "vlm_and_action"),
        )
        with self.assertRaisesRegex(ValueError, "future.*not found"):
            self._construct()

        manifest = build_resolved_dataset_manifest([future, future], "vlm_and_action")
        write_resolved_dataset_manifest(self.checkpoint, manifest)
        with self.assertRaisesRegex(ValueError, "future.*multiple"):
            self._construct()

    def test_legacy_checkpoint_requires_explicit_opt_in(self):
        self.checkpoint.mkdir()
        with self.assertRaisesRegex(ValueError, "manifest.*missing"):
            self._construct()

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            policy = self._construct(
                allow_legacy_checkpoint_without_manifest=True
            )
        self.assertEqual(policy.dataset_spec.dataset_entry, "future")
        self.assertTrue(any("legacy" in str(item.message).lower() for item in caught))


if __name__ == "__main__":
    unittest.main()
