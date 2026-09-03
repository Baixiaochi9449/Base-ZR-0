import json
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path

import numpy as np

from utils.dataset_spec import ResolvedDatasetSpec


def _spec(entry, adapter, horizon=32):
    return ResolvedDatasetSpec(
        dataset_entry=entry,
        dataset_path=f"/datasets/{entry}",
        dataset_type="vla",
        adapter=adapter,
        target_text_field="train_data" if adapter.endswith("future_difference") else None,
        camera_keys=("front", "side", "wrist"),
        grounding_camera_keys=(),
        state_key="state" if adapter.endswith("future_difference") else "observation.state",
        action_key="actions" if adapter.endswith("future_difference") else "action",
        state_dim=7,
        action_dim=7,
        action_horizon=horizon,
        stats_path="meta/stats_gr00t.json" if adapter.endswith("future_difference") else None,
        stats_key="statistics.state/actions.q01/q99",
        normalization="quantile_min_max_q01_q99",
        normalization_stats={
            "observation.state": {"q01": np.zeros(7), "q99": np.ones(7)},
            "action": {"q01": np.zeros(7), "q99": np.ones(7)},
        },
        sample_ratio=1.0,
        training_eligibility_exists=adapter.endswith("future_difference"),
        training_eligibility_used=False,
        data_version="v3.0" if adapter.endswith("future_difference") else "v2",
    )


class DatasetManifestTest(unittest.TestCase):
    def test_single_v2_single_v3_and_mixed_are_entry_scoped(self):
        from utils.dataset_manifest import build_resolved_dataset_manifest

        v2 = _spec("legacy", "lerobot_v2")
        v3 = _spec("future", "lerobot_v3_future_difference")
        for specs, expected in (([v2], ["legacy"]), ([v3], ["future"]), ([v2, v3], ["legacy", "future"])):
            with self.subTest(expected=expected):
                manifest = build_resolved_dataset_manifest(specs, "vlm_and_action")
                self.assertEqual(
                    [entry["dataset_entry"] for entry in manifest["entries"]], expected
                )
                for entry in manifest["entries"]:
                    self.assertEqual(entry["action_horizon"], 32)
                    expected_requirements = [
                        "images", "task", "target_text", "state", "actions", "stats"
                    ]
                    if entry["resolved_adapter"] == "lerobot_v2":
                        expected_requirements.append("fast_tokenizer")
                    self.assertEqual(entry["loss_requirements"], expected_requirements)
                    self.assertIn("training_eligibility", entry)

    def test_serialization_and_content_hash_are_deterministic(self):
        from utils.dataset_manifest import (
            build_resolved_dataset_manifest,
            resolved_manifest_json,
        )

        manifest = build_resolved_dataset_manifest(
            [_spec("future", "lerobot_v3_future_difference")], "vlm"
        )
        first = resolved_manifest_json(manifest)
        second = resolved_manifest_json(json.loads(first))
        self.assertEqual(first, second)
        parsed = json.loads(first)
        self.assertEqual(len(parsed["content_hash"]), 64)

    def test_vision_input_contract_is_serialized_and_affects_hash(self):
        from utils.dataset_manifest import (
            build_resolved_dataset_manifest,
            resolved_manifest_hash,
        )

        contract = {
            "version": 1,
            "camera_order": ["front", "side", "wrist"],
            "total_visual_tokens": 147,
        }
        base = _spec("future", "lerobot_v3_future_difference")
        resolved = replace(base, vision_input_contract=contract)
        changed = replace(
            base,
            vision_input_contract={**contract, "total_visual_tokens": 150},
        )
        manifest = build_resolved_dataset_manifest([resolved], "vlm")
        self.assertEqual(
            manifest["entries"][0]["vision_input_contract"], contract
        )
        self.assertNotEqual(
            resolved_manifest_hash(manifest),
            resolved_manifest_hash(
                build_resolved_dataset_manifest([changed], "vlm")
            ),
        )

    def test_normalization_values_change_manifest_hash_at_same_stats_path(self):
        from utils.dataset_manifest import (
            build_resolved_dataset_manifest,
            resolved_manifest_hash,
        )

        base = _spec("future", "lerobot_v3_future_difference")
        changed_stats = {
            "observation.state": {
                "q01": np.full(7, -1.0),
                "q99": np.ones(7),
            },
            "action": {"q01": np.zeros(7), "q99": np.full(7, 2.0)},
        }
        changed = replace(base, normalization_stats=changed_stats)
        original_manifest = build_resolved_dataset_manifest([base], "vlm_and_action")
        changed_manifest = build_resolved_dataset_manifest([changed], "vlm_and_action")

        self.assertNotEqual(
            resolved_manifest_hash(original_manifest),
            resolved_manifest_hash(changed_manifest),
        )
        entry = original_manifest["entries"][0]
        self.assertEqual(entry["state_q01"], [0.0] * 7)
        self.assertEqual(entry["state_q99"], [1.0] * 7)
        self.assertEqual(entry["action_q01"], [0.0] * 7)
        self.assertEqual(entry["action_q99"], [1.0] * 7)
        self.assertEqual(len(entry["stats_sha256"]), 64)

    def test_resolved_stats_file_mutation_changes_identity_at_the_same_path(self):
        from utils.dataset_manifest import (
            build_resolved_dataset_manifest,
            resolved_manifest_hash,
        )
        from utils.dataset_spec import resolve_dataset_spec

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "meta").mkdir()
            features = {
                "first": {"dtype": "image"},
                "second": {"dtype": "image"},
                "wrist": {"dtype": "image"},
                "state": {"shape": [7]},
                "actions": {"shape": [7]},
                "train_data": {"dtype": "string"},
            }
            (root / "meta/info.json").write_text(
                json.dumps({"features": features, "codebase_version": "v3"}),
                encoding="utf-8",
            )
            stats_path = root / "meta/stats_gr00t.json"

            def resolve(q99):
                stats_path.write_text(
                    json.dumps(
                        {
                            "statistics": {
                                "state": {"q01": [0] * 7, "q99": [1] * 7},
                                "actions": {"q01": [0] * 7, "q99": [q99] * 7},
                            }
                        },
                        sort_keys=True,
                    ),
                    encoding="utf-8",
                )
                spec = resolve_dataset_spec(
                    "future",
                    {
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
                    },
                    action_horizon=32,
                    require_action=True,
                )
                return build_resolved_dataset_manifest([spec], "vlm_and_action")

            original = resolve(1)
            changed = resolve(2)
            self.assertNotEqual(
                original["entries"][0]["stats_sha256"],
                changed["entries"][0]["stats_sha256"],
            )
            self.assertNotEqual(
                resolved_manifest_hash(original), resolved_manifest_hash(changed)
            )

    def test_save_resume_and_semantic_conflicts(self):
        from utils.dataset_manifest import (
            build_resolved_dataset_manifest,
            validate_resume_manifest,
            write_resolved_dataset_manifest,
        )

        base = _spec("future", "lerobot_v3_future_difference")
        manifest = build_resolved_dataset_manifest([base], "vlm_and_action")
        with tempfile.TemporaryDirectory() as directory:
            path = write_resolved_dataset_manifest(directory, manifest)
            self.assertEqual(path.name, "resolved_dataset_manifest.json")
            validate_resume_manifest(manifest, directory)

            conflicts = (
                replace(base, target_text_field="other"),
                replace(base, camera_keys=("side", "front", "wrist")),
                replace(base, stats_path="meta/other.json"),
                replace(base, action_horizon=16),
            )
            for conflict in conflicts:
                with self.subTest(conflict=conflict):
                    current = build_resolved_dataset_manifest(
                        [conflict], "vlm_and_action"
                    )
                    with self.assertRaisesRegex(ValueError, "dataset manifest mismatch"):
                        validate_resume_manifest(current, directory)

            changed_stats = replace(
                base,
                normalization_stats={
                    "observation.state": {
                        "q01": np.zeros(7),
                        "q99": np.full(7, 3.0),
                    },
                    "action": {"q01": np.zeros(7), "q99": np.ones(7)},
                },
            )
            current = build_resolved_dataset_manifest(
                [changed_stats], "vlm_and_action"
            )
            with self.assertRaisesRegex(ValueError, "dataset manifest mismatch"):
                validate_resume_manifest(current, directory)

    def test_grounding_camera_members_and_order_are_semantic(self):
        from utils.dataset_manifest import (
            build_resolved_dataset_manifest,
            validate_policy_dataset_manifest,
            validate_resume_manifest,
            write_resolved_dataset_manifest,
        )

        base = replace(
            _spec("legacy", "lerobot_v2"),
            grounding_camera_keys=("front", "wrist"),
        )
        future = _spec("future", "lerobot_v3_future_difference")
        checkpoint_manifest = build_resolved_dataset_manifest(
            [future, base], "vlm_and_action"
        )
        with tempfile.TemporaryDirectory() as directory:
            write_resolved_dataset_manifest(directory, checkpoint_manifest)
            validate_policy_dataset_manifest(base, directory)

            for grounding_keys in (("front", "side"), ("wrist", "front")):
                changed = replace(base, grounding_camera_keys=grounding_keys)
                with self.subTest(grounding_keys=grounding_keys):
                    with self.assertRaisesRegex(
                        ValueError,
                        r"grounding_camera_keys.*checkpoint=.*current=.*legacy",
                    ):
                        validate_policy_dataset_manifest(changed, directory)
                    with self.assertRaisesRegex(
                        ValueError,
                        r"dataset manifest mismatch.*grounding_camera_keys.*"
                        r"checkpoint=.*current=.*legacy",
                    ):
                        validate_resume_manifest(
                            build_resolved_dataset_manifest(
                                [future, changed], "vlm_and_action"
                            ),
                            directory,
                        )

    def test_legacy_resume_manifest_requires_opt_in_and_prints_strong_warning(self):
        from utils.dataset_manifest import validate_resume_manifest

        manifest = {"format_version": 2, "loss_type": "vlm", "entries": [{}]}
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "manifest is missing"):
                validate_resume_manifest(manifest, directory)
            output = io.StringIO()
            with redirect_stdout(output):
                validate_resume_manifest(
                    manifest, directory, allow_legacy_missing=True
                )
            self.assertIn("[WARNING]", output.getvalue())


if __name__ == "__main__":
    unittest.main()
