import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.difference_query import (
    DIFFERENCE_QUERY_CONFIG_NAME,
    DIFFERENCE_QUERY_WEIGHT_NAME,
    resolve_difference_query_config,
    save_difference_query_artifacts,
)


class DifferenceQueryConfigTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def write_checkpoint(
        self,
        name: str,
        *,
        enabled: bool = True,
        num_queries: int = 8,
        hidden_size: int = 16,
        weights: torch.Tensor | None = None,
        write_config: bool = True,
        write_weights: bool | None = None,
    ) -> Path:
        checkpoint = self.root / name
        checkpoint.mkdir()
        if write_config:
            config = {
                "version": 1,
                "enabled": enabled,
                "num_difference_queries": num_queries if enabled else None,
                "hidden_size": hidden_size,
                "attention_backend": "sdpa" if enabled else None,
            }
            (checkpoint / DIFFERENCE_QUERY_CONFIG_NAME).write_text(
                json.dumps(config), encoding="utf-8"
            )
        if write_weights is None:
            write_weights = enabled
        if write_weights:
            tensor = weights if weights is not None else torch.arange(
                num_queries * hidden_size, dtype=torch.float32
            ).reshape(num_queries, hidden_size)
            save_file({"difference_query": tensor}, checkpoint / DIFFERENCE_QUERY_WEIGHT_NAME)
        return checkpoint

    def test_no_checkpoint_defaults_to_disabled_and_preserves_backend_auto_selection(self):
        resolved = resolve_difference_query_config(None, None)

        self.assertFalse(resolved.enabled)
        self.assertIsNone(resolved.num_difference_queries)
        self.assertIsNone(resolved.attention_backend)
        self.assertIsNone(resolved.checkpoint_tensor)

    def test_explicit_enable_uses_32_only_when_query_checkpoint_is_absent(self):
        resolved = resolve_difference_query_config(
            self.root / "missing-vlm",
            self.root / "missing-action",
            use_difference_query=True,
        )

        self.assertTrue(resolved.enabled)
        self.assertEqual(resolved.num_difference_queries, 32)
        self.assertEqual(resolved.attention_backend, "sdpa")
        self.assertTrue(resolved.random_initialization)

    def test_random_initialization_requires_explicit_enable(self):
        with self.assertRaisesRegex(ValueError, "use_difference_query=True"):
            resolve_difference_query_config(
                None,
                None,
                num_difference_queries=8,
            )

    def test_disabled_checkpoint_is_an_explicit_architecture_declaration(self):
        checkpoint = self.write_checkpoint(
            "disabled-declaration", enabled=False, hidden_size=16
        )

        resolved = resolve_difference_query_config(checkpoint, None)
        self.assertFalse(resolved.enabled)
        self.assertEqual(resolved.expected_hidden_size, 16)
        self.assertEqual(resolved.checkpoint_directories, (checkpoint.resolve(),))

        with self.assertRaisesRegex(ValueError, "explicitly enables"):
            resolve_difference_query_config(
                checkpoint,
                None,
                use_difference_query=True,
            )

    def test_disabled_declarations_in_both_loading_directories_block_enable(self):
        vlm_checkpoint = self.write_checkpoint(
            "disabled-vlm", enabled=False, hidden_size=16
        )
        action_checkpoint = self.write_checkpoint(
            "disabled-action", enabled=False, hidden_size=16
        )

        with self.assertRaisesRegex(ValueError, "explicitly enables"):
            resolve_difference_query_config(
                vlm_checkpoint,
                action_checkpoint,
                use_difference_query=True,
            )

    def test_legacy_checkpoint_and_explicit_false_remain_disabled(self):
        legacy_checkpoint = self.root / "legacy"
        legacy_checkpoint.mkdir()

        resolved = resolve_difference_query_config(
            legacy_checkpoint,
            None,
            use_difference_query=False,
        )

        self.assertFalse(resolved.enabled)
        self.assertIsNone(resolved.expected_hidden_size)
        self.assertFalse(resolved.random_initialization)

    def test_enabled_query_and_legacy_directories_cannot_be_mixed(self):
        query_checkpoint = self.write_checkpoint("enabled-query")
        legacy_checkpoint = self.root / "enabled-legacy"
        legacy_checkpoint.mkdir()

        for first, second in (
            (query_checkpoint, legacy_checkpoint),
            (legacy_checkpoint, query_checkpoint),
        ):
            with self.subTest(first=first.name, second=second.name):
                with self.assertRaises(ValueError) as context:
                    resolve_difference_query_config(first, second)
                message = str(context.exception)
                self.assertIn(
                    "cannot mix Difference Query checkpoint with legacy checkpoint",
                    message,
                )
                self.assertIn(str(query_checkpoint.resolve()), message)
                self.assertIn(str(legacy_checkpoint.resolve()), message)
                self.assertIn("missing Difference Query declaration", message)

    def test_disabled_query_and_legacy_directories_cannot_be_mixed(self):
        disabled_checkpoint = self.write_checkpoint(
            "disabled-query", enabled=False
        )
        legacy_checkpoint = self.root / "disabled-legacy"
        legacy_checkpoint.mkdir()

        for first, second in (
            (disabled_checkpoint, legacy_checkpoint),
            (legacy_checkpoint, disabled_checkpoint),
        ):
            with self.subTest(first=first.name, second=second.name):
                with self.assertRaises(ValueError) as context:
                    resolve_difference_query_config(first, second)
                message = str(context.exception)
                self.assertIn(
                    "cannot mix Difference Query checkpoint with legacy checkpoint",
                    message,
                )
                self.assertIn(str(disabled_checkpoint.resolve()), message)
                self.assertIn(str(legacy_checkpoint.resolve()), message)
                self.assertIn("missing Difference Query declaration", message)

    def test_two_legacy_directories_remain_allowed(self):
        first = self.root / "legacy-first"
        second = self.root / "legacy-second"
        first.mkdir()
        second.mkdir()

        resolved = resolve_difference_query_config(first, second)

        self.assertFalse(resolved.enabled)
        self.assertEqual(resolved.checkpoint_directories, ())

    def test_two_consistent_query_declarations_remain_allowed(self):
        enabled_first = self.write_checkpoint("enabled-first")
        enabled_second = self.write_checkpoint("enabled-second")
        enabled = resolve_difference_query_config(enabled_first, enabled_second)

        self.assertTrue(enabled.enabled)
        self.assertEqual(
            enabled.checkpoint_directories,
            (enabled_first.resolve(), enabled_second.resolve()),
        )

        disabled_first = self.write_checkpoint(
            "consistent-disabled-first", enabled=False
        )
        disabled_second = self.write_checkpoint(
            "consistent-disabled-second", enabled=False
        )
        disabled = resolve_difference_query_config(disabled_first, disabled_second)

        self.assertFalse(disabled.enabled)
        self.assertEqual(
            disabled.checkpoint_directories,
            (disabled_first.resolve(), disabled_second.resolve()),
        )

    def test_same_directory_and_missing_action_path_do_not_trigger_mixed_error(self):
        enabled = self.write_checkpoint("deduplicated-enabled")
        same_directory = resolve_difference_query_config(enabled, enabled)
        no_action_path = resolve_difference_query_config(enabled, None)

        self.assertEqual(same_directory.checkpoint_directories, (enabled.resolve(),))
        self.assertEqual(no_action_path.checkpoint_directories, (enabled.resolve(),))

    def test_checkpoint_auto_enables_and_explicit_values_must_not_conflict(self):
        checkpoint = self.write_checkpoint("query")

        resolved = resolve_difference_query_config(checkpoint, checkpoint)

        self.assertTrue(resolved.enabled)
        self.assertEqual(resolved.num_difference_queries, 8)
        self.assertEqual(resolved.expected_hidden_size, 16)
        torch.testing.assert_close(
            resolved.checkpoint_tensor,
            torch.arange(128, dtype=torch.float32).reshape(8, 16),
        )
        self.assertFalse(resolved.random_initialization)

        with self.assertRaisesRegex(ValueError, "explicitly disabled"):
            resolve_difference_query_config(
                checkpoint,
                None,
                use_difference_query=False,
            )
        with self.assertRaisesRegex(ValueError, "num_difference_queries"):
            resolve_difference_query_config(
                checkpoint,
                None,
                num_difference_queries=32,
            )
        with self.assertRaisesRegex(ValueError, "SDPA"):
            resolve_difference_query_config(
                checkpoint,
                None,
                vlm_attention_backend="eager",
            )

    def test_backend_control_allows_sdpa_without_query(self):
        resolved = resolve_difference_query_config(
            None,
            None,
            vlm_attention_backend="sdpa",
        )

        self.assertFalse(resolved.enabled)
        self.assertEqual(resolved.attention_backend, "sdpa")

    def test_checkpoint_artifacts_must_be_complete_and_consistent(self):
        config_only = self.write_checkpoint("config-only", write_weights=False)
        weights_only = self.write_checkpoint("weights-only", write_config=False)
        disabled_with_weights = self.write_checkpoint(
            "disabled-with-weights", enabled=False, write_weights=True
        )

        for checkpoint, message in (
            (config_only, "missing"),
            (weights_only, "config"),
            (disabled_with_weights, "disabled"),
        ):
            with self.subTest(checkpoint=checkpoint.name):
                with self.assertRaisesRegex(ValueError, message):
                    resolve_difference_query_config(checkpoint, None)

    def test_two_query_directories_must_have_identical_config_and_weights(self):
        first = self.write_checkpoint("first")
        second = self.write_checkpoint("second")

        resolve_difference_query_config(first, second)

        differing = torch.zeros(8, 16)
        save_file(
            {"difference_query": differing},
            second / DIFFERENCE_QUERY_WEIGHT_NAME,
        )
        with self.assertRaisesRegex(ValueError, "different Difference Query weights"):
            resolve_difference_query_config(first, second)

    def test_two_disabled_directories_must_have_identical_configs(self):
        first = self.write_checkpoint(
            "disabled-first", enabled=False, hidden_size=16
        )
        second = self.write_checkpoint(
            "disabled-second", enabled=False, hidden_size=32
        )

        with self.assertRaisesRegex(ValueError, "different Difference Query configs"):
            resolve_difference_query_config(first, second)

    def test_enabled_and_disabled_loading_directories_always_conflict(self):
        enabled = self.write_checkpoint("enabled-vlm")
        disabled = self.write_checkpoint(
            "disabled-action", enabled=False, hidden_size=16
        )

        for explicit_value in (None, True, False):
            with self.subTest(use_difference_query=explicit_value):
                with self.assertRaisesRegex(ValueError, "enabled state conflicts"):
                    resolve_difference_query_config(
                        enabled,
                        disabled,
                        use_difference_query=explicit_value,
                    )

    def test_non_finite_weights_are_rejected_on_load_and_save(self):
        checkpoint = self.write_checkpoint(
            "non-finite", weights=torch.full((8, 16), torch.nan)
        )
        with self.assertRaisesRegex(ValueError, "finite"):
            resolve_difference_query_config(checkpoint, None)

        with self.assertRaisesRegex(ValueError, "finite"):
            save_difference_query_artifacts(
                self.root / "non-finite-save",
                enabled=True,
                hidden_size=16,
                difference_query=torch.full((8, 16), torch.inf),
            )

    def test_shape_and_positive_query_count_are_validated(self):
        bad_shape = self.write_checkpoint(
            "bad-shape", weights=torch.zeros(7, 16)
        )
        with self.assertRaisesRegex(ValueError, "shape"):
            resolve_difference_query_config(bad_shape, None)

        with self.assertRaisesRegex(ValueError, "positive"):
            resolve_difference_query_config(
                None,
                None,
                use_difference_query=True,
                num_difference_queries=0,
            )

    def test_save_artifacts_round_trip_enabled_and_disabled_configs(self):
        enabled_dir = self.root / "saved-enabled"
        weight = torch.randn(8, 16, dtype=torch.bfloat16)
        save_difference_query_artifacts(
            enabled_dir,
            enabled=True,
            hidden_size=16,
            difference_query=weight,
        )

        resolved = resolve_difference_query_config(enabled_dir, None)
        self.assertTrue(resolved.enabled)
        self.assertEqual(resolved.checkpoint_tensor.dtype, torch.float32)
        torch.testing.assert_close(resolved.checkpoint_tensor, weight.float())

        disabled_dir = self.root / "saved-disabled"
        disabled_dir.mkdir()
        save_file(
            {"difference_query": torch.zeros(1, 16)},
            disabled_dir / DIFFERENCE_QUERY_WEIGHT_NAME,
        )
        save_difference_query_artifacts(
            disabled_dir,
            enabled=False,
            hidden_size=16,
            difference_query=None,
        )

        config = json.loads(
            (disabled_dir / DIFFERENCE_QUERY_CONFIG_NAME).read_text(encoding="utf-8")
        )
        self.assertEqual(
            config,
            {
                "version": 1,
                "enabled": False,
                "num_difference_queries": None,
                "hidden_size": 16,
                "attention_backend": None,
            },
        )
        self.assertFalse((disabled_dir / DIFFERENCE_QUERY_WEIGHT_NAME).exists())


if __name__ == "__main__":
    unittest.main()
