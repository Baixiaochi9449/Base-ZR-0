import os
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(sys.executable)


class DifferenceQueryCliTest(unittest.TestCase):
    @staticmethod
    def isolated_environment() -> dict[str, str]:
        env = os.environ.copy()
        env["PYTHONNOUSERSITE"] = "1"
        env["PYTHONPATH"] = str(ROOT)
        return env

    def run_help(self, script: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [str(PYTHON), str(ROOT / script), "--help"],
            cwd=ROOT,
            env=self.isolated_environment(),
            text=True,
            capture_output=True,
            check=False,
        )

    def test_train_and_server_expose_tristate_query_and_backend_options(self):
        for script in ("train_vla.py", "server.py"):
            with self.subTest(script=script):
                result = self.run_help(script)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("--use_difference_query", result.stdout)
                self.assertIn("--num_difference_queries", result.stdout)
                self.assertIn("--vlm_attention_backend", result.stdout)

    def test_train_help_does_not_claim_max_steps_leave_scheduler_unchanged(self):
        result = self.run_help("train_vla.py")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("scheduler horizon remains unchanged", result.stdout)
        self.assertIn("--dataloader_num_workers", result.stdout)

    def test_gradient_accumulation_defaults_to_one_and_accepts_32(self):
        from utils.cli_options import parse_train_options

        self.assertEqual(parse_train_options([]).gradient_accumulation_steps, 1)
        options = parse_train_options(
            [
                "--gradient_accumulation_steps",
                "32",
                "--expected_global_batch_size",
                "128",
            ]
        )
        self.assertEqual(options.gradient_accumulation_steps, 32)
        self.assertEqual(options.expected_global_batch_size, 128)

    def test_gradient_accumulation_rejects_non_positive_values(self):
        from utils.cli_options import parse_train_options

        for value in ("0", "-1"):
            with self.subTest(value=value), self.assertRaises(SystemExit):
                parse_train_options(["--gradient_accumulation_steps", value])

    def test_train_and_server_parse_none_true_and_false(self):
        code = """
import json
from utils.cli_options import parse_server_options, parse_train_options

results = {}
for name, parser in (("train", parse_train_options), ("server", parse_server_options)):
    results[name] = [
        parser([]).use_difference_query,
        parser(["--use_difference_query"]).use_difference_query,
        parser(["--no-use_difference_query"]).use_difference_query,
        parser(["--num_difference_queries", "64"]).num_difference_queries,
    ]
print(json.dumps(results, sort_keys=True))
"""
        result = subprocess.run(
            [str(PYTHON), "-c", code],
            cwd=ROOT,
            env=self.isolated_environment(),
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            result.stdout.strip(),
            '{"server": [null, true, false, 64], "train": [null, true, false, 64]}',
        )

    def test_legacy_checkpoint_manifest_compatibility_requires_explicit_opt_in(self):
        from utils.cli_options import parse_server_options, parse_train_options

        for parser in (parse_train_options, parse_server_options):
            with self.subTest(parser=parser.__name__):
                self.assertFalse(
                    parser([]).allow_legacy_checkpoint_without_manifest
                )
                self.assertTrue(
                    parser(
                        ["--allow_legacy_checkpoint_without_manifest"]
                    ).allow_legacy_checkpoint_without_manifest
                )


if __name__ == "__main__":
    unittest.main()
