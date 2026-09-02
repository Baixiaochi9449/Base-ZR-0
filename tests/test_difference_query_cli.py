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


if __name__ == "__main__":
    unittest.main()
