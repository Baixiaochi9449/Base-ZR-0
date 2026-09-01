import os
import json
import importlib.util
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "run_libero_wo_ecot_pt.sh"
PREFLIGHT = ROOT / "scripts" / "preflight_libero_wo_ecot_pt.py"


def load_preflight_module():
    spec = importlib.util.spec_from_file_location("libero_preflight", PREFLIGHT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class LiberoWoEcotPtLauncherTest(unittest.TestCase):
    def run_launcher(
        self,
        mode: str,
        *,
        provide_run_id: bool = True,
        extra_env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        env["ZR0_DRY_RUN"] = "1"
        env["CONDA_DEFAULT_ENV"] = "ZR-0"
        env["WANDB_API_KEY"] = "test-secret-that-must-not-be-printed"
        env["ZR0_RUN_NAME"] = "test-libero-run"
        if provide_run_id:
            env["ZR0_WANDB_RUN_ID"] = "test1234"
        else:
            env.pop("ZR0_WANDB_RUN_ID", None)
        if extra_env:
            env.update(extra_env)
        return subprocess.run(
            ["bash", str(LAUNCHER), mode],
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_train_uses_random_action_expert_and_paper_batch_size(self):
        result = self.run_launcher("train")

        self.assertEqual(result.returncode, 0, result.stderr)
        command = result.stdout
        self.assertIn("PYTHONNOUSERSITE=1", command)
        self.assertIn("--num_processes 4", command)
        self.assertIn("--per_device_train_batch_size 16", command)
        self.assertIn("--epochs 8", command)
        self.assertIn("--loss_type action", command)
        self.assertIn("--action_expert_loss_weight 1.0", command)
        self.assertIn("--action_horizon 10", command)
        self.assertIn("--peak_learning_rate 2e-5", command)
        self.assertIn("--min_lr_rate 0.1", command)
        self.assertIn("--dataset_entries libero_wo_ecot_pt", command)
        self.assertIn("--save_optimizer_and_lr_states", command)
        self.assertIn("--save_step_interval 2000", command)
        self.assertIn("--wandb_project ZR-0-LIBERO", command)
        self.assertIn("--wandb_run_name test-libero-run", command)
        self.assertIn("--wandb_run_id test1234", command)
        self.assertIn("--wandb_resume never", command)
        self.assertIn("--wandb_group libero-wo-ecot-pt", command)
        self.assertIn("test-libero-run", command)
        self.assertNotIn("test-secret-that-must-not-be-printed", command)
        self.assertNotIn("--action_expert_name_or_path", command)

    def test_resume_loads_model_action_expert_and_training_state(self):
        result = self.run_launcher("resume")

        self.assertEqual(result.returncode, 0, result.stderr)
        command = result.stdout
        checkpoint = (
            str(ROOT)
            + "/outputs/ckpts/"
            + "Qwen3-VL-2B-Instruct-LIBERO-wo-ECoT-PT-FinalRMSNorm/"
            + "latest-model-optimizer-lr"
        )
        self.assertIn(f"--vlm_name_or_path {checkpoint}", command)
        self.assertIn(f"--action_expert_name_or_path {checkpoint}", command)
        self.assertIn("--resume_training", command)
        self.assertIn("--save_optimizer_and_lr_states", command)
        self.assertIn("--wandb_run_id test1234", command)
        self.assertIn("--wandb_resume must", command)

    def test_smoke_overrides_add_max_steps_and_isolated_output(self):
        smoke_output = ROOT / "outputs" / "ckpts" / "gradient-fix-smoke-test"
        result = self.run_launcher(
            "train",
            extra_env={
                "ZR0_OUTPUT_DIR": str(smoke_output),
                "ZR0_MAX_TRAIN_STEPS": "200",
                "ZR0_SAVE_STEP_INTERVAL": "100",
            },
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--max_train_steps 200", result.stdout)
        self.assertIn("--save_step_interval 100", result.stdout)
        self.assertIn(f"--output_ckpt_dir {smoke_output}", result.stdout)

    def test_default_output_does_not_reuse_failed_run(self):
        result = self.run_launcher("train")

        self.assertEqual(result.returncode, 0, result.stderr)
        failed_output = (
            ROOT
            / "outputs"
            / "ckpts"
            / "Qwen3-VL-2B-Instruct-LIBERO-wo-ECoT-PT"
        )
        self.assertIn("FinalRMSNorm", result.stdout)
        self.assertNotIn(
            f"--output_ckpt_dir {failed_output} ",
            result.stdout,
        )

    def test_unknown_mode_fails(self):
        result = self.run_launcher("not-a-mode")

        self.assertEqual(result.returncode, 2)
        self.assertIn("Usage:", result.stderr)

    def test_train_generates_an_eight_character_run_id(self):
        result = self.run_launcher("train", provide_run_id=False)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertRegex(result.stdout, r"--wandb_run_id [a-z0-9]{8}(?: |\n)")

    def test_dataset_registry_points_to_downloaded_v21_dataset(self):
        registry = (ROOT / "dataset2feature.yaml").read_text(encoding="utf-8")
        expected = """libero_wo_ecot_pt:
  dataset_path: /opt/data/private/lq/datasets/HuggingFaceVLA/libero
  dataset_type: vla
  sample_ratio: 1.0
  use_quantile: true"""
        self.assertIn(expected, registry)

    def test_wandb_is_a_declared_dependency(self):
        requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")

        self.assertIn("wandb==0.29.0", requirements.splitlines())


class LiberoWoEcotPtPreflightTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        self.model_path = root / "model"
        self.fast_path = root / "fast"
        self.dataset_path = root / "libero"
        self.model_path.mkdir()
        self.fast_path.mkdir()
        (self.dataset_path / "meta").mkdir(parents=True)
        (self.dataset_path / "data" / "chunk-000").mkdir(parents=True)

        (self.model_path / "config.json").write_text(
            json.dumps({"model_type": "qwen3_vl"}), encoding="utf-8"
        )
        for filename in (
            "processor_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "processing_action_tokenizer.py",
        ):
            (self.fast_path / filename).write_text("{}", encoding="utf-8")

        info = {
            "total_episodes": 1693,
            "total_frames": 273465,
            "total_tasks": 40,
            "features": {
                "observation.images.image": {"dtype": "image", "shape": [256, 256, 3]},
                "observation.images.image2": {"dtype": "image", "shape": [256, 256, 3]},
                "observation.state": {"dtype": "float32", "shape": [8]},
                "action": {"dtype": "float32", "shape": [7]},
            },
        }
        (self.dataset_path / "meta" / "info.json").write_text(
            json.dumps(info), encoding="utf-8"
        )
        (self.dataset_path / "meta" / "stats.json").write_text("{}", encoding="utf-8")
        (self.dataset_path / "meta" / "episodes.jsonl").write_text(
            "".join(json.dumps({"episode_index": index}) + "\n" for index in range(1693)),
            encoding="utf-8",
        )
        (self.dataset_path / "meta" / "tasks.jsonl").write_text(
            "".join(json.dumps({"task_index": index}) + "\n" for index in range(40)),
            encoding="utf-8",
        )
        for index in range(1693):
            (self.dataset_path / "data" / "chunk-000" / f"episode_{index:06d}.parquet").touch()

    def tearDown(self):
        self.temp_dir.cleanup()

    def run_preflight(self) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                "python",
                str(PREFLIGHT),
                "--model-path",
                str(self.model_path),
                "--fast-path",
                str(self.fast_path),
                "--dataset-path",
                str(self.dataset_path),
                "--static-only",
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_static_preflight_accepts_complete_v21_layout(self):
        result = self.run_preflight()

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("1693 episodes", result.stdout)
        self.assertIn("273465 frames", result.stdout)
        self.assertIn("40 tasks", result.stdout)
        self.assertIn("observation.images.image2", result.stdout)

    def test_static_preflight_rejects_incomplete_download(self):
        incomplete = self.dataset_path / ".cache" / "episode.incomplete"
        incomplete.parent.mkdir()
        incomplete.touch()

        result = self.run_preflight()

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("incomplete", result.stderr.lower())

    def test_cgroup_memory_check_accepts_required_headroom(self):
        preflight = load_preflight_module()
        root = Path(self.temp_dir.name) / "cgroup"
        root.mkdir()
        limit = root / "memory.limit_in_bytes"
        usage = root / "memory.usage_in_bytes"
        limit.write_text(str(480 * 1024**3), encoding="utf-8")
        usage.write_text(str(300 * 1024**3), encoding="utf-8")

        headroom = preflight.validate_cgroup_memory_headroom(
            limit, usage, min_headroom_gib=120
        )

        self.assertEqual(headroom, 180.0)

    def test_cgroup_memory_check_rejects_insufficient_headroom(self):
        preflight = load_preflight_module()
        root = Path(self.temp_dir.name) / "cgroup"
        root.mkdir()
        limit = root / "memory.limit_in_bytes"
        usage = root / "memory.usage_in_bytes"
        limit.write_text(str(480 * 1024**3), encoding="utf-8")
        usage.write_text(str(400 * 1024**3), encoding="utf-8")

        with self.assertRaisesRegex(preflight.PreflightError, "headroom"):
            preflight.validate_cgroup_memory_headroom(
                limit, usage, min_headroom_gib=120
            )


if __name__ == "__main__":
    unittest.main()
