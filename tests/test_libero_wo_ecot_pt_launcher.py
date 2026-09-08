import os
import json
import importlib.util
import subprocess
import tempfile
import unittest
import shlex
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "run_libero_wo_ecot_pt.sh"
PREFLIGHT = ROOT / "scripts" / "preflight_libero_wo_ecot_pt.py"
AFTER_PRETRAIN = ROOT / "scripts" / "run_libero_finetune_after_pretrain.sh"


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
        arm: str | None = None,
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
        command = ["bash", str(LAUNCHER), mode]
        if arm is not None:
            command.append(arm)
        return subprocess.run(
            command,
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

    def test_train_and_resume_preserve_resource_gate_uuid_selection(self):
        identities = "GPU-2,GPU-0,GPU-3,GPU-1"
        for mode in ("train", "resume"):
            result = self.run_launcher(mode, extra_env={"ZR0_CUDA_VISIBLE_DEVICES": identities})
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn(f"CUDA_VISIBLE_DEVICES={identities}", shlex.split(result.stdout))

    def test_pretrained_arm_warm_starts_complete_joint_checkpoint(self):
        source = ROOT / "outputs" / "pretrain" / "tabletop_v3_dq32_joint_gbs128_seed42_mbs16_gas2" / "step-19424"
        result = self.run_launcher("train", arm="difference_query_pretrained")

        self.assertEqual(result.returncode, 0, result.stderr)
        command = result.stdout
        self.assertIn(f"--vlm_name_or_path {source}", command)
        self.assertIn(f"--action_expert_name_or_path {source}", command)
        self.assertNotIn("--resume_training", command)
        self.assertIn("--gradient_accumulation_steps 1", command)
        self.assertIn("--expected_global_batch_size 64", command)
        self.assertIn("--per_device_train_batch_size 16", command)
        self.assertIn("--warmup_ratio 0.08", command)
        self.assertIn("--max_length 1200", command)
        self.assertIn("--dataloader_num_workers 24", command)
        self.assertIn("--wandb_failure_policy required", command)
        self.assertIn("--loss_type action", command)
        self.assertIn("--action_horizon 10", command)
        self.assertIn("--use_difference_query", command)
        self.assertIn("--num_difference_queries 32", command)
        self.assertIn("--vlm_attention_backend sdpa", command)
        self.assertIn(
            "--wandb_group libero-wo-ecot-pt-difference-query-tabletop-v3-joint-init",
            command,
        )
        self.assertIn("libero_zero2_bf16_mbs16_gas1.yaml", command)

        launch_tokens = shlex.split(command)
        train_index = launch_tokens.index(str(ROOT / "train_vla.py"))
        self.assertNotIn("--gradient_accumulation_steps", launch_tokens[:train_index])
        self.assertEqual(
            launch_tokens[train_index:].count("--gradient_accumulation_steps"), 1
        )

    def test_pretrained_arm_resume_uses_its_own_full_checkpoint(self):
        output = ROOT / "outputs" / "ckpts" / "test-pretrained-libero"
        result = self.run_launcher(
            "resume",
            arm="difference_query_pretrained",
            extra_env={"ZR0_OUTPUT_DIR": str(output)},
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("missing LIBERO initialization/resume checkpoint", result.stderr)

    def test_pretrained_accelerate_config_has_exact_batch_contract(self):
        config_path = (
            ROOT / "accelerate_configs" / "libero_zero2_bf16_mbs16_gas1.yaml"
        )
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        deepspeed = config["deepspeed_config"]

        self.assertEqual(config["num_processes"], 4)
        self.assertEqual(config["mixed_precision"], "bf16")
        self.assertEqual(deepspeed["zero_stage"], 2)
        self.assertEqual(deepspeed["gradient_accumulation_steps"], 1)
        self.assertEqual(deepspeed["train_micro_batch_size_per_gpu"], 16)
        self.assertEqual(deepspeed["train_batch_size"], 64)

    def test_after_pretrain_pipeline_keeps_smoke_and_formal_weights_isolated(self):
        result = subprocess.run(
            ["bash", str(AFTER_PRETRAIN), "dry-run"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("step-19424", result.stdout)
        self.assertIn("smoke=train:2,resume:3", result.stdout)
        pipeline_source = AFTER_PRETRAIN.read_text(encoding="utf-8")
        self.assertIn(
            'records = [record for record in all_records if "total_loss" in record]',
            pipeline_source,
        )
        self.assertIn("tabletop-v3-joint-init", result.stdout)

        source = AFTER_PRETRAIN.read_text(encoding="utf-8")
        self.assertIn('ZR0_MAX_TRAIN_STEPS=2', source)
        self.assertIn('ZR0_MAX_TRAIN_STEPS=3', source)
        self.assertIn('resume difference_query_pretrained', source)
        self.assertIn('train difference_query_pretrained', source)
        self.assertIn('if [[ -e "$FORMAL_OUTPUT" ]]', source)

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

    def test_three_experiment_arms_differ_only_in_allowed_controls(self):
        commands = {
            arm: self.run_launcher("train", arm=arm)
            for arm in ("baseline_fa2", "baseline_sdpa", "difference_query")
        }
        for arm, result in commands.items():
            self.assertEqual(result.returncode, 0, f"{arm}: {result.stderr}")

        baseline_fa2 = commands["baseline_fa2"].stdout
        baseline_sdpa = commands["baseline_sdpa"].stdout
        difference_query = commands["difference_query"].stdout
        self.assertNotIn("--vlm_attention_backend", baseline_fa2)
        self.assertNotIn("--use_difference_query", baseline_fa2)
        self.assertIn("--vlm_attention_backend sdpa", baseline_sdpa)
        self.assertNotIn("--use_difference_query", baseline_sdpa)
        self.assertIn("--vlm_attention_backend sdpa", difference_query)
        self.assertIn("--use_difference_query", difference_query)
        self.assertIn("--num_difference_queries 32", difference_query)

        def normalized(command: str) -> list[str]:
            tokens = shlex.split(command)
            flags_with_values = {
                "--vlm_attention_backend",
                "--num_difference_queries",
                "--output_ckpt_dir",
                "--tensorboard_log_dir",
                "--wandb_run_name",
                "--wandb_group",
            }
            normalized_tokens = []
            index = 0
            while index < len(tokens):
                token = tokens[index]
                if token in flags_with_values:
                    index += 2
                    continue
                if token == "--use_difference_query":
                    index += 1
                    continue
                if token == "--wandb_tags":
                    break
                normalized_tokens.append(token)
                index += 1
            return normalized_tokens

        expected = normalized(baseline_fa2)
        self.assertEqual(normalized(baseline_sdpa), expected)
        self.assertEqual(normalized(difference_query), expected)

    def test_unknown_experiment_arm_fails(self):
        result = self.run_launcher("train", arm="unknown-arm")

        self.assertEqual(result.returncode, 2)
        self.assertIn("experiment arm", result.stderr)

    def test_launcher_records_experiment_metadata_before_exec(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            fake_bin = root / "bin"
            fake_bin.mkdir()
            for executable in ("python", "accelerate"):
                path = fake_bin / executable
                path.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
                path.chmod(0o755)

            output_dir = root / "experiment-output"
            env = os.environ.copy()
            env.pop("ZR0_DRY_RUN", None)
            env.update(
                {
                    "PATH": f"{fake_bin}:{env['PATH']}",
                    "ZR0_OUTPUT_DIR": str(output_dir),
                    "ZR0_RUN_NAME": "metadata-test-run",
                    "ZR0_WANDB_RUN_ID": "meta1234",
                    "WANDB_API_KEY": "test-secret-that-must-not-be-printed",
                }
            )

            result = subprocess.run(
                ["bash", str(LAUNCHER), "train", "difference_query"],
                cwd=ROOT,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            experiment_doc = output_dir / "experiment.md"
            self.assertTrue(experiment_doc.is_file())
            contents = experiment_doc.read_text(encoding="utf-8")
            self.assertIn("## Runtime launch record", contents)
            self.assertIn("`difference_query`", contents)
            self.assertIn("`metadata-test-run`", contents)
            self.assertIn("`meta1234`", contents)
            self.assertIn("--use_difference_query", contents)
            self.assertIn("--vlm_attention_backend sdpa", contents)


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

    def test_joint_checkpoint_gate_accepts_complete_dq32_source(self):
        source = ROOT / "outputs" / "pretrain" / "tabletop_v3_dq32_joint_gbs128_seed42_mbs16_gas2" / "step-19424"
        reference_path = source / "action_expert_config.json"

        preflight = load_preflight_module()
        preflight.validate_checkpoint_artifacts(
            source,
            "joint",
            32,
            32,
            reference_path,
        )

    def test_joint_checkpoint_gate_rejects_missing_action_weights(self):
        from safetensors.torch import save_file
        import torch

        source = Path(self.temp_dir.name) / "incomplete-joint"
        source.mkdir()
        (source / "zr0_checkpoint_metadata.json").write_text(
            json.dumps({"version": 1, "checkpoint_kind": "joint"}),
            encoding="utf-8",
        )
        save_file({"weight": torch.zeros(1)}, source / "model.safetensors")

        preflight = load_preflight_module()
        with self.assertRaisesRegex(preflight.PreflightError, "Action Expert"):
            preflight.validate_checkpoint_artifacts(source, "joint", None, None, None)

    def test_output_space_check_accepts_existing_parent(self):
        preflight = load_preflight_module()
        free_gib = preflight.validate_output_space(
            Path(self.temp_dir.name) / "not-yet-created" / "output", 0.0
        )

        self.assertGreater(free_gib, 0.0)

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
