import os
import shlex
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

from scripts.record_query_pretrain_launch import resolve_model_weight_identity
from utils.cli_options import parse_train_options


ROOT = Path(__file__).resolve().parents[1]
SMOKE_LAUNCHER = ROOT / "scripts" / "run_query_ar_joint_smoke.sh"
FORMAL_LAUNCHER = ROOT / "scripts" / "run_query_ar_joint_formal.sh"
ZR0_BIN = Path("/opt/data/private/lq/miniconda3/envs/ZR-0/bin")


def write_test_accelerate_config(directory: str | Path, gas: int = 32) -> Path:
    config = yaml.safe_load(
        (ROOT / "accelerate_configs/accelerate_config.yaml").read_text(
            encoding="utf-8"
        )
    )
    config["deepspeed_config"]["gradient_accumulation_steps"] = gas
    config["deepspeed_config"]["train_micro_batch_size_per_gpu"] = 1
    config["deepspeed_config"]["train_batch_size"] = 4 * gas
    config["num_processes"] = 4
    path = Path(directory) / f"accelerate-gas{gas}.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


class QueryArJointSmokeLauncherTest(unittest.TestCase):
    def run_launcher(
        self, mode: str, *, remove: str | None = None
    ) -> subprocess.CompletedProcess[str]:
        with tempfile.TemporaryDirectory() as config_directory:
            env = os.environ.copy()
            env.update(
                {
                    "ZR0_DRY_RUN": "1",
                    "ZR0_MODEL_PATH": "/models/qwen",
                    "ZR0_DATASET_ENTRIES": "future_difference",
                    "ZR0_DATASET_SAMPLE_RATIOS": "1.0",
                    "ZR0_MAX_LENGTH": "4096",
                    "ZR0_SMOKE_OUTPUT_ROOT": "/tmp/zr0-smoke",
                    "ZR0_EXPERIMENT_DOC": str(
                        ROOT / "experiments/query_conditioned_ar_joint_smoke/experiment.md"
                    ),
                    "ZR0_NUM_GPUS": "4",
                    "ZR0_PER_DEVICE_BATCH_SIZE": "1",
                    "ZR0_GRADIENT_ACCUMULATION_STEPS": "32",
                    "ZR0_EXPECTED_GLOBAL_BATCH_SIZE": "128",
                    "ZR0_WANDB_PROJECT": "ZR-0-Pretraining",
                    "ZR0_WANDB_GROUP": "tabletop-v3-dq32-gbs128-seed42",
                    "ZR0_WANDB_AR_RUN_NAME": "ar-smoke",
                    "ZR0_WANDB_AR_RUN_ID": "arsmoke01",
                    "ZR0_WANDB_JOINT_RUN_NAME": "joint-smoke",
                    "ZR0_WANDB_JOINT_RUN_ID": "jointsmoke01",
                    "WANDB_API_KEY": "test-key",
                    "WANDB_ENTITY": "test-entity",
                    "ZR0_TRAIN_PYTHON": sys.executable,
                    "ZR0_ACCELERATE_CONFIG": str(
                        write_test_accelerate_config(config_directory)
                    ),
                }
            )
            if remove is not None:
                env.pop(remove, None)
            return subprocess.run(
                ["bash", str(SMOKE_LAUNCHER), mode],
                cwd=ROOT,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )

    @staticmethod
    def command_tokens(result: subprocess.CompletedProcess[str]) -> list[str]:
        return shlex.split(result.stdout.strip())

    def test_ar_step1_is_one_step_vlm_only_with_trainable_vlm(self):
        result = self.run_launcher("ar-step1")

        self.assertEqual(result.returncode, 0, result.stderr)
        tokens = self.command_tokens(result)
        self.assertIn("--tune_vlm", tokens)
        self.assertNotIn("--tune_action_expert", tokens)
        self.assertNotIn("--action_expert_name_or_path", tokens)
        self.assertEqual(tokens[tokens.index("--loss_type") + 1], "vlm")
        self.assertEqual(tokens[tokens.index("--max_train_steps") + 1], "1")
        self.assertEqual(tokens[tokens.index("--max_length") + 1], "4096")
        self.assertEqual(tokens[tokens.index("--dataloader_num_workers") + 1], "4")
        self.assertEqual(
            tokens[tokens.index("--gradient_accumulation_steps") + 1], "32"
        )
        self.assertEqual(
            tokens[tokens.index("--expected_global_batch_size") + 1], "128"
        )
        train_script_index = next(
            index for index, token in enumerate(tokens) if token.endswith("/train_vla.py")
        )
        outer_gas_index = tokens.index("--gradient_accumulation_steps")
        self.assertLess(outer_gas_index, train_script_index)
        self.assertEqual(tokens[outer_gas_index + 1], "32")
        self.assertEqual(tokens.count("--gradient_accumulation_steps"), 2)
        self.assertEqual(
            tokens[tokens.index("--dataset_sample_ratios") + 1], "1.0"
        )

    def test_ar_resume_restores_full_training_state_to_step2(self):
        result = self.run_launcher("ar-resume-step2")

        self.assertEqual(result.returncode, 0, result.stderr)
        tokens = self.command_tokens(result)
        checkpoint = "/tmp/zr0-smoke/ar/latest-model-optimizer-lr"
        self.assertEqual(tokens[tokens.index("--vlm_name_or_path") + 1], checkpoint)
        self.assertNotIn("--action_expert_name_or_path", tokens)
        self.assertIn("--resume_training", tokens)
        self.assertNotIn("--tune_action_expert", tokens)
        self.assertEqual(tokens[tokens.index("--max_train_steps") + 1], "2")

    def test_joint_step1_warm_starts_only_vlm_and_query_from_ar_checkpoint(self):
        result = self.run_launcher("joint-step1")

        self.assertEqual(result.returncode, 0, result.stderr)
        tokens = self.command_tokens(result)
        self.assertEqual(
            tokens[tokens.index("--vlm_name_or_path") + 1],
            "/tmp/zr0-smoke/ar/latest-model-optimizer-lr",
        )
        self.assertNotIn("--action_expert_name_or_path", tokens)
        self.assertNotIn("--resume_training", tokens)
        self.assertIn("--tune_vlm", tokens)
        self.assertIn("--tune_action_expert", tokens)
        self.assertEqual(tokens[tokens.index("--loss_type") + 1], "vlm_and_action")
        self.assertEqual(tokens[tokens.index("--max_train_steps") + 1], "1")

    def test_joint_resume_restores_full_training_state_to_step2(self):
        result = self.run_launcher("joint-resume-step2")

        self.assertEqual(result.returncode, 0, result.stderr)
        tokens = self.command_tokens(result)
        checkpoint = "/tmp/zr0-smoke/joint/latest-model-optimizer-lr"
        self.assertEqual(tokens[tokens.index("--vlm_name_or_path") + 1], checkpoint)
        self.assertEqual(
            tokens[tokens.index("--action_expert_name_or_path") + 1], checkpoint
        )
        self.assertIn("--resume_training", tokens)
        self.assertEqual(tokens[tokens.index("--max_train_steps") + 1], "2")

    def test_missing_required_input_fails_clearly(self):
        result = self.run_launcher("ar-step1", remove="ZR0_MAX_LENGTH")

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("ZR0_MAX_LENGTH", result.stderr)

    def test_unknown_stage_fails(self):
        result = self.run_launcher("unknown")

        self.assertEqual(result.returncode, 2)
        self.assertIn("Usage:", result.stderr)

    def test_resume_preserves_existing_experiment_record(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_root = Path(temporary_directory)
            checkpoint = output_root / "ar" / "latest-model-optimizer-lr"
            checkpoint.mkdir(parents=True)
            (checkpoint / "scheduler.pt").write_bytes(b"scheduler")
            experiment = output_root / "ar" / "experiment.md"
            experiment.write_text("existing-stage-record\n", encoding="utf-8")
            env = os.environ.copy()
            env.update(
                {
                    "ZR0_ACCELERATE_BIN": "/bin/true",
                    "ZR0_MODEL_PATH": "/models/qwen",
                    "ZR0_DATASET_ENTRIES": "future_difference",
                    "ZR0_DATASET_SAMPLE_RATIOS": "1.0",
                    "ZR0_MAX_LENGTH": "4096",
                    "ZR0_SMOKE_OUTPUT_ROOT": str(output_root),
                    "ZR0_EXPERIMENT_DOC": str(
                        ROOT / "experiments/query_conditioned_ar_joint_smoke/experiment.md"
                    ),
                    "ZR0_NUM_GPUS": "4",
                    "ZR0_PER_DEVICE_BATCH_SIZE": "1",
                    "ZR0_GRADIENT_ACCUMULATION_STEPS": "32",
                    "ZR0_EXPECTED_GLOBAL_BATCH_SIZE": "128",
                    "ZR0_WANDB_PROJECT": "ZR-0-Pretraining",
                    "ZR0_WANDB_GROUP": "test",
                    "ZR0_WANDB_AR_RUN_NAME": "ar-smoke",
                    "ZR0_WANDB_AR_RUN_ID": "arsmoke01",
                    "ZR0_WANDB_JOINT_RUN_NAME": "joint-smoke",
                    "ZR0_WANDB_JOINT_RUN_ID": "jointsmoke01",
                    "WANDB_API_KEY": "test-key",
                    "WANDB_ENTITY": "test-entity",
                    "ZR0_TRAIN_PYTHON": sys.executable,
                    "ZR0_ACCELERATE_CONFIG": str(
                        write_test_accelerate_config(output_root)
                    ),
                }
            )

            result = subprocess.run(
                ["bash", str(SMOKE_LAUNCHER), "ar-resume-step2"],
                cwd=ROOT,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            record = experiment.read_text(encoding="utf-8")
            self.assertIn("existing-stage-record", record)
            self.assertIn("ar-resume-step2", record)
            self.assertIn("exit status: 0", record)


    def test_joint_warm_start_requires_ar_only_checkpoint_kind(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_root = Path(temporary_directory)
            checkpoint = output_root / "ar" / "latest-model-optimizer-lr"
            checkpoint.mkdir(parents=True)
            (checkpoint / "scheduler.pt").write_bytes(b"scheduler")
            (checkpoint / "difference_query_config.json").write_text(
                '{"version":1,"enabled":true,"num_difference_queries":32,'
                '"hidden_size":3,"attention_backend":"sdpa"}\n',
                encoding="utf-8",
            )
            (checkpoint / "difference_query.safetensors").write_bytes(b"query")
            (checkpoint / "zr0_checkpoint_metadata.json").write_text(
                '{"version":1,"checkpoint_kind":"joint"}\n', encoding="utf-8"
            )
            env = os.environ.copy()
            env.update(
                {
                    "ZR0_ACCELERATE_BIN": "/bin/true",
                    "ZR0_MODEL_PATH": "/models/qwen",
                    "ZR0_DATASET_ENTRIES": "future_difference",
                    "ZR0_DATASET_SAMPLE_RATIOS": "1.0",
                    "ZR0_MAX_LENGTH": "4096",
                    "ZR0_SMOKE_OUTPUT_ROOT": str(output_root),
                    "ZR0_EXPERIMENT_DOC": str(
                        ROOT / "experiments/query_conditioned_ar_joint_smoke/experiment.md"
                    ),
                    "ZR0_NUM_GPUS": "4",
                    "ZR0_PER_DEVICE_BATCH_SIZE": "1",
                    "ZR0_GRADIENT_ACCUMULATION_STEPS": "32",
                    "ZR0_EXPECTED_GLOBAL_BATCH_SIZE": "128",
                    "ZR0_WANDB_PROJECT": "ZR-0-Pretraining",
                    "ZR0_WANDB_GROUP": "test",
                    "ZR0_WANDB_AR_RUN_NAME": "ar-smoke",
                    "ZR0_WANDB_AR_RUN_ID": "arsmoke01",
                    "ZR0_WANDB_JOINT_RUN_NAME": "joint-smoke",
                    "ZR0_WANDB_JOINT_RUN_ID": "jointsmoke01",
                    "WANDB_API_KEY": "test-key",
                    "WANDB_ENTITY": "test-entity",
                    "ZR0_TRAIN_PYTHON": sys.executable,
                    "ZR0_ACCELERATE_CONFIG": str(
                        write_test_accelerate_config(output_root)
                    ),
                }
            )

            result = subprocess.run(
                ["bash", str(SMOKE_LAUNCHER), "joint-step1"],
                cwd=ROOT,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("ar_only", result.stderr)

    def test_joint_resume_requires_action_expert_sidecars(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_root = Path(temporary_directory)
            checkpoint = output_root / "joint" / "latest-model-optimizer-lr"
            checkpoint.mkdir(parents=True)
            (checkpoint / "scheduler.pt").write_bytes(b"scheduler")
            env = os.environ.copy()
            env.update(
                {
                    "ZR0_ACCELERATE_BIN": "/bin/true",
                    "ZR0_MODEL_PATH": "/models/qwen",
                    "ZR0_DATASET_ENTRIES": "future_difference",
                    "ZR0_DATASET_SAMPLE_RATIOS": "1.0",
                    "ZR0_MAX_LENGTH": "4096",
                    "ZR0_SMOKE_OUTPUT_ROOT": str(output_root),
                    "ZR0_EXPERIMENT_DOC": str(
                        ROOT / "experiments/query_conditioned_ar_joint_smoke/experiment.md"
                    ),
                    "ZR0_NUM_GPUS": "4",
                    "ZR0_PER_DEVICE_BATCH_SIZE": "1",
                    "ZR0_GRADIENT_ACCUMULATION_STEPS": "32",
                    "ZR0_EXPECTED_GLOBAL_BATCH_SIZE": "128",
                    "ZR0_WANDB_PROJECT": "ZR-0-Pretraining",
                    "ZR0_WANDB_GROUP": "test",
                    "ZR0_WANDB_AR_RUN_NAME": "ar-smoke",
                    "ZR0_WANDB_AR_RUN_ID": "arsmoke01",
                    "ZR0_WANDB_JOINT_RUN_NAME": "joint-smoke",
                    "ZR0_WANDB_JOINT_RUN_ID": "jointsmoke01",
                    "WANDB_API_KEY": "test-key",
                    "WANDB_ENTITY": "test-entity",
                    "ZR0_TRAIN_PYTHON": sys.executable,
                    "ZR0_ACCELERATE_CONFIG": str(
                        write_test_accelerate_config(output_root)
                    ),
                }
            )

            result = subprocess.run(
                ["bash", str(SMOKE_LAUNCHER), "joint-resume-step2"],
                cwd=ROOT,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("incomplete", result.stderr)


class ModelWeightIdentityTest(unittest.TestCase):
    def test_single_safetensors_identity_preserves_file_digest(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            model_path = Path(temporary_directory)
            weight_path = model_path / "model.safetensors"
            weight_path.write_bytes(b"single-weight")

            identity = resolve_model_weight_identity(model_path)

            self.assertEqual(identity["format"], "single_safetensors")
            self.assertEqual(identity["aggregate_sha256"], identity["files"][0]["sha256"])

    def test_sharded_safetensors_identity_uses_index_weight_map(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            model_path = Path(temporary_directory)
            (model_path / "model-00001-of-00002.safetensors").write_bytes(b"one")
            (model_path / "model-00002-of-00002.safetensors").write_bytes(b"two")
            (model_path / "model.safetensors.index.json").write_text(
                '{"weight_map":{"b":"model-00002-of-00002.safetensors",'
                '"a":"model-00001-of-00002.safetensors"}}\n',
                encoding="utf-8",
            )

            identity = resolve_model_weight_identity(model_path)

            self.assertEqual(identity["format"], "sharded_safetensors")
            self.assertEqual(
                [item["relative_path"] for item in identity["files"]],
                [
                    "model-00001-of-00002.safetensors",
                    "model-00002-of-00002.safetensors",
                ],
            )
            self.assertEqual(len(identity["aggregate_sha256"]), 64)


class QueryArJointFormalLauncherTest(unittest.TestCase):
    def setUp(self):
        self.temp_directory = tempfile.TemporaryDirectory()
        self.experiment_doc = Path(self.temp_directory.name) / "experiment.md"
        self.experiment_doc.write_text("# test experiment\n", encoding="utf-8")
        self.ar_checkpoint = Path(self.temp_directory.name) / "ar-checkpoint"
        self.ar_checkpoint.mkdir()
        (self.ar_checkpoint / "difference_query_config.json").write_text(
            '{"version":1,"enabled":true,"num_difference_queries":32,'
            '"hidden_size":2048,"attention_backend":"sdpa"}\n',
            encoding="utf-8",
        )
        (self.ar_checkpoint / "difference_query.safetensors").write_bytes(b"query")
        (self.ar_checkpoint / "zr0_checkpoint_metadata.json").write_text(
            '{"version":1,"checkpoint_kind":"ar_only"}\n',
            encoding="utf-8",
        )
        self.accelerate_config = write_test_accelerate_config(
            self.temp_directory.name
        )

    def tearDown(self):
        self.temp_directory.cleanup()

    def environment(self) -> dict[str, str]:
        env = os.environ.copy()
        env["PATH"] = f"{ZR0_BIN}:{env['PATH']}"
        env.update(
            {
                "ZR0_DRY_RUN": "1",
                "MODEL_PATH": str(self.ar_checkpoint),
                "OUTPUT_DIR": "/tmp/zr0-formal",
                "EXPERIMENT_DOC": str(self.experiment_doc),
                "MAX_LENGTH": "8192",
                "EPOCHS": "8",
                "MAX_TRAIN_STEPS": "19424",
                "SAVE_STEP_INTERVAL": "4856",
                "EXPECTED_GLOBAL_BATCH_SIZE": "128",
                "NUM_GPUS": "4",
                "PER_DEVICE_BATCH_SIZE": "1",
                "GRADIENT_ACCUMULATION_STEPS": "32",
                "DATASET_ENTRIES": "future_difference second_dataset",
                "SAMPLE_RATIOS": "0.75 0.25",
                "CUDA_VISIBLE_DEVICES": "0,1,2,3",
                "WANDB_API_KEY": "secret-must-not-be-printed",
                "WANDB_ENTITY": "test-entity",
                "WANDB_PROJECT": "ZR-0-Pretraining",
                "WANDB_GROUP": "query-ar-joint",
                "WANDB_RUN_NAME": "formal-test",
                "WANDB_RUN_ID": "formal01",
                "ACCELERATE_CONFIG": str(self.accelerate_config),
            }
        )
        return env

    def run_launcher(
        self, stage: str, *, env: dict[str, str] | None = None
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", str(FORMAL_LAUNCHER), stage],
            cwd=ROOT,
            env=env or self.environment(),
            text=True,
            capture_output=True,
            check=False,
        )

    def test_missing_required_training_scale_fails(self):
        env = self.environment()
        env.pop("EPOCHS")

        result = self.run_launcher("joint", env=env)

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("EPOCHS", result.stderr)

    def test_global_batch_must_equal_128(self):
        env = self.environment()
        env["GRADIENT_ACCUMULATION_STEPS"] = "1"

        result = self.run_launcher("joint", env=env)

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("global batch size must be 128", result.stderr)

    def test_accelerate_config_gas_must_match_training_cli(self):
        env = self.environment()
        env["ACCELERATE_CONFIG"] = str(
            write_test_accelerate_config(self.temp_directory.name, gas=16)
        )

        result = self.run_launcher("joint", env=env)

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("config GAS must be integer 32", result.stderr)

    def test_formal_training_rejects_disabled_wandb(self):
        env = self.environment()
        env["WANDB_MODE"] = "disabled"

        result = self.run_launcher("joint", env=env)

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("WANDB_MODE", result.stderr)

    def test_dataset_entries_and_ratios_must_have_equal_length(self):
        env = self.environment()
        env["SAMPLE_RATIOS"] = "1.0"

        result = self.run_launcher("joint", env=env)

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("DATASET_ENTRIES", result.stderr)
        self.assertIn("SAMPLE_RATIOS", result.stderr)

    def test_dataset_ratios_must_not_exceed_one(self):
        env = self.environment()
        env["SAMPLE_RATIOS"] = "1.1 0.25"

        result = self.run_launcher("joint", env=env)

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("at most one", result.stderr)

    def test_joint_command_pins_paper_optimizer_loss_and_horizon(self):
        result = self.run_launcher("joint")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("secret-must-not-be-printed", result.stdout)
        tokens = shlex.split(result.stdout.strip())
        expected_pairs = {
            "--num_processes": "4",
            "--gradient_accumulation_steps": "32",
            "--expected_global_batch_size": "128",
            "--mixed_precision": "bf16",
            "--per_device_train_batch_size": "1",
            "--epochs": "8",
            "--max_train_steps": "19424",
            "--action_horizon": "32",
            "--num_difference_queries": "32",
            "--vlm_loss_weight": "1.0",
            "--action_expert_loss_weight": "5.0",
            "--peak_learning_rate": "1e-5",
            "--min_lr_rate": "0.1",
            "--warmup_ratio": "0.05",
            "--adam_beta1": "0.9",
            "--adam_beta2": "0.95",
            "--adam_epsilon": "1e-8",
            "--max_length": "8192",
            "--dataloader_num_workers": "4",
        }
        for option, expected in expected_pairs.items():
            self.assertEqual(tokens[tokens.index(option) + 1], expected, option)
        dataset_index = tokens.index("--dataset_entries")
        self.assertEqual(
            tokens[dataset_index + 1 : dataset_index + 3],
            ["future_difference", "second_dataset"],
        )
        ratio_index = tokens.index("--dataset_sample_ratios")
        self.assertEqual(tokens[ratio_index + 1 : ratio_index + 3], ["0.75", "0.25"])
        self.assertIn("--tune_vlm", tokens)
        self.assertIn("--tune_action_expert", tokens)
        self.assertIn("--save_optimizer_and_lr_states", tokens)
        self.assertEqual(tokens[tokens.index("--loss_type") + 1], "vlm_and_action")
        self.assertIn("WANDB_MODE=online", tokens)

        train_script_index = next(
            index for index, token in enumerate(tokens) if token.endswith("/train_vla.py")
        )
        outer_gas_index = tokens.index("--gradient_accumulation_steps")
        self.assertLess(outer_gas_index, train_script_index)
        self.assertEqual(tokens[outer_gas_index + 1], "32")
        self.assertEqual(tokens.count("--gradient_accumulation_steps"), 2)
        parsed = parse_train_options(tokens[train_script_index + 1 :])
        self.assertEqual(parsed.dataset_entries, ["future_difference", "second_dataset"])
        self.assertEqual(parsed.dataset_sample_ratios, [0.75, 0.25])
        self.assertEqual(parsed.max_length, 8192)
        self.assertEqual(parsed.gradient_accumulation_steps, 32)
        self.assertEqual(parsed.expected_global_batch_size, 128)

    def test_joint_requires_enabled_32_query_ar_checkpoint(self):
        (self.ar_checkpoint / "difference_query_config.json").write_text(
            '{"version":1,"enabled":false,"num_difference_queries":null,'
            '"hidden_size":2048,"attention_backend":"sdpa"}\n',
            encoding="utf-8",
        )

        result = self.run_launcher("joint")

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("enabled", result.stderr)
        self.assertIn("32", result.stderr)

    def test_joint_rejects_non_ar_checkpoint_kind(self):
        (self.ar_checkpoint / "zr0_checkpoint_metadata.json").write_text(
            '{"version":1,"checkpoint_kind":"joint"}\n',
            encoding="utf-8",
        )
        result = self.run_launcher("joint")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("ar_only", result.stderr)

    def test_ar_command_does_not_tune_action_expert(self):
        result = self.run_launcher("ar")

        self.assertEqual(result.returncode, 0, result.stderr)
        tokens = shlex.split(result.stdout.strip())
        self.assertIn("--tune_vlm", tokens)
        self.assertNotIn("--tune_action_expert", tokens)
        self.assertEqual(tokens[tokens.index("--loss_type") + 1], "vlm")


if __name__ == "__main__":
    unittest.main()
