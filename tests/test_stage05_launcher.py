import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from utils.action_expert_config import load_action_expert_config
from utils.dataset_manifest import write_resolved_dataset_manifest
from utils.stage05_checkpoint_contract import (
    STAGE05_AR_JOINT_CONTRACT_KEY,
    STAGE05_DATASET_ENTRIES,
    build_stage05_ar_joint_contract,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_stage05_four_dataset_pretraining.sh"
TOKEN_AUDIT = (
    ROOT
    / "outputs/stage05_four_dataset_pretraining_20260904/audits/"
    "token_length_audit_v9_format2.json"
)
MODEL = Path("/opt/data/private/lq/models/Qwen3-VL-2B-Instruct")
ACTION_CONFIG = ROOT / "configs/stage05_four_dataset_action_expert.json"


def _run(tmp_path, stage, **extra):
    env = {
        **os.environ,
        "ZR0_DRY_RUN": "1",
        "ZR0_MAX_LENGTH": "1024",
        "ZR0_OUTPUT_ROOT": str(tmp_path / "output"),
        "ZR0_TOKEN_LENGTH_AUDIT": str(TOKEN_AUDIT),
        **extra,
    }
    return subprocess.run(
        ["bash", str(SCRIPT), stage], cwd=ROOT, env=env, text=True, capture_output=True
    )


def test_ar_smoke_dry_run_has_fixed_scale_and_no_action_expert(tmp_path):
    result = _run(tmp_path, "ar-smoke")
    assert result.returncode == 0, result.stderr
    command = result.stdout
    assert "--gradient_accumulation_steps 2" in command
    assert "--expected_global_batch_size 128" in command
    assert "--loss_type vlm" in command
    assert "--num_difference_queries 32" in command
    assert "--vlm_attention_backend sdpa" in command
    assert "--wandb_failure_policy best_effort" in command
    assert "--wandb_pending_capacity 256" in command
    assert "--wandb_retry_base_steps 1" in command
    assert "--wandb_retry_max_steps 128" in command
    assert "--wandb_finish_max_attempts 2" in command
    assert "--wandb_finish_timeout_seconds 15" in command
    assert "--action_expert_config_path" in command
    assert "--tune_action_expert" not in command
    assert not (tmp_path / "output").exists()


def test_formal_template_is_explicitly_parameterized_and_locked(tmp_path):
    missing_batch = _run(
        tmp_path,
        "ar-formal",
        ZR0_FORMAL_MAX_STEPS="1234",
        ZR0_WANDB_PROJECT="stage05",
    )
    assert missing_batch.returncode != 0
    assert "probe-validated" in missing_batch.stderr
    missing = _run(
        tmp_path,
        "ar-formal",
        ZR0_PER_DEVICE_BATCH_SIZE="16",
        ZR0_GRADIENT_ACCUMULATION_STEPS="2",
        ZR0_WANDB_PROJECT="stage05",
    )
    assert missing.returncode != 0
    assert "ZR0_FORMAL_MAX_STEPS" in missing.stderr
    ready = _run(
        tmp_path,
        "ar-formal",
        ZR0_FORMAL_MAX_STEPS="1234",
        ZR0_PER_DEVICE_BATCH_SIZE="16",
        ZR0_GRADIENT_ACCUMULATION_STEPS="2",
        ZR0_WANDB_PROJECT="stage05",
    )
    assert ready.returncode == 0, ready.stderr
    assert "--max_train_steps 1234" in ready.stdout


def test_experiment_entry_and_save_interval_are_explicit(tmp_path):
    entry = ROOT / "scripts/stage05_experiment_train.py"
    output = tmp_path / "isolated-formal"
    result = _run(
        tmp_path, "ar-formal", ZR0_FORMAL_MAX_STEPS="27582",
        ZR0_PER_DEVICE_BATCH_SIZE="16", ZR0_GRADIENT_ACCUMULATION_STEPS="2",
        ZR0_WANDB_PROJECT="stage05", ZR0_SAVE_STEP_INTERVAL="5000",
        ZR0_RUN_OUTPUT_DIR=str(output), ZR0_TRAIN_ENTRYPOINT=str(entry),
        ZR0_TOKEN_AUDIT_SPEC=str(ROOT / "configs/stage05_four_dataset_experiment.json"),
        ZR0_TOKEN_AUDIT_SCRIPT=str(ROOT / "scripts/audit_stage05_token_lengths.py"),
    )
    assert result.returncode == 0, result.stderr
    for fragment in (
        str(entry), str(output), "--max_train_steps 27582", "--save_step_interval 5000",
        "--vlm_loss_weight 1.0", "--peak_learning_rate 1e-5", "--warmup_ratio 0.05",
        "--adam_beta1 0.9", "--adam_beta2 0.95", "--adam_epsilon 1e-8",
    ):
        assert fragment in result.stdout
    assert not output.exists()


@pytest.mark.parametrize(("max_length", "success"), [("940", False), ("941", True), ("1024", True)])
def test_stage05_max_length_audit_floor(tmp_path, max_length, success):
    result = _run(tmp_path, "ar-smoke", ZR0_MAX_LENGTH=max_length)
    assert (result.returncode == 0) is success
    if not success:
        assert "audited Stage05 minimum 941" in result.stderr


def _make_ar_checkpoint(tmp_path):
    checkpoint = tmp_path / "output/pilot/ar/latest-model-optimizer-lr"
    checkpoint.mkdir(parents=True)
    for source in MODEL.iterdir():
        if source.name != "model.safetensors" and source.is_file():
            shutil.copy2(source, checkpoint / source.name)
    source_config = ACTION_CONFIG.read_bytes()
    (checkpoint / "action_expert_config.json").write_bytes(source_config)
    (checkpoint / "difference_query.safetensors").touch()
    (checkpoint / "difference_query_config.json").write_text(
        json.dumps(
            {
                "version": 1,
                "enabled": True,
                "num_difference_queries": 32,
                "hidden_size": 2048,
                "attention_backend": "sdpa",
            }
        ),
        encoding="utf-8",
    )
    manifest = {
        "format_version": 4,
        "loss_type": "vlm",
        "entries": [
            {
                "dataset_entry": name,
                "resolved_adapter": "stage05_mixed_pretraining",
            }
            for name in STAGE05_DATASET_ENTRIES
        ],
    }
    write_resolved_dataset_manifest(checkpoint, manifest)
    resolved = load_action_expert_config(ACTION_CONFIG)
    contract = build_stage05_ar_joint_contract(
        checkpoint_directory=checkpoint,
        resolved_action_expert_config=resolved.payload,
        source_config_sha256=hashlib.sha256(source_config).hexdigest(),
        resolved_dataset_manifest=manifest,
        vlm_hidden_size=2048,
        num_difference_queries=32,
    )
    metadata = {
        "version": 1,
        "checkpoint_kind": "ar_only",
        "action_expert": {
            "status": "not_constructed_future_joint_config_reference",
            "config_file": "action_expert_config.json",
            "config_sha256": resolved.parsed_sha256,
            "weights_file": None,
        },
        STAGE05_AR_JOINT_CONTRACT_KEY: contract,
    }
    (checkpoint / "zr0_checkpoint_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return checkpoint


def test_joint_dry_run_validates_explicit_action_expert_config(tmp_path):
    _make_ar_checkpoint(tmp_path)
    result = _run(tmp_path, "joint-smoke")
    assert result.returncode == 0, result.stderr
    assert "--action_expert_config_path" in result.stdout


@pytest.mark.parametrize("saved_world_size", [1, 4])
def test_ar_resume_dry_run_uses_own_purpose_and_validates_state(tmp_path, saved_world_size):
    from test_stage05_ar_scheduler_resume import (
        _advance, _make_scheduler, _save_scheduler_checkpoint,
    )

    checkpoint = _make_ar_checkpoint(tmp_path)
    target = tmp_path / "output/smoke/ar/latest-model-optimizer-lr"
    target.parent.mkdir(parents=True)
    shutil.move(checkpoint, target)
    state = _make_scheduler(saved_world_size)
    _advance(state)
    _save_scheduler_checkpoint(target, state)
    result = _run(tmp_path, "ar-resume")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "--checkpoint_load_purpose stage05_ar_resume" in result.stdout
    assert "--resume_training" in result.stdout
    assert "--loss_type vlm" in result.stdout
    assert "--action_expert_name_or_path" not in result.stdout
    (target / "scheduler.pt").write_bytes(b"corrupt")
    rejected = _run(tmp_path, "ar-resume")
    assert rejected.returncode != 0
    assert "scheduler state cannot be loaded" in rejected.stderr
    assert "accelerate launch" not in rejected.stdout


def test_joint_dry_run_rejects_invalid_action_expert_config_before_launch(tmp_path):
    _make_ar_checkpoint(tmp_path)
    invalid = tmp_path / "empty-action-config.json"
    invalid.write_text("{}", encoding="utf-8")
    result = _run(
        tmp_path,
        "joint-smoke",
        ZR0_ACTION_EXPERT_CONFIG_PATH=str(invalid),
    )
    assert result.returncode != 0
    assert "must be a non-empty JSON object" in result.stderr
    assert "accelerate launch" not in result.stdout


def test_joint_dry_run_rejects_legacy_tabletop_ar_checkpoint(tmp_path):
    checkpoint = _make_ar_checkpoint(tmp_path)
    metadata_path = checkpoint / "zr0_checkpoint_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.pop(STAGE05_AR_JOINT_CONTRACT_KEY)
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    result = _run(tmp_path, "joint-smoke")
    assert result.returncode != 0
    assert "Stage05 dataset identity" in result.stderr
    assert "accelerate launch" not in result.stdout


def test_launcher_uses_token_audit_validate_only_and_rejects_tampering(tmp_path):
    report = json.loads(TOKEN_AUDIT.read_text(encoding="utf-8"))
    report["data_identity"][0]["manifest_content_hash"] = "0" * 64
    from scripts.audit_stage05_token_lengths import _content_hash

    report["content_hash"] = _content_hash(report)
    tampered = tmp_path / "tampered-token-audit.json"
    tampered.write_text(json.dumps(report), encoding="utf-8")
    result = _run(
        tmp_path,
        "ar-smoke",
        ZR0_TOKEN_LENGTH_AUDIT=str(tampered),
    )
    assert result.returncode != 0
    assert "token audit report file hash differs from trusted spec" in result.stderr
    assert "accelerate launch" not in result.stdout
