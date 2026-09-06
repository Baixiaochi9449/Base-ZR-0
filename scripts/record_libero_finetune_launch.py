#!/usr/bin/env python3
"""Record the immutable inputs and resolved contract for LIBERO fine-tuning."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.record_query_pretrain_launch import (  # noqa: E402
    resolve_model_weight_identity,
    sha256_file,
    unique_option_value,
)
from utils.cli_options import parse_train_options  # noqa: E402
from utils.dataset_manifest import resolved_manifest_json  # noqa: E402
from utils.load_training_dataset import build_concat_streaming_dataset  # noqa: E402


EXPECTED_FRAMES = 273_465
EXPECTED_WORLD_SIZE = 4
EXPECTED_MICRO_BATCH = 16
EXPECTED_GAS = 1
EXPECTED_GLOBAL_BATCH = 64
EXPECTED_STEPS_PER_EPOCH = 4_273
EXPECTED_FINAL_GLOBAL_BATCH = 60


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-mode", choices=("train", "resume"), required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--launcher", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser.parse_args()


def git_output(*args: str, binary: bool = False):
    return subprocess.check_output(
        ["git", *args], cwd=REPO_ROOT, text=not binary
    )


def file_identity(path: Path) -> dict:
    if not path.is_file():
        raise ValueError(f"required checkpoint file is missing: {path}")
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def batch_contract(
    *, samples: int, world_size: int, micro_batch: int, gas: int, epochs: int
) -> dict:
    aligned_presentations = math.ceil(samples / world_size) * world_size
    samples_per_rank = aligned_presentations // world_size
    microbatches_per_rank = math.ceil(samples_per_rank / micro_batch)
    optimizer_steps_per_epoch = math.ceil(microbatches_per_rank / gas)
    final_microbatches_per_rank = microbatches_per_rank - (
        optimizer_steps_per_epoch - 1
    ) * gas
    final_samples_per_rank = samples_per_rank - (
        optimizer_steps_per_epoch - 1
    ) * gas * micro_batch
    return {
        "unique_frames_per_epoch": samples,
        "aligned_presentations_per_epoch": aligned_presentations,
        "deterministic_duplicates_per_epoch": aligned_presentations - samples,
        "world_size": world_size,
        "per_device_micro_batch": micro_batch,
        "gradient_accumulation_steps": gas,
        "nominal_global_batch": world_size * micro_batch * gas,
        "optimizer_steps_per_epoch": optimizer_steps_per_epoch,
        "optimizer_steps_total": optimizer_steps_per_epoch * epochs,
        "samples_per_rank": samples_per_rank,
        "microbatches_per_rank": microbatches_per_rank,
        "final_microbatches_per_rank": final_microbatches_per_rank,
        "final_samples_per_rank": final_samples_per_rank,
        "final_global_batch": final_samples_per_rank * world_size,
        "drop_last": False,
        "pad_final_optimizer_window": False,
    }


def main() -> None:
    args = parse_args()
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    try:
        train_index = next(
            index for index, token in enumerate(command) if token.endswith("/train_vla.py")
        )
    except StopIteration as error:
        raise ValueError("launch command does not contain train_vla.py") from error

    options = parse_train_options(command[train_index + 1 :])
    outer_command = command[:train_index]
    accelerate_config_path = Path(
        unique_option_value(outer_command, "--config_file")
    ).resolve()
    accelerate_config = yaml.safe_load(
        accelerate_config_path.read_text(encoding="utf-8")
    )
    deepspeed = accelerate_config["deepspeed_config"]
    config_world_size = int(accelerate_config["num_processes"])
    outer_world_size = int(unique_option_value(outer_command, "--num_processes"))
    scale = {
        "world_size": args.world_size,
        "outer_world_size": outer_world_size,
        "config_world_size": config_world_size,
        "micro_batch": options.per_device_train_batch_size,
        "config_micro_batch": deepspeed["train_micro_batch_size_per_gpu"],
        "gradient_accumulation_steps": options.gradient_accumulation_steps,
        "config_gradient_accumulation_steps": deepspeed[
            "gradient_accumulation_steps"
        ],
        "global_batch": options.expected_global_batch_size,
        "config_global_batch": deepspeed["train_batch_size"],
    }
    expected_scale = {
        "world_size": EXPECTED_WORLD_SIZE,
        "outer_world_size": EXPECTED_WORLD_SIZE,
        "config_world_size": EXPECTED_WORLD_SIZE,
        "micro_batch": EXPECTED_MICRO_BATCH,
        "config_micro_batch": EXPECTED_MICRO_BATCH,
        "gradient_accumulation_steps": EXPECTED_GAS,
        "config_gradient_accumulation_steps": EXPECTED_GAS,
        "global_batch": EXPECTED_GLOBAL_BATCH,
        "config_global_batch": EXPECTED_GLOBAL_BATCH,
    }
    if scale != expected_scale:
        raise ValueError(f"unexpected LIBERO distributed scale: {scale}")

    concat = build_concat_streaming_dataset(
        dataset_entries=options.dataset_entries,
        model_name_or_path=options.vlm_name_or_path,
        fast_tokenizer_path=options.FAST_tokenizer_path,
        window_size=options.window_size,
        action_horizon=options.action_horizon,
        accelerator=None,
        process_mode="train",
        max_pad_state_and_action_length=options.max_pad_state_and_action_length,
        loss_type=options.loss_type,
        max_length=options.max_length,
        dataset_sample_ratios=options.dataset_sample_ratios,
    )
    if len(concat.datasets) != 1 or len(concat) != EXPECTED_FRAMES:
        raise ValueError(
            f"expected one {EXPECTED_FRAMES}-frame LIBERO dataset, got {len(concat)}"
        )

    contract = batch_contract(
        samples=len(concat),
        world_size=args.world_size,
        micro_batch=options.per_device_train_batch_size,
        gas=options.gradient_accumulation_steps,
        epochs=options.epochs,
    )
    required_contract = {
        "optimizer_steps_per_epoch": EXPECTED_STEPS_PER_EPOCH,
        "optimizer_steps_total": 34_184,
        "final_global_batch": EXPECTED_FINAL_GLOBAL_BATCH,
        "deterministic_duplicates_per_epoch": 3,
    }
    if any(contract[key] != value for key, value in required_contract.items()):
        raise ValueError(f"unexpected LIBERO batch contract: {contract}")

    model_path = Path(options.vlm_name_or_path).resolve()
    action_path = Path(options.action_expert_name_or_path).resolve()
    dataset_manifest = json.loads(
        resolved_manifest_json(concat.resolved_dataset_manifest)
    )
    staged_diff = git_output("diff", "--cached", "--binary", binary=True)
    record = {
        "version": 1,
        "experiment": "libero_wo_ecot_pt_dq32_tabletop_v3_joint_init",
        "run_mode": args.run_mode,
        "git_head": git_output("rev-parse", "HEAD").strip(),
        "git_status_porcelain": git_output("status", "--porcelain=v1").splitlines(),
        "staged_diff_sha256": hashlib.sha256(staged_diff).hexdigest(),
        "launcher": {
            "path": str(args.launcher.resolve()),
            "sha256": sha256_file(args.launcher.resolve()),
        },
        "distributed_config": {
            "path": str(accelerate_config_path),
            "sha256": sha256_file(accelerate_config_path),
            "parsed": accelerate_config,
            "resolved_scale": scale,
        },
        "vlm_and_query_source": {
            "path": str(model_path),
            "weights": resolve_model_weight_identity(model_path),
            "difference_query_config": file_identity(
                model_path / "difference_query_config.json"
            ),
            "difference_query_weights": file_identity(
                model_path / "difference_query.safetensors"
            ),
        },
        "action_expert_source": {
            "path": str(action_path),
            "config": file_identity(action_path / "action_expert_config.json"),
            "weights": file_identity(action_path / "action_expert.safetensors"),
        },
        "dataset_manifest": dataset_manifest,
        "dataset_manifest_sha256": dataset_manifest["content_hash"],
        "parsed_train_config": vars(options),
        "batch_contract": contract,
        "launch_command": command,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(record, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(record, ensure_ascii=True, sort_keys=True))


if __name__ == "__main__":
    main()
