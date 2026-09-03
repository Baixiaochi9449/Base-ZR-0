#!/usr/bin/env python3
"""Write the authoritative pre-launch identity for Query pretraining."""

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

from utils.cli_options import parse_train_options
from utils.dataset_manifest import resolved_manifest_json
from utils.load_training_dataset import build_concat_streaming_dataset


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_model_weight_identity(model_path: Path) -> dict:
    single_weight = model_path / "model.safetensors"
    if single_weight.is_file():
        digest = sha256_file(single_weight)
        return {
            "format": "single_safetensors",
            "aggregate_sha256": digest,
            "index_file": None,
            "index_sha256": None,
            "files": [
                {
                    "path": str(single_weight),
                    "relative_path": single_weight.name,
                    "size_bytes": single_weight.stat().st_size,
                    "sha256": digest,
                }
            ],
        }

    index_path = model_path / "model.safetensors.index.json"
    if not index_path.is_file():
        raise ValueError(
            f"expected model.safetensors or model.safetensors.index.json in {model_path}"
        )
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"invalid safetensors weight map: {index_path}")
    relative_paths = sorted(set(weight_map.values()))
    if not all(isinstance(path, str) and path for path in relative_paths):
        raise ValueError(f"invalid safetensors shard path in: {index_path}")

    files = []
    for relative_path in relative_paths:
        shard = model_path / relative_path
        if not shard.is_file():
            raise ValueError(f"model weight shard is missing: {shard}")
        files.append(
            {
                "path": str(shard),
                "relative_path": relative_path,
                "size_bytes": shard.stat().st_size,
                "sha256": sha256_file(shard),
            }
        )
    aggregate_payload = json.dumps(
        [
            {
                "relative_path": item["relative_path"],
                "size_bytes": item["size_bytes"],
                "sha256": item["sha256"],
            }
            for item in files
        ],
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return {
        "format": "sharded_safetensors",
        "aggregate_sha256": hashlib.sha256(aggregate_payload).hexdigest(),
        "index_file": str(index_path),
        "index_sha256": sha256_file(index_path),
        "files": files,
    }


def git_output(*args: str, binary: bool = False):
    return subprocess.check_output(
        ["git", *args], cwd=REPO_ROOT, text=not binary
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("ar", "joint"), required=True)
    parser.add_argument("--run-mode", choices=("train", "resume"), required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--launcher", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser.parse_args()


def unique_option_value(command: list[str], option: str) -> str:
    indices = [index for index, token in enumerate(command) if token == option]
    if len(indices) != 1 or indices[0] + 1 >= len(command):
        raise ValueError(f"launch command must contain exactly one {option}")
    return command[indices[0] + 1]


def main():
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
    config_gas = accelerate_config["deepspeed_config"][
        "gradient_accumulation_steps"
    ]
    config_micro_batch = accelerate_config["deepspeed_config"][
        "train_micro_batch_size_per_gpu"
    ]
    config_global_batch = accelerate_config["deepspeed_config"]["train_batch_size"]
    outer_gas = int(
        unique_option_value(outer_command, "--gradient_accumulation_steps")
    )
    if not isinstance(config_gas, int) or config_gas < 1:
        raise ValueError("Accelerate/DeepSpeed gradient accumulation must be a positive integer")
    if config_gas != outer_gas or outer_gas != options.gradient_accumulation_steps:
        raise ValueError(
            "gradient accumulation mismatch across config/launcher/train CLI: "
            f"{config_gas}/{outer_gas}/{options.gradient_accumulation_steps}"
        )
    if config_micro_batch != options.per_device_train_batch_size:
        raise ValueError(
            "micro-batch mismatch across config/train CLI: "
            f"{config_micro_batch}/{options.per_device_train_batch_size}"
        )
    if config_global_batch != options.expected_global_batch_size:
        raise ValueError(
            "global batch mismatch across config/train CLI: "
            f"{config_global_batch}/{options.expected_global_batch_size}"
        )
    config_world_size = int(accelerate_config["num_processes"])
    outer_world_size = int(unique_option_value(outer_command, "--num_processes"))
    if config_world_size != args.world_size or outer_world_size != args.world_size:
        raise ValueError(
            "world size mismatch across config/launcher/recorder: "
            f"{config_world_size}/{outer_world_size}/{args.world_size}"
        )
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
    if len(concat.datasets) != 1:
        raise ValueError("this launch recorder requires exactly one dataset")

    world_size = args.world_size
    nominal_global_batch = (
        world_size
        * options.per_device_train_batch_size
        * options.gradient_accumulation_steps
    )
    if nominal_global_batch != options.expected_global_batch_size:
        raise ValueError("expected global batch is not divisible by micro-batch and GAS")
    unique_frames = len(concat)
    aligned_presentations = math.ceil(unique_frames / world_size) * world_size
    per_rank_samples = aligned_presentations // world_size
    per_rank_microbatches = math.ceil(
        per_rank_samples / options.per_device_train_batch_size
    )
    optimizer_steps_per_epoch = math.ceil(
        per_rank_microbatches / options.gradient_accumulation_steps
    )
    final_microbatches_per_rank = per_rank_microbatches - (
        optimizer_steps_per_epoch - 1
    ) * options.gradient_accumulation_steps
    final_samples_per_rank = per_rank_samples - (
        optimizer_steps_per_epoch - 1
    ) * (
        options.gradient_accumulation_steps
        * options.per_device_train_batch_size
    )
    final_global_batch = final_samples_per_rank * world_size

    model_path = Path(options.vlm_name_or_path).resolve()
    model_weight_identity = resolve_model_weight_identity(model_path)
    model_config_path = model_path / "config.json"
    launcher = Path(args.launcher).resolve()
    staged_diff = git_output("diff", "--cached", "--binary", binary=True)
    dataset_manifest = json.loads(
        resolved_manifest_json(concat.resolved_dataset_manifest)
    )
    record = {
        "version": 1,
        "stage": args.stage,
        "run_mode": args.run_mode,
        "git_head": git_output("rev-parse", "HEAD").strip(),
        "git_status_porcelain": git_output("status", "--porcelain=v1").splitlines(),
        "staged_diff_sha256": hashlib.sha256(staged_diff).hexdigest(),
        "launcher": {
            "path": str(launcher),
            "sha256": sha256_file(launcher),
        },
        "distributed_config": {
            "path": str(accelerate_config_path),
            "sha256": sha256_file(accelerate_config_path),
            "parsed": accelerate_config,
            "config_gradient_accumulation_steps": config_gas,
            "launcher_gradient_accumulation_steps": outer_gas,
            "train_cli_gradient_accumulation_steps": options.gradient_accumulation_steps,
            "config_micro_batch_size_per_gpu": config_micro_batch,
            "config_global_batch_size": config_global_batch,
            "config_world_size": config_world_size,
            "launcher_world_size": outer_world_size,
        },
        "model": {
            "path": str(model_path),
            "revision": "ModelScope master; immutable remote commit unavailable",
            "weight_file": (
                model_weight_identity["files"][0]["path"]
                if len(model_weight_identity["files"]) == 1
                else None
            ),
            "weight_sha256": model_weight_identity["aggregate_sha256"],
            "weight_identity": model_weight_identity,
            "config": json.loads(model_config_path.read_text(encoding="utf-8")),
        },
        "dataset_manifest": dataset_manifest,
        "dataset_manifest_sha256": dataset_manifest["content_hash"],
        "parsed_train_config": vars(options),
        "launch_command": command,
        "batch_contract": {
            "unique_frames_per_epoch": unique_frames,
            "aligned_presentations_per_epoch": aligned_presentations,
            "deterministic_duplicates_per_epoch": aligned_presentations - unique_frames,
            "world_size": world_size,
            "per_device_micro_batch": options.per_device_train_batch_size,
            "gradient_accumulation_steps": options.gradient_accumulation_steps,
            "nominal_global_batch": nominal_global_batch,
            "optimizer_steps_per_epoch": optimizer_steps_per_epoch,
            "samples_per_rank": per_rank_samples,
            "microbatches_per_rank": per_rank_microbatches,
            "final_microbatches_per_rank": final_microbatches_per_rank,
            "final_samples_per_rank": final_samples_per_rank,
            "final_global_batch": final_global_batch,
            "drop_last": False,
            "pad_final_optimizer_window": False,
        },
    }
    required_batch_contract = {
        "unique_frames_per_epoch": 310743,
        "aligned_presentations_per_epoch": 310744,
        "deterministic_duplicates_per_epoch": 1,
        "world_size": 4,
        "nominal_global_batch": 128,
        "optimizer_steps_per_epoch": 2428,
        "samples_per_rank": 77686,
        "final_samples_per_rank": 22,
        "final_global_batch": 88,
        "drop_last": False,
        "pad_final_optimizer_window": False,
    }
    if any(
        record["batch_contract"][key] != value
        for key, value in required_batch_contract.items()
    ):
        raise ValueError(f"unexpected batch contract: {record['batch_contract']}")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(record, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(record, ensure_ascii=True, sort_keys=True))


if __name__ == "__main__":
    main()
