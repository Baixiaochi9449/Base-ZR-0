#!/usr/bin/env python3
"""Six bounded production processes. Formal schedules are printable only."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "lerobot")]
STAGES = ("stage1_ar", "stage2_aux", "stage3_joint")


def validate_config(config):
    fixed = {"base_model": "/opt/data/private/lq/models/ZR-0", "num_gpus": 4,
        "per_device_train_batch_size": 16, "gradient_accumulation_steps": 2, "expected_global_batch_size": 128,
        "validation_updates_per_stage": 100, "validation_process_exit_steps": [50, 100],
        "validation_warmup_steps": 8, "max_consecutive_skipped_windows": 20, "num_difference_queries": 32,
        "num_flow_queries": 16, "action_horizon": 10, "max_pad_state_and_action_length": 64,
        "adam_epsilon": 1e-6, "warmup_ratio": .08, "wandb_failure_policy": "required"}
    fixed.update(seed=42, mixed_precision="bf16", zero_stage=2, vlm_attention_backend="sdpa",
        window_size=1, max_length=1024, adam_beta1=.9, adam_beta2=.95, weight_decay=.01,
        lr_scheduler="cosine", min_lr_rate=.1, gradient_clipping=1.,
        peak_learning_rates=dict(stage1_ar=1e-5, stage2_aux=2e-5, stage3_joint=1e-5))
    for key, expected in fixed.items():
        if config.get(key) != expected:
            raise ValueError(f"bounded validation fixed configuration mismatch: {key}")
    if config["loss_weights"] != {
        "stage1_ar": dict(ar=1., slot=0., optical_flow=0., fm=0.),
        "stage2_aux": dict(ar=0., slot=1., optical_flow=1., fm=0.),
        "stage3_joint": dict(ar=1., slot=1., optical_flow=1., fm=5.)}:
        raise ValueError("three-stage loss weights differ from the authorized contract")
    if [config["formal_command_templates_only"][s]["steps"] for s in STAGES] != [10000, 5000, 150000]:
        raise ValueError("formal command budgets differ from 10k/5k/150k")


def training_command(config, stage, stop, *, formal=False, formal_run=None):
    validate_config(config)
    from utils.three_stage_preflight import preparation_directory
    output = Path(config["output_root"])
    preparation = preparation_directory(config)
    if formal_run is not None and not formal:
        raise ValueError("formal execution settings cannot enter bounded validation")
    if stage not in STAGES or (not formal and stop not in (50, 100)):
        raise ValueError("unknown stage or validation milestone")
    stage_index = STAGES.index(stage)
    formal_resume = (formal_run or {}).get("stage1_resume_from_checkpoint") if stage_index == 0 else None
    resume = bool(formal_resume) or (not formal and stop == 100)
    run = output / ("formal_templates" if formal else "validation") / stage / ("run" if formal else f"to{stop}")
    if formal_run is not None:
        run = Path(formal_run["output_root"]) / stage
    if formal_resume:
        source = Path(formal_resume)
    elif resume:
        source = output / "validation" / stage / "to50/latest-model-optimizer-lr"
    elif stage_index:
        source = output / ("formal_templates" if formal else "validation") / STAGES[stage_index - 1] / (
            "run/latest-model-optimizer-lr" if formal else "to100/latest-model-optimizer-lr")
    else:
        source = Path(config["base_model"])
    if formal_run is not None and stage_index:
        source = Path(formal_run["output_root"]) / STAGES[stage_index - 1] / "latest-model-optimizer-lr"
    weights = config["loss_weights"][stage]
    run_id = config["experiment"] + ("-formal-template-" if formal else "-") + stage
    if formal_run is not None:
        run_id = formal_run["experiment"] + "-" + stage
    attempt = os.environ.get("ZR0_VALIDATION_ATTEMPT", "") if not formal else ""
    if attempt:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,31}", attempt):
            raise ValueError("validation attempt must be a short alphanumeric identifier")
        run_id += "-" + attempt
    expert_config = (source / "action_expert_config.json" if resume else
        Path(config["base_model"]) / "action_expert_config.json")
    schedule = config["formal_command_templates_only"][stage] if formal else dict(steps=100, save_step_interval=50)
    args = [config["python"], "-m", "accelerate.commands.launch", "--config_file", str(output / "accelerate.yaml"),
        "--num_processes", "4", "--mixed_precision", "bf16", "--gradient_accumulation_steps", "2", str(ROOT / "train_vla.py"),
        "--training_stage", stage, "--vlm_name_or_path", str(source),
        "--action_expert_config_path", str(expert_config),
        "--output_ckpt_dir", str(run), "--tensorboard_log_dir", str(run / "tensorboard"),
        "--max_train_steps", str(schedule["steps"]), "--save_step_interval", str(schedule["save_step_interval"]),
        "--epochs", str(config["epochs"]), "--logging_steps", "1", "--peak_learning_rate", str(config["peak_learning_rates"][stage]),
        "--vlm_loss_weight", str(weights["ar"]), "--action_expert_loss_weight", str(weights["fm"]),
        "--slot_loss_weight", str(weights["slot"]), "--optical_flow_loss_weight", str(weights["optical_flow"]),
        "--use_difference_query", "--component_optimizer_groups", "--log_training_diagnostics", "--save_optimizer_and_lr_states",
        "--fast_resume_data_skip",
        "--component_update_diagnostics", "inactive" if formal else "full",
        "--aux_dataset_config", str(preparation / "cached_aux_dataset_routes.json"),
        "--preparation_audit_cache", str(preparation / "audit_snapshot.json"),
        "--wandb_project", (formal_run or config)["wandb_project"], "--wandb_group", (formal_run or config)["experiment"],
        "--wandb_run_name", run_id, "--wandb_run_id", run_id,
        "--wandb_resume", "must" if resume else "never", "--wandb_dir", str(run / "wandb")]
    for key in ("per_device_train_batch_size", "gradient_accumulation_steps", "expected_global_batch_size", "seed",
                "num_difference_queries", "num_flow_queries", "window_size", "action_horizon", "max_pad_state_and_action_length",
                "max_length", "dataloader_num_workers", "prefetch_factor", "adam_beta1", "adam_beta2", "adam_epsilon", "lr_scheduler",
                "min_lr_rate", "warmup_ratio", "wandb_failure_policy", "vlm_attention_backend", "max_consecutive_skipped_windows"):
        args += ["--" + key, str(config[key])]
    args += ["--dataset_entries", *config["dataset_entries"], "--dataset_sample_ratios", "1", "1", "1", "1"]
    if formal:
        args += ["--batch_metric_reductions"]
    if stage != "stage2_aux":
        args += ["--tune_vlm"]
    if stage != "stage1_ar":
        args += ["--slot_aux_type", "structured_slots_v1", "--slot_supervision_dir", str(output),
                 "--optical_flow_aux_type", "dense_regression_v1", "--optical_flow_data_root", str(output),
                 "--flow_delta_frames", "20", "--stage2_aux_sampling", "slot_valid"]
    if resume:
        args += ["--resume_from_checkpoint", str(source)]
        if formal:
            args += ["--verify_resume_state"]
    elif stage_index:
        args += ["--init_from_checkpoint", str(source)]
    if stage == "stage3_joint":
        args += ["--tune_action_expert", "--action_expert_name_or_path", str(source if resume else Path(config["base_model"]))]
    if not formal:
        args += ["--bounded_three_stage_validation", "--save_and_exit_after_updates", str(stop),
                 "--three_stage_preparation_config", str(preparation / "requested_config.json")]
    else:
        args += ["--verify_three_stage_initialization"]
    return run, args


def record_run(output, run, **values):
    values.update(time=datetime.now(timezone.utc).isoformat(), output_dir=str(run))
    with (output / "validation_events.jsonl").open("a") as stream:
        stream.write(json.dumps(values, sort_keys=True) + "\n")
    for directory in (output, run):
        with (directory / "experiment.md").open("a") as stream:
            stream.write("\nRuntime record: `" + json.dumps(values, sort_keys=True) + "`\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/three_stage_validation_20260908.json")
    parser.add_argument("--mode", choices=("commands", "formal-commands", "validate"), default="commands")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    validate_config(config)
    if args.mode != "validate":
        for stage in STAGES:
            for stop in ((None,) if args.mode == "formal-commands" else (50, 100)):
                print(shlex.join(training_command(config, stage, stop, formal=args.mode == "formal-commands")[1]))
        return
    output = Path(config["output_root"])
    # This certificate is created only after every preparation phase succeeds.
    from utils.three_stage_preflight import validate_preparation
    validate_preparation(config)
    with (output / "validation_started.json").open("x") as stream:
        json.dump({"pid": os.getpid(), "maximum_successful_updates": [100, 100, 100],
            "validation_attempt": os.environ.get("ZR0_VALIDATION_ATTEMPT", "")}, stream)
    from utils.gpu_resource_gate import wait_for_gpus
    from scripts.run_stage05_h10_experiment import cleanup_group
    children = []
    for stage in STAGES:
        for stop in (50, 100):
            gate = wait_for_gpus(env={**os.environ, "CUDA_VISIBLE_DEVICES": "0,1,2,3"}, expected_count=4,
                children=children, process_groups=[child.pid for child in children], log_path=output / "gpu_gate.jsonl")
            run, command = training_command(config, stage, stop)
            run.mkdir(parents=True, exist_ok=False)
            (run / "experiment.md").write_text((output / "experiment.md").read_text() +
                "\nActual command:\n\n```bash\n" + shlex.join(command) + "\n```\n")
            from utils.cli_options import parse_train_options
            expanded = vars(parse_train_options(command[command.index(str(ROOT / "train_vla.py")) + 1:]))
            with (run / "expanded_options.json").open("x") as stream:
                json.dump(expanded, stream, indent=2, sort_keys=True)
            record_run(output, run, stage=stage, target_update=stop, status="starting", gpu_gate=gate)
            with (run / "train.log").open("x") as stream:
                child = subprocess.Popen(command, cwd=ROOT, env={**os.environ,
                    "CUDA_VISIBLE_DEVICES": gate["cuda_visible_devices"], "PYTHONNOUSERSITE": "1",
                    "PYTHONPATH": f"{ROOT}:{ROOT / 'lerobot'}", "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
                    "WANDB_MODE": "online", "LEROBOT_PYAV_THREADS": "1"},
                    stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                children.append(child)
                try:
                    code = child.wait()
                except BaseException:
                    cleanup_group(child.pid)
                    record_run(output, run, stage=stage, status="interrupted", checkpoint=None)
                    raise
            if code:
                cleanup_group(child.pid)
                record_run(output, run, stage=stage, status="failed", exit_code=code,
                           log=str(run / "train.log"), automatic_retry=False)
                raise RuntimeError(f"{stage} to{stop} failed; no retry: {run / 'train.log'}")
            metrics = [json.loads(line) for line in (run / "training_metrics.jsonl").read_text().splitlines()]
            updated = [m for m in metrics if m.get("optimizer_update_applied")]
            if len(updated) != 50 or updated[-1]["step"] != stop:
                raise RuntimeError("bounded validation successful-update count mismatch")
            checkpoint = run / "latest-model-optimizer-lr"
            for rank in range(4):
                expected = checkpoint / f"validation_resume_rank{rank}.pt" if stop == 50 else run / f"resume_verified_rank{rank}.json"
                if not expected.is_file():
                    raise RuntimeError(f"missing recovery evidence: {expected}")
                records = [json.loads(line) for line in (run / f"component_updates_rank{rank}.jsonl").read_text().splitlines()]
                applied = [row for row in records if row["global_update_applied"]]
                if len(applied) != 50 or applied[-1]["global_step_after"] != stop:
                    raise RuntimeError(f"rank {rank} component update evidence differs from the global budget")
            record_run(output, run, stage=stage, status="complete", completed_updates=stop,
                checkpoint=str(checkpoint), minimum_training_loss=min(m["total_loss"] for m in updated),
                peak_reserved_gib=max(m.get("gpu_peak_memory_reserved_gib", 0) for m in updated),
                wandb=json.loads((run / "wandb_identity.json").read_text()),
                final_metrics=updated[-1], resume_verified=stop == 100)
    result = {"status": "complete", "successful_updates": [100, 100, 100], "formal_training_started": False,
        "checkpoints": {stage: [str(output / "validation" / stage / f"to{step}/latest-model-optimizer-lr")
            for step in (50, 100)] for stage in STAGES}, "events": str(output / "validation_events.jsonl")}
    with (output / "validation_summary.json").open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
