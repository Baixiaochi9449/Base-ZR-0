#!/usr/bin/env python3
"""Launch the explicitly authorized formal sequence using the saved preparation."""

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "lerobot")]

from scripts.run_three_stage_validation import STAGES, training_command, validate_config


def load_plan(path):
    formal = json.loads(Path(path).read_text())
    config = json.loads(Path(formal["preparation_config"]).read_text())
    validate_config(config)
    if formal.get("version") != 1 or formal.get("stage_updates") != [10000, 5000, 150000]:
        raise ValueError("formal execution requires the authorized 10k/5k/150k budgets")
    if not formal.get("authorization") or not formal.get("wandb_project"):
        raise ValueError("formal execution requires recorded authorization and online W&B")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", formal["experiment"]):
        raise ValueError("invalid formal experiment identifier")
    output = Path(formal["output_root"]).resolve()
    for protected in (Path(config["output_root"]).resolve(), Path(config["base_model"]).resolve()):
        if output.is_relative_to(protected) or protected.is_relative_to(output):
            raise ValueError("formal output must be independent of preparation, validation and base")
    stages = []
    from utils.cli_options import parse_train_options
    for stage, updates, warmup, interval in zip(STAGES, formal["stage_updates"], (800, 400, 12000), (1000, 1000, 2000)):
        schedule = config["formal_command_templates_only"][stage]
        if schedule != dict(steps=updates, warmup_steps=warmup, save_step_interval=interval):
            raise ValueError("formal schedule differs from the authorized contract")
        run, command = training_command(config, stage, None, formal=True, formal_run=formal)
        options = vars(parse_train_options(command[command.index(str(ROOT / "train_vla.py")) + 1:]))
        if options["bounded_three_stage_validation"] or options["save_and_exit_after_updates"] is not None:
            raise ValueError("formal training cannot use validation execution limits")
        item = dict(stage=stage, updates=updates, output_dir=str(run), command=command, options=options)
        if options["resume_training"]:
            source = Path(options["resume_from_checkpoint"]).resolve()
            if source.is_relative_to(output) or output.is_relative_to(source):
                raise ValueError("formal resume output must preserve its source checkpoint")
            metadata = json.loads((source / "zr0_checkpoint_metadata.json").read_text())
            start = metadata["completed_optimizer_windows"]
            if metadata["training_stage"] != stage or type(start) is not int or not 0 < start < updates:
                raise ValueError("formal resume checkpoint stage/update count mismatch")
            item["resume_step"] = start
        stages.append(item)
    return {"formal": formal, "preparation": config, "stages": stages}


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)


def prepare(plan):
    output = Path(plan["formal"]["output_root"])
    template = ROOT / "docs/experiments/three_stage_formal_20260909/experiment.md"
    document = template.read_text()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "launch_plan.json", plan)
    from utils.three_stage_preflight import implementation_identity
    from utils.stage05_sidecar import sha256_file
    write_json(output / "code_identity.json", {"runtime": implementation_identity(),
        "formal_launcher_sha256": sha256_file(Path(__file__)),
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "git_status": subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True)})
    (output / "experiment.md").write_text(document)
    for item in plan["stages"]:
        run = Path(item["output_dir"])
        run.mkdir()
        write_json(run / "expanded_options.json", item["options"])
        (run / "experiment.md").write_text(document + "\n## This Process\n\n" +
            f"Stage: {item['stage']}; output: `{run}`; target successful updates: {item['updates']}.\n\n" +
            "Complete command (environment and UUID selection are recorded at launch):\n\n```bash\n" +
            shlex.join(item["command"]) + "\n```\n\nAll parser defaults: `expanded_options.json`.\n")
    return output


def record(output, run=None, **values):
    values.update(time=datetime.now(timezone.utc).isoformat())
    with (output / "formal_events.jsonl").open("a") as stream:
        stream.write(json.dumps(values, sort_keys=True) + "\n")
    for directory in (output,) if run is None else (output, run):
        with (directory / "experiment.md").open("a") as stream:
            stream.write("\nRuntime record: `" + json.dumps(values, sort_keys=True) + "`\n")
    print(json.dumps(values, sort_keys=True), flush=True)


def verify_finished_stage(item, *, start_step=0, checkpoint=None):
    import torch
    run = Path(item["output_dir"])
    count, minimum, peak = start_step, math.inf, 0.
    final = None
    with (run / "training_metrics.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            if not row.get("optimizer_update_applied"):
                continue
            count += 1
            if row["step"] != count or not math.isfinite(row["total_loss"]):
                raise RuntimeError("formal update count or finite-loss contract failed")
            minimum = min(minimum, row["total_loss"])
            peak = max(peak, row.get("gpu_peak_memory_reserved_gib", 0.))
            final = row
    if count != item["updates"]:
        raise RuntimeError(f"formal stage stopped at {count}/{item['updates']} updates")
    checkpoint = Path(checkpoint) if checkpoint is not None else run / "latest-model-optimizer-lr"
    metadata = json.loads((checkpoint / "zr0_checkpoint_metadata.json").read_text())
    if metadata["training_stage"] != item["stage"]:
        raise RuntimeError("formal checkpoint stage mismatch")
    if torch.load(checkpoint / "scheduler.pt", map_location="cpu", weights_only=True)["last_epoch"] != count:
        raise RuntimeError("formal scheduler/checkpoint step mismatch")
    for rank in range(4):
        runtime = torch.load(checkpoint / f"training_runtime_rank{rank}.pt", map_location="cpu", weights_only=True)
        if runtime["global_step"] != count or runtime["world_size"] != 4:
            raise RuntimeError("formal rank runtime/checkpoint step mismatch")
        if not (checkpoint / f"bf16_zero_pp_rank_{rank}_mp_rank_00_optim_states.pt").is_file():
            raise RuntimeError("formal checkpoint missing an optimizer partition")
    return dict(completed_updates=count, checkpoint=str(checkpoint), minimum_training_loss=minimum if final else None,
        peak_reserved_gib=peak, final_metrics=final,
        wandb=json.loads((run / "wandb_identity.json").read_text()))


def execute(plan):
    from utils.gpu_resource_gate import wait_for_gpus
    from utils.three_stage_preflight import implementation_identity, validate_preparation
    from utils.stage05_sidecar import sha256_file
    from scripts.run_stage05_h10_experiment import cleanup_group
    output = Path(plan["formal"]["output_root"])
    if json.loads((output / "launch_plan.json").read_text()) != plan:
        raise ValueError("formal launch plan changed after preparation")
    identity = json.loads((output / "code_identity.json").read_text())
    if identity["runtime"] != implementation_identity() or identity["formal_launcher_sha256"] != sha256_file(Path(__file__)):
        raise ValueError("formal launch code changed after preparation")
    for item in plan["stages"]:
        run = Path(item["output_dir"])
        if not (run / "experiment.md").is_file() or json.loads((run / "expanded_options.json").read_text()) != item["options"]:
            raise ValueError("formal experiment documentation or expanded options changed")
    write_json(output / "formal_started.json", {"pid": os.getpid(), "time": datetime.now(timezone.utc).isoformat(),
        "authorized_updates": plan["formal"]["stage_updates"], "authorization": plan["formal"]["authorization"]})
    children, completed = [], []
    run = None
    stage = None
    try:
        record(output, status="checking_saved_preparation", source_payloads_reaudited=False)
        validate_preparation(plan["preparation"])
        for item in plan["stages"]:
            run, stage = Path(item["output_dir"]), item["stage"]
            if identity["runtime"] != implementation_identity():
                raise ValueError("runtime implementation changed during the formal sequence")
            if item.get("resume_step"):
                from scripts.watch_three_stage_formal import validate_checkpoint
                source = item["options"]["resume_from_checkpoint"]
                if validate_checkpoint(source, stage, item["updates"]) != item["resume_step"]:
                    raise ValueError("formal resume checkpoint changed after preparation")
            gate = wait_for_gpus(env={**os.environ, "CUDA_VISIBLE_DEVICES": "0,1,2,3"}, expected_count=4,
                children=children, process_groups=[child.pid for child in children], log_path=output / "gpu_gate.jsonl")
            environment = {"CUDA_VISIBLE_DEVICES": gate["cuda_visible_devices"], "PYTHONNOUSERSITE": "1",
                "PYTHONPATH": f"{ROOT}:{ROOT / 'lerobot'}", "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
                "WANDB_MODE": "online", "LEROBOT_PYAV_THREADS": "1"}
            record(output, run, stage=stage, status="starting", target_updates=item["updates"], gpu_gate=gate,
                environment=environment, command=item["command"])
            with (run / "train.log").open("x") as stream:
                child = subprocess.Popen(item["command"], cwd=ROOT, env={**os.environ, **environment},
                    stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                children.append(child)
                record(output, run, stage=stage, status="running", pid=child.pid)
                code = child.wait()
            if code:
                raise RuntimeError(f"{stage} failed with exit code {code}; no automatic retry")
            result = verify_finished_stage(item, start_step=item.get("resume_step", 0))
            record(output, run, stage=stage, status="complete", **result)
            completed.append({"stage": stage, **result})
        write_json(output / "formal_summary.json", {"status": "complete", "stages": completed})
    except BaseException as error:
        for child in children:
            cleanup_group(child.pid)
        record(output, run, stage=stage, status="stopped", error=repr(error), automatic_retry=False,
            completed_stages=[item["stage"] for item in completed])
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/three_stage_formal_20260909.json")
    parser.add_argument("--mode", choices=("prepare", "run"), default="prepare")
    args = parser.parse_args()
    plan = load_plan(args.config)
    if args.mode == "prepare":
        print(prepare(plan))
    else:
        def terminate(signum, frame):
            raise KeyboardInterrupt(f"formal launcher received signal {signum}")
        signal.signal(signal.SIGTERM, terminate)
        execute(plan)


if __name__ == "__main__":
    main()
