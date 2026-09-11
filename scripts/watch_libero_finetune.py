"""Bounded recovery for the explicitly authorized step-14000 LIBERO experiment."""

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
import uuid

from train_libero_finetune import TAG, inventory

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "3eefb602417d3bd4b20bef2b47660b404aefb565"
RUNTIME = ROOT / "outputs/runtime_snapshots" / COMMIT
OUTPUT = ROOT / "outputs/ckpts/ZR0-stage3-step14000-LIBERO-action-only-dq32-h10-gbs64-seed42"
SOURCE = ROOT / ("outputs/three_stage_formal_20260910/stage3_resume8000_slot05_flow5/"
                 "recovery_checkpoints/stage3_joint/step-014000-attempt-000") / TAG
TARGET = 34184


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def append_event(output, **values):
    event = dict(time=timestamp(), **values)
    with (output / "supervisor_events.jsonl").open("a") as stream:
        stream.write(json.dumps(event, sort_keys=True) + "\n")
    with (output / "experiment.md").open("a") as stream:
        stream.write(f"\nRuntime event: `{json.dumps(event, sort_keys=True)}`\n")
    print(json.dumps(event, sort_keys=True), flush=True)


def code_identity():
    files = [*RUNTIME.rglob("*.py"), *RUNTIME.rglob("*.yaml"),
             ROOT / "scripts/run_libero_wo_ecot_pt.sh",
             ROOT / "scripts/record_libero_finetune_launch.py", Path(__file__),
             ROOT / "scripts/train_libero_finetune.py",
             ROOT / "accelerate_configs/libero_zero2_bf16_mbs16_gas1.yaml"]
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(files))}


def latest_complete(output):
    valid = []
    for receipt in output.glob("attempt-*/recovery_checkpoints/step-*/checkpoint_complete.json"):
        record = json.loads(receipt.read_text())
        checkpoint = receipt.parent / TAG
        if (record["checkpoint_kind"] != "action_only" or not 0 < record["step"] <= TARGET
                or inventory(checkpoint) != record["files"]):
            raise ValueError(f"immutable LIBERO checkpoint is inconsistent: {receipt}")
        valid.append((record["step"], str(checkpoint)))
    return Path(max(valid)[1]) if valid else None


def validate_resume(checkpoint):
    # Deserialization is CPU-only and used only after a failed process exits.
    sys.modules["triton"] = None
    import torch
    def load(name):
        return torch.load(checkpoint / name, map_location="cpu", weights_only=False, mmap=True)
    receipt = json.loads((checkpoint.parent / "checkpoint_complete.json").read_text())
    step = receipt["step"]
    metadata = json.loads((checkpoint / "zr0_checkpoint_metadata.json").read_text())
    scheduler, client = load("scheduler.pt"), load("mp_rank_00_model_states.pt")
    if (metadata["checkpoint_kind"] != "action_only" or client["last_global_step"] != step
            or client["dp_world_size"] != 4 or not client.get("module")
            or scheduler["last_epoch"] != step * 4 or scheduler["_step_count"] != step * 4 + 1):
        raise ValueError("LIBERO checkpoint model/scheduler/update contract mismatch")
    for rank in range(4):
        zero = load(f"bf16_zero_pp_rank_{rank}_mp_rank_00_optim_states.pt")["optimizer_state_dict"]
        native = zero["base_optimizer_state"]
        groups = native["param_groups"]
        if zero["zero_stage"] != 2 or zero["partition_count"] != [4] or len(groups) != 1:
            raise ValueError("LIBERO checkpoint must contain four single-group ZeRO-2 partitions")
        group = groups[0]
        if len(group["params"]) != 1:
            raise ValueError("LIBERO checkpoint optimizer partition mismatch")
        state = native["state"][group["params"][0]]
        master = zero["single_partition_of_fp32_groups"][0]
        if (int(state["step"]) != step or state["exp_avg"].numel() != master.numel() + zero["group_paddings"][0]
                or state["exp_avg_sq"].shape != state["exp_avg"].shape):
            raise ValueError("LIBERO checkpoint Adam state is incomplete")
    return step


def wandb_failure(log):
    with log.open("rb") as stream:
        stream.seek(max(0, log.stat().st_size - 262144))
        tail = stream.read().decode(errors="replace")
    return bool(re.search(r"(?:wandb|W&B)[^\n]*(?:ERROR|failed|failure|timed out)|"
                          r"(?:CommError|AuthenticationError|UsageError):[^\n]*(?:wandb|W&B)", tail))


def finish_group(child):
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        try:
            os.killpg(child.pid, 0)
        except ProcessLookupError:
            return
        time.sleep(1)
    try:
        os.killpg(child.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def run(output):
    sys.path[:0] = [str(RUNTIME), str(RUNTIME / "lerobot")]
    from utils.gpu_resource_gate import wait_for_gpus

    if inventory(SOURCE) != json.loads((SOURCE.parent / "checkpoint_complete.json").read_text())["files"]:
        raise ValueError("source step-14000 archive changed")
    identity = code_identity()
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(ROOT / "docs/experiments/libero_stage3_step14000/experiment.md", output / "experiment.md")
    (output / "code_identity.json").write_text(json.dumps(identity, indent=2) + "\n")
    (output / "retry_policy.json").write_text(json.dumps(dict(max_retries=3, delays=[60, 120, 240],
        source=str(SOURCE), total_steps=TARGET, runtime_commit=COMMIT,
        fresh_restart_after_optimizer_started=False, wandb_required=True), indent=2) + "\n")
    run_id = uuid.uuid4().hex[:8]
    run_name = "zr0-step14000-libero-dq32-seed42-" + datetime.now().strftime("%Y%m%d-%H%M%S")
    checkpoint = None
    env = os.environ.copy()
    env.update(PYTHONNOUSERSITE="1", CONDA_DEFAULT_ENV="ZR-0", WANDB_MODE="online",
               CUDA_VISIBLE_DEVICES="0,1,2,3",
               PYTHONPATH=f"{RUNTIME}:{RUNTIME / 'lerobot'}", ZR0_RUNTIME_ROOT=str(RUNTIME),
               ZR0_TRAIN_PYTHON=sys.executable, ZR0_PRETRAIN_JOINT_CKPT=str(SOURCE),
               OMP_NUM_THREADS="1", TOKENIZERS_PARALLELISM="false",
               PATH=str(Path(sys.executable).parent) + ":" + env.get("PATH", ""))
    for name in ("ZR0_MAX_TRAIN_STEPS", "ZR0_DRY_RUN", "ZR0_RESUME_CKPT"):
        env.pop(name, None)
    env.update(ZR0_SAVE_STEP_INTERVAL="2000", ZR0_LIBERO_ACTION_HORIZON="10")
    for attempt in range(4):
        if (output / "retry_disabled").exists() or code_identity() != identity:
            raise RuntimeError("retry disabled or runtime/launcher identity changed")
        if attempt:
            append_event(output, status="retry_wait", attempt=attempt, seconds=60 * 2 ** (attempt - 1))
            deadline = time.monotonic() + 60 * 2 ** (attempt - 1)
            while time.monotonic() < deadline:
                if (output / "retry_disabled").exists():
                    raise RuntimeError("retry disabled")
                time.sleep(max(0, min(1, deadline - time.monotonic())))
        gate = wait_for_gpus(env=env, expected_count=4, log_path=output / "gpu_gate.jsonl")
        env["ZR0_CUDA_VISIBLE_DEVICES"] = gate["cuda_visible_devices"]
        env["CUDA_VISIBLE_DEVICES"] = gate["cuda_visible_devices"]
        destination = output / f"attempt-{attempt:03d}"
        mode = "resume" if checkpoint else "train"
        env.update(ZR0_OUTPUT_DIR=str(destination), ZR0_WANDB_RUN_ID=run_id, ZR0_RUN_NAME=run_name)
        if checkpoint:
            validate_resume(checkpoint)
            env["ZR0_RESUME_CKPT"] = str(checkpoint)
        elif attempt:
            run_id = uuid.uuid4().hex[:8]
            env["ZR0_WANDB_RUN_ID"] = run_id
        command = ["bash", str(ROOT / "scripts/run_libero_wo_ecot_pt.sh"), mode, "difference_query_stage3"]
        log = output / f"attempt-{attempt:03d}.log"
        append_event(output, status="launching", attempt=attempt, mode=mode, source=str(checkpoint or SOURCE),
                     output_dir=str(destination), log=str(log), command=command)
        with log.open("x") as stream:
            child = subprocess.Popen(command, env=env, cwd=RUNTIME, stdin=subprocess.DEVNULL,
                                     stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        append_event(output, status="running", attempt=attempt, pid=child.pid)
        try:
            while child.poll() is None:
                if (output / "retry_disabled").exists():
                    finish_group(child)
                    raise RuntimeError("operator disabled training/retries")
                time.sleep(10)
            returncode = child.wait()
        finally:
            finish_group(child)
        append_event(output, status="process_exited", attempt=attempt, exit_code=returncode)
        checkpoint = latest_complete(output)
        if checkpoint and validate_resume(checkpoint) == TARGET:
            append_event(output, status="complete", step=TARGET, checkpoint=str(checkpoint), exit_code=returncode)
            return
        if returncode == 0:
            raise RuntimeError("trainer exited successfully without a complete final LIBERO checkpoint")
        if wandb_failure(log):
            raise RuntimeError("W&B failure: formal training stopped with online recording required")
        if checkpoint is None and any(output.glob("attempt-*/optimizer_windows_started")):
            raise RuntimeError("training started without a complete own checkpoint; refusing a fresh reset")
    raise RuntimeError("LIBERO recovery exhausted three retries")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output-root", type=Path, default=OUTPUT)
    options = parser.parse_args()
    if not options.execute:
        print(json.dumps(dict(source=str(SOURCE), output=str(options.output_root),
                              total_steps=TARGET, runtime=str(RUNTIME), execute=False), indent=2))
        return
    options.output_root.parent.mkdir(parents=True, exist_ok=True)
    with (options.output_root.parent / (options.output_root.name + ".lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            run(options.output_root)
        except Exception as error:
            if (options.output_root / "experiment.md").is_file():
                append_event(options.output_root, status="stopped", error=repr(error))
            raise


if __name__ == "__main__":
    main()
