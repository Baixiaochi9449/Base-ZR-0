#!/usr/bin/env python3
"""Attach recovery supervision to an existing formal run without restarting it."""

import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "lerobot")]
from scripts.run_three_stage_formal import load_plan, record, verify_finished_stage, write_json

TAG = "latest-model-optimizer-lr"


def process_identity(pid):
    try:
        fields = (Path("/proc") / str(pid) / "stat").read_text().rsplit(") ", 1)[1].split()
    except FileNotFoundError:
        return None
    if fields[0] == "Z":
        return None
    return {"pid": pid, "pgrp": int(fields[2]), "session": int(fields[3]), "start_ticks": int(fields[19])}


def last_metric(run):
    path = Path(run) / "training_metrics.jsonl"
    if not path.exists():
        return None
    with path.open("rb") as stream:
        stream.seek(max(0, path.stat().st_size - 262144))
        lines = stream.read().splitlines()
    for line in reversed(lines):
        try:
            row = json.loads(line)
        except (ValueError, UnicodeDecodeError):
            continue
        if row.get("optimizer_update_applied"):
            return row
    return None


def checkpoint_state(path):
    # This CPU-only inspector skips optional Triton imports during deserialization.
    # Training children exec a clean interpreter and retain the original stack.
    sys.modules["triton"] = None
    import torch
    return torch.load(path, map_location="cpu", weights_only=False, mmap=True)


def validate_checkpoint(path, stage, target):
    import numpy as np
    from utils.optical_flow_checkpoint import read_flow_artifacts
    from utils.stage05_checkpoint_contract import validate_action_expert_config_provenance
    from utils.three_stage_sources import weight_map
    path = Path(path)
    metadata = json.loads((path / "zr0_checkpoint_metadata.json").read_text())
    step = metadata["completed_optimizer_windows"]
    if metadata["training_stage"] != stage or type(step) is not int or not 0 < step <= target:
        raise ValueError("checkpoint stage/update count is outside the formal contract")
    validate_action_expert_config_provenance(path, metadata)
    read_flow_artifacts(path)
    weight_map(path)
    for name in ("difference_query.safetensors", "difference_query_config.json", "tokenizer.json",
                 "preprocessor_config.json", "resolved_dataset_manifest.json"):
        if not (path / name).is_file():
            raise ValueError(f"missing model/processor artifact: {name}")
    if stage != "stage1_ar" and not (path / "slot_head.safetensors").is_file():
        raise ValueError("missing Slot weights")
    if stage == "stage3_joint" and not (path / "action_expert.safetensors").is_file():
        raise ValueError("missing Action Expert weights")
    scheduler = checkpoint_state(path / "scheduler.pt")
    client = checkpoint_state(path / "mp_rank_00_model_states.pt")
    if (scheduler["last_epoch"] != step or scheduler["_step_count"] != step + 1 or
            client["last_global_step"] != step or client["dp_world_size"] != 4 or not client.get("module")):
        raise ValueError("checkpoint scheduler/client step or world size mismatch")
    expected_groups = {"difference_query"}
    if stage != "stage2_aux":
        expected_groups.add("vlm")
    if stage != "stage1_ar":
        expected_groups.update(("slot_aux", "optical_flow_aux"))
    if stage == "stage3_joint":
        expected_groups.add("action_expert")
    reference_runtime = None
    for rank in range(4):
        runtime = checkpoint_state(path / f"training_runtime_rank{rank}.pt")
        if runtime["global_step"] != step or runtime["world_size"] != 4 or not runtime.get("rng"):
            raise ValueError("checkpoint rank RNG/step/world size mismatch")
        contract = {key: runtime[key] for key in ("cursor", "sampler_contract")}
        if reference_runtime is not None and contract != reference_runtime:
            raise ValueError("checkpoint rank sampler/cursor mismatch")
        reference_runtime = contract
        zero = checkpoint_state(path / f"bf16_zero_pp_rank_{rank}_mp_rank_00_optim_states.pt")["optimizer_state_dict"]
        native = zero["base_optimizer_state"]
        groups = native["param_groups"]
        masters = zero["single_partition_of_fp32_groups"]
        paddings = zero["group_paddings"]
        if (zero["zero_stage"] != 2 or len(masters) != len(groups) or len(paddings) != len(groups) or
                zero["partition_count"] != [4] * len(groups) or
                {group["component"] for group in groups} != expected_groups):
            raise ValueError("checkpoint optimizer partitions/components mismatch")
        for group, master, padding in zip(groups, masters, paddings):
            if len(group["params"]) != 1 or master.numel() == 0 or type(padding) is not int or padding < 0:
                raise ValueError("checkpoint optimizer partition is empty")
            state = native["state"].get(group["params"][0], {})
            if state and (not {"step", "exp_avg", "exp_avg_sq"}.issubset(state) or
                    not 0 <= float(state["step"]) <= step or
                    state["exp_avg"].numel() != master.numel() + padding or
                    state["exp_avg_sq"].shape != state["exp_avg"].shape):
                raise ValueError("checkpoint Adam state is incomplete")
            if not state and group["component"] in {"vlm", "difference_query", "action_expert"}:
                raise ValueError("checkpoint shared/Expert optimizer state is absent")
    seen = json.loads((path / "data_seen_manifest.json").read_text())
    if seen["global_step"] != step:
        raise ValueError("checkpoint exposure state belongs to a different update")
    with np.load(path / "data_seen_state.npz", allow_pickle=False) as arrays:
        if len(arrays["seen"]) != 4 or int(arrays["seen"].sum()) != seen["total_seen"]:
            raise ValueError("checkpoint exposure counters are incomplete")
        for name in arrays.files:
            arrays[name]
    return step


def file_inventory(path):
    return {str(p.relative_to(path)): (p.stat().st_size, p.stat().st_mtime_ns)
            for p in sorted(path.rglob("*")) if p.is_file()}


def preserve_checkpoint(source, destination, stage, target):
    step = validate_checkpoint(source, stage, target)
    before = file_inventory(source)
    temporary = destination.parent / (".pending-" + uuid.uuid4().hex)
    temporary.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, temporary / TAG)
    if (before != file_inventory(source) or before != file_inventory(temporary / TAG) or
            validate_checkpoint(temporary / TAG, stage, target) != step):
        raise ValueError(f"checkpoint changed while copying; incomplete copy retained at {temporary}")
    write_json(temporary / "checkpoint_complete.json", {"step": step, "stage": stage,
        "source": str(source), "files": before})
    temporary.rename(destination)
    return destination / TAG


def candidate_checkpoint(item, archive, emit):
    root = Path(item["output_dir"])
    candidates = [root / TAG, *sorted(root.glob(f"retries/*/{TAG}")), *sorted(archive.glob(f"*/{TAG}"))]
    initial_resume = item.get("options", {}).get("resume_from_checkpoint")
    if initial_resume:
        candidates.append(Path(initial_resume))
    valid = []
    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            step = validate_checkpoint(candidate, item["stage"], item["updates"])
        except Exception as error:
            emit(status="checkpoint_rejected", checkpoint=str(candidate), error=repr(error))
            continue
        valid.append((step, (candidate / "scheduler.pt").stat().st_mtime_ns, str(candidate)))
    return Path(max(valid)[2]) if valid else None


def attempt_item(item, run, number, *, checkpoint=None, predecessor=None):
    result = copy.deepcopy(item)
    command = result["command"]
    def replace(name, value):
        flag = "--" + name
        if flag in command:
            command[command.index(flag) + 1] = str(value)
        else:
            command.extend((flag, str(value)))
    for name, value in (("output_ckpt_dir", run), ("tensorboard_log_dir", run / "tensorboard"),
                        ("wandb_dir", run / "wandb")):
        replace(name, value)
    for name in ("wandb_run_name", "wandb_run_id"):
        replace(name, item["options"][name] + f"-retry{number:03d}")
    replace("wandb_resume", "never")
    if checkpoint is not None:
        if "--init_from_checkpoint" in command:
            index = command.index("--init_from_checkpoint")
            del command[index:index + 2]
        replace("vlm_name_or_path", checkpoint)
        replace("action_expert_config_path", checkpoint / "action_expert_config.json")
        replace("resume_from_checkpoint", checkpoint)
        if item["stage"] == "stage3_joint":
            replace("action_expert_name_or_path", checkpoint)
    elif predecessor is not None:
        replace("vlm_name_or_path", predecessor)
        replace("init_from_checkpoint", predecessor)
    from utils.cli_options import parse_train_options
    result.update(output_dir=str(run), options=vars(parse_train_options(command[command.index(str(ROOT / "train_vla.py")) + 1:])))
    return result


def read_events(output):
    path = output / "formal_events.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


class RecoverySupervisor:
    def __init__(self, policy, plan):
        self.policy, self.plan = policy, plan
        self.output = Path(plan["formal"]["output_root"])
        self.archive = self.output / "recovery_checkpoints"
        self.identity = json.loads((self.output / "code_identity.json").read_text())["runtime"]
        self.copied = set()

    def emit(self, **values):
        run = Path(values["output_dir"]) if values.get("output_dir") else None
        record(self.output, run, retry_supervisor=True, **values)

    def check_runtime(self):
        from utils.three_stage_preflight import implementation_identity
        if implementation_identity() != self.identity:
            raise ValueError("runtime identity changed; automatic retries cannot change the training contract")
        if (self.output / "retry_disabled").exists():
            raise KeyboardInterrupt("automatic retries disabled by operator")

    def preserve(self, item, run):
        metric = last_metric(run)
        if metric is None:
            return
        source = Path(run) / TAG
        metadata = source / "zr0_checkpoint_metadata.json"
        if not metadata.exists():
            return
        try:
            step = json.loads(metadata.read_text())["completed_optimizer_windows"]
            key = (str(source), step)
            if key in self.copied or step > metric["step"]:
                return
            destination = self.archive / item["stage"] / (f"step-{step:06d}-" + Path(run).name)
            if not destination.exists():
                checkpoint = preserve_checkpoint(source, destination, item["stage"], item["updates"])
                self.emit(status="checkpoint_preserved", stage=item["stage"], step=step, checkpoint=str(checkpoint))
            self.copied.add(key)
        except Exception as error:
            self.emit(status="checkpoint_copy_deferred", source=str(source), error=repr(error))

    def watch_original(self, original):
        self.emit(status="watching_existing_training", original_supervisor=original,
            max_retries_per_stage=self.policy["max_retries_per_stage"], active_training_restarted=False)
        while original is not None and process_identity(original["pid"]) == original:
            if (self.output / "retry_disabled").exists():
                raise KeyboardInterrupt("automatic retries disabled by operator")
            for item in self.plan["stages"]:
                self.preserve(item, Path(item["output_dir"]))
            time.sleep(self.policy["poll_seconds"])
        events = read_events(self.output)
        if any(event.get("status") == "stopped" and "KeyboardInterrupt" in event.get("error", "")
               and not event.get("retry_supervisor") for event in events):
            raise KeyboardInterrupt("original supervisor was deliberately interrupted")
        if (self.output / "formal_summary.json").exists():
            self.emit(status="original_sequence_completed", retries_used=0)
            return
        self.recover(events)

    def recover(self, events):
        from utils.gpu_resource_gate import wait_for_gpus
        completed = {row["stage"]: row for row in events if row.get("status") == "complete" and not row.get("retry_supervisor")}
        predecessor = None
        for item in self.plan["stages"]:
            stage = item["stage"]
            if stage in completed:
                predecessor = Path(completed[stage]["checkpoint"])
                if validate_checkpoint(predecessor, stage, item["updates"]) != item["updates"]:
                    raise ValueError("completed predecessor checkpoint is below its formal target")
                continue
            failed_before = any(row.get("stage") == stage and row.get("status") in {"starting", "running", "stopped"}
                                and not row.get("retry_supervisor") for row in events)
            first = 1 if failed_before else 0
            failed_run = Path(item["output_dir"]) if failed_before else None
            for number in range(first, self.policy["max_retries_per_stage"] + 1):
                self.check_runtime()
                if number:
                    delay = self.policy["retry_delays_seconds"][number - 1]
                    self.emit(status="retry_wait", stage=stage, retry=number, seconds=delay)
                    for _ in range(delay):
                        if (self.output / "retry_disabled").exists():
                            raise KeyboardInterrupt("automatic retries disabled by operator")
                        time.sleep(1)
                child = None
                child_identity = None
                run = None
                try:
                    gate = wait_for_gpus(env={**os.environ, "CUDA_VISIBLE_DEVICES": "0,1,2,3"}, expected_count=4,
                        log_path=self.output / "retry_gpu_gate.jsonl")
                    source = candidate_checkpoint(item, self.archive / stage, lambda **v: self.emit(stage=stage, **v))
                    start = validate_checkpoint(source, stage, item["updates"]) if source else 0
                    run = Path(item["output_dir"]) / "retries" / f"attempt-{number:03d}"
                    run.mkdir(parents=True, exist_ok=False)
                    current = attempt_item(item, run, number, checkpoint=source, predecessor=predecessor)
                    write_json(run / "expanded_options.json", current["options"])
                    import shlex
                    (run / "experiment.md").write_text((Path(item["output_dir"]) / "experiment.md").read_text() +
                        f"\n## Recovery Attempt {number}\n\nResume checkpoint: `{source}`; initial logical update: {start}; target: {item['updates']}.\n\n" +
                        "```bash\n" + shlex.join(current["command"]) + "\n```\n")
                    logged = last_metric(failed_run) if failed_run else None
                    lost = max(0, int(logged["step"]) - start) if logged else 0
                    self.emit(status="retry_starting", stage=stage, retry=number, resume_step=start,
                        checkpoint=str(source) if source else None, output_dir=str(run), command=current["command"],
                        discarded_logged_updates=max(0, lost), unlogged_final_update_unknown=True)
                    failed_run = None
                    with (run / "train.log").open("x") as stream:
                        environment = {**os.environ, "CUDA_VISIBLE_DEVICES": gate["cuda_visible_devices"],
                            "PYTHONNOUSERSITE": "1", "PYTHONPATH": f"{ROOT}:{ROOT / 'lerobot'}",
                            "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "WANDB_MODE": "online", "LEROBOT_PYAV_THREADS": "1"}
                        child = subprocess.Popen(current["command"], cwd=ROOT, env=environment, stdout=stream,
                            stderr=subprocess.STDOUT, start_new_session=True)
                        child_identity = process_identity(child.pid)
                        self.emit(status="retry_running", stage=stage, retry=number, pid=child.pid,
                            identity=child_identity, output_dir=str(run))
                        while child.poll() is None:
                            if (self.output / "retry_disabled").exists():
                                raise KeyboardInterrupt("automatic retries disabled by operator")
                            self.preserve(item, run)
                            time.sleep(self.policy["poll_seconds"])
                    if child.returncode:
                        raise RuntimeError(f"training child exited with code {child.returncode}")
                    checkpoint = run / TAG if start < item["updates"] else source
                    validate_checkpoint(checkpoint, stage, item["updates"])
                    result = verify_finished_stage(current, start_step=start, checkpoint=checkpoint)
                    completed[stage] = {"stage": stage, **result}
                    self.emit(status="retry_stage_complete", stage=stage, retry=number, output_dir=str(run), **result)
                    predecessor = Path(result["checkpoint"])
                    break
                except Exception as error:
                    if child is not None and run is not None:
                        failed_run = run
                    self.emit(status="retry_failed", stage=stage, retry=number, error=repr(error),
                        output_dir=str(run) if run is not None and (run / "experiment.md").is_file() else None)
                    if number == self.policy["max_retries_per_stage"]:
                        raise RuntimeError(f"{stage}: automatic retry limit exhausted") from error
                finally:
                    if child is not None and child.poll() is None and child_identity == process_identity(child.pid):
                        os.killpg(child.pid, signal.SIGTERM)
                        try:
                            child.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            if child_identity == process_identity(child.pid):
                                os.killpg(child.pid, signal.SIGKILL)
                            child.wait()
        write_json(self.output / "formal_recovered_summary.json", {"status": "complete", "stages": completed})
        self.emit(status="recovered_sequence_completed", successful_updates=self.plan["formal"]["stage_updates"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/three_stage_formal_retry_20260909.json")
    parser.add_argument("--mode", choices=("check", "watch"), default="check")
    args = parser.parse_args()
    policy = json.loads(args.config.read_text())
    if (policy.get("version") != 1 or policy.get("enabled") is not True or
            policy.get("max_retries_per_stage") != 3 or policy.get("retry_delays_seconds") != [60, 120, 240] or
            policy.get("poll_seconds") != 30 or policy.get("preserve_complete_checkpoints") is not True or
            not policy.get("authorization")):
        raise ValueError("unsupported or disabled retry policy")
    plan = load_plan(policy["formal_config"])
    output = Path(plan["formal"]["output_root"])
    if plan != json.loads((output / "launch_plan.json").read_text()):
        raise ValueError("formal plan changed; retries cannot alter the sealed launch plan")
    supervisor = RecoverySupervisor(policy, plan)
    supervisor.check_runtime()
    pid = json.loads((output / "formal_started.json").read_text())["pid"]
    original = process_identity(pid)
    if original is not None:
        command = (Path("/proc") / str(pid) / "cmdline").read_bytes().split(b"\0")
        if not any(Path(os.fsdecode(arg)).name == "run_three_stage_formal.py" for arg in command if arg):
            raise ValueError("saved supervisor PID now belongs to a different process")
    if args.mode == "check":
        print(json.dumps({"status": "ready", "policy": policy, "original_supervisor": original}))
        return
    def terminate(signum, frame):
        raise KeyboardInterrupt(f"retry supervisor received signal {signum}")
    signal.signal(signal.SIGTERM, terminate)
    with (output / "retry_watch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        from utils.stage05_sidecar import sha256_file
        write_json(output / "retry_watch_started.json", {"pid": os.getpid(), "identity": process_identity(os.getpid()),
            "policy": policy, "original_supervisor": original,
            "implementation_sha256": sha256_file(Path(__file__)),
            "formal_helper_sha256": sha256_file(ROOT / "scripts/run_three_stage_formal.py")})
        try:
            supervisor.watch_original(original)
        except BaseException as error:
            supervisor.emit(status="retry_supervisor_stopped", error=repr(error))
            raise


if __name__ == "__main__":
    main()
