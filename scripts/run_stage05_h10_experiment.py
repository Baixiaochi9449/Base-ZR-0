#!/usr/bin/env python3
"""Sequential, failure-stopping launch of the authorized H10 experiment."""

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PYTHON = "/opt/data/private/lq/miniconda3/envs/ZR-0/bin/python"
LAUNCHER = ROOT / "scripts/run_stage05_four_dataset_pretraining.sh"
OWNED_CHILDREN = []


class ProbeOOM(RuntimeError):
    pass


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def record(output, stage, **details):
    event = {"time": datetime.now(timezone.utc).isoformat(), "stage": stage, **details}
    with (output / "lifecycle.jsonl").open("a") as stream:
        stream.write(json.dumps(event, sort_keys=True) + "\n")
    with (output / "experiment.md").open("a") as stream:
        stream.write("\n- Runtime: `" + json.dumps(event, sort_keys=True) + "`\n")
    if details.get("output_dir"):
        child_doc = Path(details["output_dir"]) / "experiment.md"
        if child_doc.is_file():
            with child_doc.open("a") as stream:
                stream.write("\n- Runtime: `" + json.dumps(event, sort_keys=True) + "`\n")
    print(json.dumps(event, sort_keys=True), flush=True)


def resource_gate(output=None, env=None):
    from utils.gpu_resource_gate import wait_for_gpus

    env = dict(os.environ if env is None else env)
    env["CUDA_VISIBLE_DEVICES"] = env.get("ZR0_CUDA_VISIBLE_DEVICES", "0,1,2,3")
    result = wait_for_gpus(env=env, expected_count=4, children=OWNED_CHILDREN,
                          process_groups=[child.pid for child in OWNED_CHILDREN],
                          log_path=Path(output or ROOT / "outputs") / "gpu_gate.jsonl")
    if shutil.disk_usage(ROOT / "outputs").free < 1024**4:
        raise RuntimeError("less than the reserved 1 TiB checkpoint budget is available")
    return result


def source_identity():
    paths = [ROOT / "train_vla.py", LAUNCHER, Path(__file__).resolve(),
             ROOT / "scripts/stage05_experiment_train.py",
             ROOT / "lerobot/lerobot/common/datasets/video_utils.py"]
    for directory in ("model", "utils"):
        paths.extend(sorted((ROOT / directory).glob("*.py")))
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def cleanup_group(pid):
    # The child is started with its own session. Never select unrelated PIDs.
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    time.sleep(3)
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def select_token_audit(output, processor_path, env):
    from scripts import audit_stage05_token_lengths as audit_tool

    processor_path = Path(processor_path).resolve()
    view = output / "config_view"
    base_report = output / "token_audit_h10_format2.json"
    base_spec = output / "trusted_spec.json"
    candidates = [(base_report, base_spec)]
    candidates.extend((path / "report.json", path / "trusted_spec.json")
                      for path in sorted((output / "processor_audits").glob("audit-*")))
    processor_files = audit_tool._processor_files(processor_path)
    reports, errors = [], []
    for report_path, spec_path in candidates:
        report = json.loads(report_path.read_text())
        audit_tool._validate_trusted_spec(report_path, report, spec_path)
        reports.append((report_path, report))
        if report["processor_identity"]["model_files"] != processor_files:
            continue
        try:
            audit_tool.validate_token_audit(
                report_path, processor_path=processor_path, max_length=int(env["ZR0_MAX_LENGTH"]),
                repository_root=view, trusted_spec_path=spec_path,
            )
        except ValueError as error:
            errors.append({"report": str(report_path), "error": str(error)})
            continue
        record(output, "processor_audit", status="reused", processor_path=str(processor_path),
               report=str(report_path), trusted_spec=str(spec_path))
        return {**env, "ZR0_TOKEN_LENGTH_AUDIT": str(report_path), "ZR0_TOKEN_AUDIT_SPEC": str(spec_path)}

    # Reuse failure is diagnostic only. Further full audits require authorization.
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(processor_path, local_files_only=True)
    runtime = audit_tool._processor_runtime_identity(processor, processor_path)
    data = audit_tool._data_identities(repository_root=view)
    implementation = audit_tool._repository_implementation_identity(view)
    differences = []
    for report_path, report in reports:
        old_runtime = report["processor_identity"]
        old_files = {item["path"]: item for item in old_runtime["model_files"]}
        new_files = {item["path"]: item for item in processor_files}
        fields = {f"processor_identity.model_files.{name}": {"old": old_files.get(name), "new": new_files.get(name)}
                  for name in sorted(old_files.keys() | new_files.keys()) if old_files.get(name) != new_files.get(name)}
        runtime_changes = {key: {"old": old_runtime.get(key), "new": runtime.get(key)}
                           for key in (old_runtime.keys() | runtime.keys()) - {"audited_processor_path", "model_files"}
                           if old_runtime.get(key) != runtime.get(key)}
        fields.update({f"processor_identity.{key}": value for key, value in runtime_changes.items()})
        for key, current in (("data_identity", data), ("implementation_identity", implementation)):
            if report.get(key) != current:
                fields[key] = {"old": report.get(key), "new": current}
        differences.append({"report": str(report_path), "changed_fields": fields,
                            "data_eligibility_identity_changed": report.get("data_identity") != data,
                            "loaded_processor_runtime_identity_changed": bool(runtime_changes)})
    record(output, "processor_audit", status="reuse_rejected", processor_path=str(processor_path),
           validation_errors=errors, differences=differences,
           scope="File identity changes alone do not establish tokenization behavior changes; no new audit was started.")
    raise RuntimeError("No valid existing processor audit; identity differences recorded in lifecycle.jsonl. No full audit started.")


def launch(output, stage, env, *, probe=False):
    gate = resource_gate(output, env)
    env = {**env, "CUDA_VISIBLE_DEVICES": gate["cuda_visible_devices"],
           "ZR0_CUDA_VISIBLE_DEVICES": gate["cuda_visible_devices"]}
    expected = json.loads(Path(env.get("ZR0_RUNNER_SOURCE_IDENTITY", output / "source_identity.json")).read_text())
    if source_identity() != expected:
        raise RuntimeError("training source files changed after this experiment was launched")
    if env.get("ZR0_SELECT_PROCESSOR_AUDIT") == "1":
        env = select_token_audit(output, env["ZR0_INITIAL_CHECKPOINT"], env)
        resource_gate(output, env)
    run_dir = Path(env["ZR0_RUN_OUTPUT_DIR"])
    suffix = f".{env['ZR0_CONTINUATION_ID']}" if env.get("ZR0_CONTINUATION_ID") else ""
    outer_log = output / (run_dir.relative_to(output).as_posix().replace("/", "_") + f".{stage}{suffix}.log")
    command = ["bash", str(LAUNCHER), stage]
    selected_env = {key: value for key, value in env.items() if key.startswith("ZR0_")}
    record(output, stage, status="starting", output_dir=str(run_dir), log=str(outer_log),
           command=shlex.join(command), environment=selected_env)
    with outer_log.open("x") as stream:
        child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream,
                                 stderr=subprocess.STDOUT, start_new_session=True)
        OWNED_CHILDREN.append(child)
        record(output, stage, status="running", pid=child.pid, process_group=child.pid)
        try:
            code = child.wait()
        except BaseException:
            cleanup_group(child.pid)
            raise
    if code:
        cleanup_group(child.pid)
        text = outer_log.read_text(errors="replace")
        oom = "CUDA out of memory" in text and (
            "OutOfMemoryError" in text or "RuntimeError: CUDA out of memory" in text
        )
        record(output, stage, status="failed", exit_code=code, confirmed_cuda_oom=oom, log=str(outer_log), output_dir=str(run_dir))
        if probe and oom:
            raise ProbeOOM(str(outer_log))
        raise RuntimeError(f"{stage} failed (exit {code}): {outer_log}")
    record(output, stage, status="exited", exit_code=0, output_dir=str(run_dir))


def make_fixture(output, phase, audit_report=None):
    import numpy as np
    import yaml
    registry = yaml.safe_load((output / "config_view/dataset2feature.yaml").read_text())
    audit = json.loads((audit_report or output / "token_audit_h10_format2.json").read_text())
    samples = []
    for name, measured in audit["datasets"].items():
        entry_name = f"stage05_{name}_mixed"
        sidecar = Path(registry[entry_name][f"{phase}_sidecar_path"])
        indices = np.load(sidecar / ("ar_indices.npy" if phase == "ar" else "joint_indices.npy"), mmap_mode="r")
        maximum = measured["maximum_sample"]
        index = int(maximum["global_index"])
        position = int(np.searchsorted(indices, index))
        if position >= len(indices) or int(indices[position]) != index or maximum["image_count"] != 2:
            raise RuntimeError(f"audited maximum is not an eligible two-view {phase} sample: {entry_name}")
        samples.append({"dataset_entry": entry_name, "global_index": index,
                        "reason": "audited_longest_two_view", "tokens": maximum["tokens"]})
        if (phase == "joint" and name == "droid") or name == "rh20t":
            packed = np.load(sidecar / "validity_packed.npy", mmap_mode="r")
            flag = 2 if name == "droid" else 1
            for begin in range(0, len(indices), 4096):
                block = indices[begin:begin + 4096].astype(np.int64)
                valid = (packed[block // 8, flag] >> (7 - block % 8)) & 1
                positions = np.flatnonzero(valid == 0)
                if len(positions):
                    samples.append({"dataset_entry": entry_name, "global_index": int(block[positions[0]]),
                                    "reason": "textless_fm" if name == "droid" else "wrist_fallback"})
                    break
            else:
                raise RuntimeError(f"missing representative {phase} {name} fallback sample")
    path = output / f"{phase}_probe_samples.json"
    write_json(path, samples)
    return path


def verify_run(output, run_dir, expected_step, phase, env):
    last = None
    peak = 0.0
    with (run_dir / "training_metrics.jsonl").open() as stream:
        for line in stream:
            item = json.loads(line)
            if "total_loss" not in item:
                continue
            for key in ("total_loss", "ar_loss", "flow_matching_loss", "learning_rate",
                        "vlm_grad_norm", "difference_query_grad_norm", "action_expert_grad_norm"):
                if key in item and not math.isfinite(float(item[key])):
                    raise RuntimeError(f"nonfinite saved training metric: {key}")
            last = item
            peak = max(peak, float(item.get("gpu_peak_memory_reserved_gib", 0)))
    if not last or last["step"] != expected_step:
        raise RuntimeError(f"{run_dir}: expected final optimizer step {expected_step}, got {last}")
    checkpoint = run_dir / "latest-model-optimizer-lr"
    purpose = "stage05_ar_resume" if phase == "ar" else "stage05_joint_resume"
    command = [PYTHON, "-m", "utils.stage05_checkpoint_contract", "--checkpoint", str(checkpoint),
               "--purpose", purpose, "--resume-training", "--validate-resume-artifacts",
               "--external-config", env["ZR0_ACTION_EXPERT_CONFIG_PATH"],
               "--action-horizon", env["ZR0_ACTION_HORIZON"], "--action-dim", "64", "--state-dim", "64",
               "--num-difference-queries", "32"]
    gate = resource_gate(output, env)
    # DeepSpeed deserialization imports Triton; tensor loading remains on CPU.
    subprocess.run(command, cwd=ROOT, env={**env, "CUDA_VISIBLE_DEVICES": gate["cuda_visible_devices"]}, check=True)
    for filename in ("data_seen_state.npz", "resolved_dataset_manifest.json", "scheduler.pt"):
        if not (checkpoint / filename).is_file():
            raise RuntimeError(f"missing checkpoint artifact: {checkpoint / filename}")
    if not (run_dir / f"step-{expected_step}/zr0_checkpoint_metadata.json").is_file():
        raise RuntimeError("final model snapshot missing")
    if env.get("ZR0_SELECT_PROCESSOR_AUDIT") == "1":
        select_token_audit(output, checkpoint, env)
    record(output, phase, status="checkpoint_verified", checkpoint=str(checkpoint),
           global_step=expected_step, peak_reserved_gib=peak, nominal_headroom_gib=80 - peak,
           output_dir=str(run_dir))
    return checkpoint


def main():
    import yaml
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-ar-resume-gate", action="store_true",
                        help="Continue once from the existing successful AR 32/2 two-step probe")
    parser.add_argument("--continue-ar-formal", action="store_true",
                        help="Continue after an already completed and verified AR recovery gate")
    parser.add_argument("--processor-audits", action="store_true",
                        help="Select a valid existing audit for the actual processor; never regenerate")
    parser.add_argument("--continuation-id", default="ar_gate_continuation",
                        help="Unique record suffix for an explicitly authorized continuation")
    args = parser.parse_args()
    if args.continue_ar_resume_gate and args.continue_ar_formal:
        parser.error("choose only one continuation point")
    continuing = args.continue_ar_resume_gate or args.continue_ar_formal
    if not args.continuation_id or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789_"
                                       for character in args.continuation_id):
        parser.error("--continuation-id must contain only lowercase letters, digits and underscores")
    config = json.loads(args.config.read_text())
    output = Path(config["output_root"])
    if args.dry_run:
        print(json.dumps({"sequence": ([] if continuing else ["AR probe"]) +
                                      ([] if args.continue_ar_formal else ["AR resume"]) +
                                      ["AR formal", "AR checkpoint gate",
                                       "Joint probe", "Joint resume", "Joint formal"], "config": config}, indent=2))
        return
    if not (output / "preparation_complete.json").is_file():
        raise RuntimeError("validated preparation is not complete")
    runner_lock = (output / "runner.lock").open("a")
    fcntl.flock(runner_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    started = output / "runner_started.json"
    if started.exists() and not continuing:
        raise RuntimeError("this experiment runner has already been started")
    summary = json.loads((output / "data_summary.json").read_text())
    identity_path = output / "source_identity.json"
    if continuing:
        if not started.is_file():
            raise RuntimeError("AR gate continuation requires the original runner record")
        previous = json.loads(identity_path.read_text())
        current = source_identity()
        runner_path = str(Path(__file__).resolve().relative_to(ROOT))
        if {k: v for k, v in previous.items() if k != runner_path} != {
            k: v for k, v in current.items() if k != runner_path
        }:
            raise RuntimeError("training source changed beyond the authorized runner correction")
        events = [json.loads(line) for line in (output / "lifecycle.jsonl").read_text().splitlines()]
        probe_dir = output / "probes/ar/mbs32_gas2"
        if not any(item.get("status") == "exited" and item.get("exit_code") == 0
                   and item.get("stage") == "ar-smoke" and item.get("output_dir") == str(probe_dir)
                   for item in events):
            raise RuntimeError("successful AR 32/2 probe record is missing")
        if any(item.get("stage") in ("ar-formal", "joint-smoke") for item in events):
            raise RuntimeError("experiment already advanced beyond the authorized continuation point")
        resumes = [item for item in events if item.get("stage") == "ar-resume"]
        if args.continue_ar_formal:
            if not resumes or resumes[-1].get("status") != "exited" or resumes[-1].get("exit_code") != 0:
                raise RuntimeError("formal continuation requires a successful AR recovery process")
            completed_ar_resume = Path(resumes[-1]["output_dir"])
            if not any(item.get("stage") == "ar" and item.get("status") == "checkpoint_verified"
                       and item.get("global_step") == 3 and item.get("output_dir") == str(completed_ar_resume)
                       for item in events):
                raise RuntimeError("formal continuation requires the completed step3 checkpoint gate")
        elif resumes:
            failed = resumes[-1]
            if (not args.processor_audits or failed.get("status") != "failed"
                    or "processor/tokenizer files identity changed" not in Path(failed["log"]).read_text()):
                raise RuntimeError("continuation requires the recorded processor-audit failure")
        resource_gate(output)
        started = output / f"runner_{args.continuation_id}.json"
        identity_path = output / f"source_identity_{args.continuation_id}.json"
    with started.open("x") as stream:
        json.dump({"pid": os.getpid(), "tmux": os.environ.get("TMUX"),
                   "time": datetime.now(timezone.utc).isoformat(), "config": str(args.config.resolve()),
                   "continue_ar_resume_gate": args.continue_ar_resume_gate,
                   "continue_ar_formal": args.continue_ar_formal,
                   "processor_audits": args.processor_audits}, stream, indent=2)
    write_json(identity_path, source_identity())
    suffix = f"_{args.continuation_id}" if continuing else ""
    write_json(output / f"environment{suffix}.json", {
        "python": sys.version, "executable": sys.executable,
        "packages": {name: version(name) for name in ("torch", "accelerate", "transformers", "deepspeed", "numpy", "pyarrow", "scipy")},
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "gpu": subprocess.check_output(["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version", "--format=csv"], text=True),
    })
    (output / f"launch_worktree{suffix}.patch").write_bytes(subprocess.check_output(["git", "diff", "HEAD", "--binary"], cwd=ROOT))
    base_env = {**os.environ, "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONPATH": f"{ROOT}:{ROOT / 'lerobot'}", "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
                "ZR0_OUTPUT_ROOT": str(output), "ZR0_MODEL_PATH": config["base_model"],
                "ZR0_EXPERIMENT_DOC": str(output / "experiment.md"),
                "ZR0_DATASET_REGISTRY": str(output / "config_view/dataset2feature.yaml"),
                "ZR0_TOKEN_AUDIT_SCRIPT": str(output / "config_view/scripts/audit_stage05_token_lengths.py"),
                "ZR0_TOKEN_AUDIT_SPEC": str(args.config.resolve()),
                "ZR0_TOKEN_LENGTH_AUDIT": str(output / "token_audit_h10_format2.json"),
                "ZR0_TRAIN_ENTRYPOINT": str(ROOT / "scripts/stage05_experiment_train.py"),
                "ZR0_ACTION_EXPERT_CONFIG_PATH": str(output / "action_expert_config.json"),
                "ZR0_ACTION_HORIZON": str(config["action_horizon"]), "ZR0_MAX_LENGTH": str(config["max_length"]),
                "ZR0_EXPECTED_GLOBAL_BATCH_SIZE": str(config["global_batch_size"]),
                "ZR0_NUM_GPUS": "4",
                "ZR0_CUDA_VISIBLE_DEVICES": os.environ.get("ZR0_CUDA_VISIBLE_DEVICES", "0,1,2,3"),
                "CUDA_VISIBLE_DEVICES": os.environ.get("ZR0_CUDA_VISIBLE_DEVICES", "0,1,2,3"),
                "ZR0_RUNNER_SOURCE_IDENTITY": str(identity_path),
                "ZR0_WANDB_PROJECT": config["wandb_project"], "ZR0_WANDB_GROUP": config["experiment"],
                "ZR0_ALLOW_FORMAL_TRAINING": "1"}
    if args.processor_audits:
        base_env["ZR0_SELECT_PROCESSOR_AUDIT"] = "1"
    if continuing:
        base_env["ZR0_CONTINUATION_ID"] = args.continuation_id
    try:
        source = config["base_model"]
        for phase in ("ar", "joint"):
            continuing_ar = continuing and phase == "ar"
            if continuing_ar:
                samples = output / "ar_probe_samples.json"
            elif args.processor_audits:
                phase_env = select_token_audit(output, source, base_env)
                samples = make_fixture(output, phase, Path(phase_env["ZR0_TOKEN_LENGTH_AUDIT"]))
            else:
                samples = make_fixture(output, phase)
            selected = None
            for micro in ([32] if continuing_ar else config["micro_batch_candidates"]):
                gas = config["global_batch_size"] // (4 * micro)
                accelerate = yaml.safe_load((ROOT / "accelerate_configs/accelerate_config.yaml").read_text())
                accelerate["deepspeed_config"].update(gradient_accumulation_steps=gas,
                    train_micro_batch_size_per_gpu=micro, train_batch_size=config["global_batch_size"],
                    gradient_clipping=config["gradient_clipping"])
                accelerate_path = output / f"{phase}_mbs{micro}_gas{gas}.yaml"
                if continuing_ar:
                    if yaml.safe_load(accelerate_path.read_text()) != accelerate or not samples.is_file():
                        raise RuntimeError("saved AR probe configuration does not match the continuation")
                else:
                    accelerate_path.write_text(yaml.safe_dump(accelerate, sort_keys=False))
                run_dir = output / "probes" / phase / f"mbs{micro}_gas{gas}"
                env = {**base_env, "ZR0_ACCELERATE_CONFIG": str(accelerate_path),
                       "ZR0_PER_DEVICE_BATCH_SIZE": str(micro), "ZR0_GRADIENT_ACCUMULATION_STEPS": str(gas),
                       "ZR0_RUN_OUTPUT_DIR": str(run_dir), "ZR0_INITIAL_CHECKPOINT": source,
                       "ZR0_PROBE_SAMPLES": str(samples), "ZR0_SAVE_STEP_INTERVAL": "2",
                       "ZR0_WANDB_RUN_NAME": f"{config['experiment']}-{phase}-probe-mbs{micro}",
                       "ZR0_WANDB_RUN_ID": f"{config['experiment']}-{phase}-probe-mbs{micro}"}
                if not continuing_ar:
                    try:
                        launch(output, f"{phase}-smoke", env, probe=True)
                    except ProbeOOM:
                        continue
                if continuing_ar and args.continue_ar_formal:
                    verify_run(output, completed_ar_resume, 3, phase, env)
                else:
                    checkpoint = verify_run(output, run_dir, 2, phase, env)
                    resume_env = {**env, "ZR0_INITIAL_CHECKPOINT": str(checkpoint), "ZR0_SAVE_STEP_INTERVAL": "3"}
                    if args.processor_audits:
                        resume_dir = run_dir.with_name(f"{run_dir.name}_resume_{args.continuation_id}")
                        if resume_dir.exists():
                            raise FileExistsError(f"refusing to overwrite recovery output: {resume_dir}")
                        resume_env["ZR0_RUN_OUTPUT_DIR"] = str(resume_dir)
                    launch(output, f"{phase}-resume", resume_env)
                    verify_run(output, Path(resume_env["ZR0_RUN_OUTPUT_DIR"]), 3, phase, resume_env)
                selected = (micro, gas, env)
                break
            if selected is None:
                raise RuntimeError(f"{phase}: all authorized micro-batch candidates exhausted by CUDA OOM")
            micro, gas, env = selected
            formal = output / "formal" / phase
            steps = summary[f"{phase}_steps"]
            formal_env = {key: value for key, value in env.items() if key != "ZR0_PROBE_SAMPLES"}
            formal_env.update(ZR0_RUN_OUTPUT_DIR=str(formal), ZR0_FORMAL_MAX_STEPS=str(steps),
                              ZR0_INITIAL_CHECKPOINT=source, ZR0_SAVE_STEP_INTERVAL=str(config["save_step_interval"]),
                              ZR0_WANDB_RUN_NAME=f"{config['experiment']}-{phase}-formal",
                              ZR0_WANDB_RUN_ID=f"{config['experiment']}-{phase}-formal")
            record(output, phase, status="probe_and_resume_passed", micro=micro, gas=gas,
                   formal_steps=steps, formal_warmup=summary[f"{phase}_warmup_steps"])
            launch(output, f"{phase}-formal", formal_env)
            source = str(verify_run(output, formal, steps, phase, formal_env))
        record(output, "experiment", status="complete")
    except BaseException as error:
        record(output, "experiment", status="stopped", error_type=type(error).__name__, error=str(error))
        raise


if __name__ == "__main__":
    main()
