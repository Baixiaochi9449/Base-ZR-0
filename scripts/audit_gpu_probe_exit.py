#!/usr/bin/env python3
"""Verify two bounded CUDA process exits using the production resource gate."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys

from utils.gpu_resource_gate import wait_for_gpus


PROBE = """
import json
import torch
assert torch.cuda.device_count() == 4
for device in range(4):
    matrix = torch.ones((64, 64), device=f'cuda:{device}')
    for _ in range(16):
        result = matrix @ matrix
    assert result[0, 0].item() == 64
    torch.cuda.synchronize(device)
print(json.dumps({'devices': 4, 'optimizer_updates': 0, 'status': 'complete'}))
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--visible-devices", default="0,1,2,3")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    doc = args.output / "experiment.md"
    command = [sys.executable, "-c", PROBE]
    doc.write_text(
        "# Sequential CUDA resource probes\n\n"
        f"- Created: {datetime.now(timezone.utc).isoformat()}. Owner: lq; executor: Codex.\n"
        "- Purpose: verify actual own-process exit and resource release before the next launch.\n"
        "- Model/data/checkpoint/loss/optimizer/W&B: none; zero training updates.\n"
        "- Hardware: four A800 GPUs, resolved and pinned by UUID; float32 64x64 matrices, 16 products per device, two fresh processes.\n"
        "- No random inputs, data preprocessing, labels, actions or checkpoints. No formal training branch.\n"
        "- Code/configuration: current unstaged resource fix; full baseline and resource policy are recorded in experiments/resource_preflight_20260907/experiment.md.\n"
        f"- Selected CUDA devices: {args.visible_devices}. Every gate uses the existing environment policy.\n"
        f"- Command for each process: `{shlex.join(command)}`\n"
    )
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": args.visible_devices}
    children, records = [], []
    report = {"started": datetime.now(timezone.utc).isoformat(), "optimizer_updates": 0, "probes": records}
    try:
        ready = wait_for_gpus(env=env, log_path=args.output / "gpu_gate.jsonl")
        env["CUDA_VISIBLE_DEVICES"] = ready["cuda_visible_devices"]
        report["initial_gate"] = ready
        for index in range(2):
            record = {"index": index, "command": command, "cuda_visible_devices": env["CUDA_VISIBLE_DEVICES"]}
            records.append(record)
            with (args.output / f"probe-{index}.log").open("x") as stream:
                child = subprocess.Popen(command, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                         start_new_session=True)
                children.append(child)
                record["pid"] = child.pid
                record["release_gate"] = wait_for_gpus(
                    env=env, children=children, process_groups=[item.pid for item in children],
                    log_path=args.output / "gpu_gate.jsonl",
                )
                record["exit_code"] = child.wait(timeout=1)
                if record["exit_code"]:
                    raise RuntimeError(f"resource probe {index} failed; see its log")
        report["status"] = "passed"
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        for child in children:
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                # Only this diagnostic's own new session can be terminated.
                os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=15)
        report["ended"] = datetime.now(timezone.utc).isoformat()
        (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
        with doc.open("a") as stream:
            stream.write(f"\n- Result: {report['status']}; ended {report['ended']}. See results.json and gpu_gate.jsonl.\n")
    print(json.dumps({"status": report["status"], "probes": len(records), "optimizer_updates": 0}))


if __name__ == "__main__":
    main()
