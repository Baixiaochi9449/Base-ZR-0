#!/usr/bin/env python3
"""Record the non-invasive four-GPU availability gate for Stage05 runs."""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import subprocess
from pathlib import Path


def _query(fields: str) -> list[list[str]]:
    result = subprocess.run(
        [
            "nvidia-smi",
            f"--query-gpu={fields}",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    return [[value.strip() for value in line.split(",")] for line in result.stdout.splitlines()]


def _processes() -> list[list[str]]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    return [[value.strip() for value in line.split(",")] for line in result.stdout.splitlines()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    rows = _query("index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu")
    gpu_by_uuid = {}
    gpus = []
    for index, uuid, name, total, used, free, utilization, temperature in rows:
        record = {
            "index": int(index),
            "uuid": uuid,
            "name": name,
            "memory_total_mib": int(total),
            "memory_used_mib": int(used),
            "memory_free_mib": int(free),
            "utilization_percent": int(utilization),
            "temperature_c": int(temperature),
            "existing_compute_processes": [],
        }
        gpu_by_uuid[uuid] = record
        gpus.append(record)
    for uuid, pid, process_name, used_memory in _processes():
        if uuid in gpu_by_uuid:
            gpu_by_uuid[uuid]["existing_compute_processes"].append(
                {
                    "pid": int(pid),
                    "process_name": process_name,
                    "used_memory_mib": int(used_memory),
                }
            )

    selected = [gpu for gpu in gpus if gpu["index"] in {0, 1, 2, 3}]
    occupied = [gpu["index"] for gpu in selected if gpu["existing_compute_processes"]]
    report = {
        "format_version": 1,
        "captured_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "requested_gpu_indices": [0, 1, 2, 3],
        "gpus": selected,
        "gate": "NO-GO" if occupied else "GO",
        "probe_started": False,
        "smoke_started": False,
        "pilot_started": False,
        "reason": (
            "All requested GPUs have pre-existing compute workloads; a stability probe "
            "would compete for memory and violate the no-preemption/no-contention constraint."
            if occupied
            else "No pre-existing compute workload was detected on the requested GPUs."
        ),
    }
    report["content_hash"] = hashlib.sha256(
        json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "gate": report["gate"]}))


if __name__ == "__main__":
    main()
