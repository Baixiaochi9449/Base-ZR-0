#!/usr/bin/env python3
"""Record the non-invasive four-GPU availability gate for Stage05 runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from utils.gpu_resource_gate import GPUResourceError, wait_for_gpus


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--visible-devices", default=os.environ.get("ZR0_CUDA_VISIBLE_DEVICES",
                                                                  os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3")))
    args = parser.parse_args()
    samples = args.output.with_suffix(".samples.jsonl")
    if args.output.exists() or samples.exists():
        raise FileExistsError(f"refusing to overwrite {args.output} or {samples}")
    try:
        result = wait_for_gpus(env={**os.environ, "CUDA_VISIBLE_DEVICES": args.visible_devices}, log_path=samples)
        outcome = {"gate": "GO", "result": result}
    except GPUResourceError as error:
        outcome = {"gate": "NO-GO", "reason": str(error)}
    report = {
        "format_version": 2,
        "requested_cuda_visible_devices": args.visible_devices,
        "sample_log": str(samples),
        **outcome,
        "probe_started": False,
        "smoke_started": False,
        "pilot_started": False,
    }
    report["content_hash"] = hashlib.sha256(
        json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "gate": report["gate"]}))
    if report["gate"] != "GO":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
