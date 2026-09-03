#!/usr/bin/env python3
"""Print a read-only future-difference target-length audit as JSON."""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.future_difference_audit import (
    FutureDifferenceTokenMeasurer,
    audit_future_difference_lengths,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--processor-path", required=True)
    parser.add_argument("--max-length", required=True, type=int)
    parser.add_argument(
        "--camera-keys",
        nargs=3,
        default=("first_view", "second_view", "wrist_image"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    from transformers import AutoProcessor

    root = Path(args.dataset_root).resolve()
    info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    camera_shapes = {}
    for camera_key in args.camera_keys:
        try:
            camera_shapes[camera_key] = info["features"][camera_key]["shape"]
        except (KeyError, TypeError) as error:
            raise ValueError(
                f"meta/info.json does not define a shape for camera {camera_key!r}"
            ) from error
    processor = AutoProcessor.from_pretrained(args.processor_path)
    measurer = FutureDifferenceTokenMeasurer(processor, camera_shapes)

    def report_progress(sample_count):
        if sample_count % 10_000 == 0:
            print(f"audited {sample_count} samples", file=sys.stderr, flush=True)

    report = audit_future_difference_lengths(
        root,
        max_length=args.max_length,
        measure_sample=measurer,
        progress_sample=report_progress,
    )
    report["processor_path"] = args.processor_path
    report["camera_shapes"] = camera_shapes
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
