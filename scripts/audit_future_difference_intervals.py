#!/usr/bin/env python3
"""Print a read-only future-difference interval audit as JSON."""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.future_difference_audit import audit_future_difference_intervals


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--annotation-root")
    parser.add_argument("--action-horizon", required=True, type=int)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    report = audit_future_difference_intervals(
        args.dataset_root,
        action_horizon=args.action_horizon,
        annotation_root=args.annotation_root,
    )
    payload = json.dumps(report, ensure_ascii=False, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.output.exists():
            raise FileExistsError(f"refusing to overwrite {args.output}")
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
