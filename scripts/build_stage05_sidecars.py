#!/usr/bin/env python3
"""Create non-destructive Stage05 pretraining sidecars for one or four sources."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.stage05_sidecar import build_stage05_sidecar


DATASETS = {
    "droid": ("/opt/data/private/lq/datasets/droid_1.0.1_stage05_full_95658_20260831", "droid", False),
    "household": ("/opt/data/private/lq/datasets/molmoact_dataset_household-v3_stage05", "molmo", True),
    "tabletop": ("/opt/data/private/lq/datasets/molmoact_dataset_tabletop-v3_stage05", "molmo", True),
    "rh20t": ("/opt/data/private/lq/datasets/RH20T-v30_stage05", "rh20t", False),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", choices=[*DATASETS, "all"], default=["all"])
    parser.add_argument("--ar-only", action="store_true", help="Build AR eligibility without canonical/FM gates.")
    parser.add_argument("--video-backend", choices=["pyav", "torchcodec"], default="pyav")
    parser.add_argument("--horizon", type=int, default=32)
    args = parser.parse_args()
    if isinstance(args.horizon, bool) or args.horizon <= 0:
        parser.error("--horizon must be a positive integer")
    names = list(DATASETS) if "all" in args.datasets else args.datasets
    summaries = {}
    for name in names:
        root, kind, embedded = DATASETS[name]
        phase = "ar" if args.ar_only else "joint"
        manifest = build_stage05_sidecar(
            root=root,
            output=args.output_root / phase / name,
            dataset_id=f"stage05_{name}",
            kind=kind,
            embedded_images=embedded,
            horizon=args.horizon,
            video_backend=args.video_backend,
            build_joint=not args.ar_only,
        )
        summaries[name] = manifest["counts"]
        print(json.dumps({name: manifest["counts"]}, sort_keys=True))
    (args.output_root / f"{phase}_summary.json").write_text(
        json.dumps(summaries, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
