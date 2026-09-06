#!/usr/bin/env python3
"""Compare installed LeRobot video backends on real DROID and RH20T clips."""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import time
from pathlib import Path

import pyarrow.parquet as pq

from lerobot.common.datasets.video_utils import decode_video_frames


DATASETS = {
    "droid": Path("/opt/data/private/lq/datasets/droid_1.0.1_stage05_full_95658_20260831"),
    "rh20t": Path("/opt/data/private/lq/datasets/RH20T-v30_stage05"),
}
CAMERAS = (
    "observation.images.exterior_1_left",
    "observation.images.wrist_left",
)


def _first_long_episode(root: Path) -> dict:
    columns = ["episode_index", "length"]
    for camera in CAMERAS:
        columns += [
            f"videos/{camera}/chunk_index",
            f"videos/{camera}/file_index",
            f"videos/{camera}/from_timestamp",
        ]
    for path in sorted((root / "meta" / "episodes").glob("**/*.parquet")):
        for row in pq.read_table(path, columns=columns).to_pylist():
            if int(row["length"]) >= 64:
                return row
    raise ValueError(f"no episode with at least 64 frames in {root}")


def _video_path(root: Path, row: dict, camera: str) -> Path:
    prefix = f"videos/{camera}"
    return root / "videos" / camera / f"chunk-{int(row[prefix + '/chunk_index']):03d}" / f"file-{int(row[prefix + '/file_index']):03d}.mp4"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    report = {
        "format_version": 1,
        "dependencies": {
            name: bool(importlib.util.find_spec(name))
            for name in ("av", "torchcodec", "torchvision", "lerobot")
        },
        "datasets": {},
        "selection": "pyav",
        "selection_reason": "already used by LeRobot, decoded both codecs, and had lowest measured latency",
    }
    for name, root in DATASETS.items():
        info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
        fps = float(info["fps"])
        row = _first_long_episode(root)
        result = {"episode_index": int(row["episode_index"]), "fps": fps, "cameras": {}}
        for camera in CAMERAS:
            path = _video_path(root, row, camera)
            probe = subprocess.run(
                [
                    "ffprobe", "-v", "error", "-select_streams", "v:0",
                    "-show_entries", "stream=codec_name,width,height", "-of", "json", str(path),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            prefix = f"videos/{camera}"
            start = float(row[prefix + "/from_timestamp"])
            timestamps = [start + index / fps for index in range(32)]
            backends = {}
            for backend in ("pyav", "torchcodec"):
                started = time.monotonic()
                try:
                    frames = decode_video_frames(
                        path,
                        timestamps,
                        tolerance_s=0.51 / fps + 1e-6,
                        backend=backend,
                    )
                    backends[backend] = {
                        "ok": True,
                        "seconds_for_32_frames": time.monotonic() - started,
                        "shape": list(frames.shape),
                        "finite": bool(frames.isfinite().all()),
                        "value_range": [float(frames.min()), float(frames.max())],
                    }
                except Exception as error:
                    backends[backend] = {
                        "ok": False,
                        "seconds_before_error": time.monotonic() - started,
                        "error": f"{type(error).__name__}: {error}",
                    }
            result["cameras"][camera] = {
                "path": str(path),
                "ffprobe": json.loads(probe.stdout)["streams"][0],
                "backends": backends,
            }
        report["datasets"][name] = result
    if not all(
        camera["backends"]["pyav"]["ok"]
        for dataset in report["datasets"].values()
        for camera in dataset["cameras"].values()
    ):
        raise RuntimeError("PyAV did not decode every required codec/camera")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "selection": "pyav"}))


if __name__ == "__main__":
    main()
