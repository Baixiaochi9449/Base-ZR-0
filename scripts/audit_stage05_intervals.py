#!/usr/bin/env python3
"""Audit Stage05 text intervals against native H=32 chunks on AR-eligible frames."""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.future_difference_audit import classify_interval_overlap
from utils.stage05_sidecar import load_stage05_sidecar


DATASETS = {
    "droid": "/opt/data/private/lq/datasets/droid_1.0.1_stage05_full_95658_20260831",
    "household": "/opt/data/private/lq/datasets/molmoact_dataset_household-v3_stage05",
    "tabletop": "/opt/data/private/lq/datasets/molmoact_dataset_tabletop-v3_stage05",
    "rh20t": "/opt/data/private/lq/datasets/RH20T-v30_stage05",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _distribution(histogram: Counter[int]) -> dict[str, float | int | None]:
    count = sum(histogram.values())
    if not count:
        return {key: None for key in ("min", "max", "mean", "p50", "p90", "p95", "p99")}
    ordered = sorted(histogram.items())

    def value_at(rank: int) -> int:
        cumulative = 0
        for value, frequency in ordered:
            cumulative += frequency
            if rank < cumulative:
                return value
        raise AssertionError("histogram rank is out of bounds")

    def percentile(percent: float) -> float:
        position = (count - 1) * percent / 100.0
        lower, upper = math.floor(position), math.ceil(position)
        low, high = value_at(lower), value_at(upper)
        return float(low + (high - low) * (position - lower))

    return {
        "min": ordered[0][0],
        "max": ordered[-1][0],
        "mean": sum(value * frequency for value, frequency in ordered) / count,
        "p50": percentile(50),
        "p90": percentile(90),
        "p95": percentile(95),
        "p99": percentile(99),
    }


def _episode_mapping(root: Path) -> dict[int, int]:
    result = {}
    path = root / "meta/stage05_episode_mapping.jsonl"
    with path.open(encoding="utf-8") as source:
        for line in source:
            if line.strip():
                row = json.loads(line)
                result[int(row["new_episode_index"])] = int(row["old_episode_index"])
    return result


def _active_annotations(annotation_root: Path, episode: int) -> Path | None:
    episode_root = annotation_root / f"episode_{episode:06d}"
    for status_path in sorted(episode_root.glob("shard_*/rank_*/status.json")):
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
            artifact = Path(status["artifacts"]["training_samples"]["path"])
        except (KeyError, TypeError, json.JSONDecodeError):
            continue
        for candidate in (artifact, status_path.parent / artifact.name):
            if candidate.is_file():
                return candidate.resolve()
    candidates = sorted(episode_root.glob("shard_*/rank_*/training_samples*.jsonl"))
    return candidates[0].resolve() if len(candidates) == 1 else None


def _intervals(path: Path) -> tuple[list[int], list[tuple[int, int]]]:
    values = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                base = json.loads(line)["base_data"]
                interval = base["language_action_interval"]
                anchor = int(base["semantic_anchor_frame"])
                start = int(interval["start_frame_inclusive"])
                stop = int(interval["end_frame_inclusive"])
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(f"{path}:{line_number}: invalid language interval") from error
            if start > stop:
                raise ValueError(f"{path}:{line_number}: reversed language interval")
            values.append((anchor, start, stop))
    values.sort()
    anchors = [value[0] for value in values]
    if len(anchors) != len(set(anchors)):
        raise ValueError(f"{path}: duplicate semantic anchor")
    return anchors, [(value[1], value[2]) for value in values]


def _audit(name: str, root: Path, sidecar: Path, horizon: int) -> dict:
    manifest = load_stage05_sidecar(sidecar, verify_source=True)
    eligible = np.load(sidecar / "ar_indices.npy", mmap_mode="r", allow_pickle=False)
    episodes = pq.read_table(sidecar / "episodes.parquet").to_pylist()
    mapping = _episode_mapping(root)
    merge_path = root / "meta/stage05_merge.json"
    merge = json.loads(merge_path.read_text(encoding="utf-8"))
    annotation_root = Path(merge["stage05_dir"]).resolve()
    if not annotation_root.is_dir():
        raise FileNotFoundError(annotation_root)

    classifications = Counter({key: 0 for key in ("exact", "full", "partial", "none")})
    annotation_lengths: Counter[int] = Counter()
    intersections: Counter[int] = Counter()
    unavailable = Counter()
    annotation_identity = hashlib.sha256()
    annotation_files = 0
    audited = 0
    eligible_cursor = 0
    for number, record in enumerate(episodes, 1):
        start = int(record["dataset_from_index"])
        stop = int(record["dataset_to_index"])
        count = int(record["ar_eligible_frames"])
        episode_eligible = eligible[eligible_cursor : eligible_cursor + count]
        eligible_cursor += count
        if count and not (
            int(episode_eligible[0]) >= start and int(episode_eligible[-1]) < stop
        ):
            raise ValueError(f"{name}: AR indices cross episode {record['episode_index']} boundary")
        if not count:
            continue
        episode = int(record["episode_index"])
        old_episode = mapping.get(episode)
        annotation_path = (
            _active_annotations(annotation_root, old_episode)
            if old_episode is not None else None
        )
        if annotation_path is None:
            unavailable["annotation_episode_missing"] += count
            continue
        anchors, intervals = _intervals(annotation_path)
        try:
            annotation_identity_path = annotation_path.relative_to(annotation_root).as_posix()
        except ValueError:
            # Some DROID status files intentionally point at the preceding immutable
            # merge directory. Keep the resolved path in the audit identity.
            annotation_identity_path = str(annotation_path)
        annotation_identity.update(annotation_identity_path.encode())
        annotation_identity.update(_sha256(annotation_path).encode())
        annotation_files += 1
        for global_index in episode_eligible:
            frame = int(global_index) - start
            position = bisect.bisect_right(anchors, frame) - 1
            if position < 0:
                unavailable["no_preceding_annotation"] += 1
                continue
            interval_start, interval_stop = intervals[position]
            overlap = classify_interval_overlap(
                frame, frame + horizon - 1, interval_start, interval_stop
            )
            classifications[overlap.classification] += 1
            annotation_lengths[overlap.annotation_length] += 1
            intersections[overlap.intersection_length] += 1
            audited += 1
        if number % 5000 == 0:
            print(f"[{name}] interval episodes {number}/{len(episodes)}", file=sys.stderr, flush=True)

    expected = int(manifest["counts"]["ar_eligible_frames"])
    if eligible_cursor != len(eligible):
        raise ValueError(f"{name}: episode AR counts do not cover the sidecar index")
    if audited + sum(unavailable.values()) != expected:
        raise ValueError(f"{name}: interval accounting mismatch")
    identity = {
        "sidecar_manifest_sha256": _sha256(sidecar / "manifest.json"),
        "stage05_merge_sha256": _sha256(merge_path),
        "episode_mapping_sha256": _sha256(root / "meta/stage05_episode_mapping.jsonl"),
        "annotation_file_count": annotation_files,
        "annotation_content_identity_sha256": annotation_identity.hexdigest(),
    }
    return {
        "dataset": name,
        "dataset_root": str(root),
        "scope": "all final AR-eligible labeled frames; this is a read-only audit, not a validation split",
        "action_horizon": horizon,
        "native_fps": manifest["generation"]["native_fps"],
        "ar_eligible_frames": expected,
        "available_count": audited,
        "unavailable_count": int(sum(unavailable.values())),
        "unavailable_reasons": dict(sorted(unavailable.items())),
        "classification_counts": dict(classifications),
        "annotation_interval_length": _distribution(annotation_lengths),
        "intersection_length": _distribution(intersections),
        "known_limitation": "language intervals and native H=32 FM chunks are not forced to align",
        "identity": identity,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sidecar-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--action-horizon", type=int, default=32)
    args = parser.parse_args()
    if args.action_horizon < 1:
        raise ValueError("Stage05 action_horizon must be a positive integer")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    report = {
        "format_version": 1,
        "datasets": {
            name: _audit(name, Path(root), args.sidecar_root / "ar" / name, 32)
            for name, root in DATASETS.items()
        },
    }
    report["content_hash"] = hashlib.sha256(
        json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "content_hash": report["content_hash"]}))


if __name__ == "__main__":
    main()
