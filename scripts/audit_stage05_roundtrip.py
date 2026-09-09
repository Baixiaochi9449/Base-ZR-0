#!/usr/bin/env python3
"""Audit real Stage05 actions through production canonicalization and clipping."""

from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.normalization import NORMALIZED_VALUE_CLIP, min_max_norm_unclipped
from utils.stage05_dataset import Stage05MixedPretrainingDataset
from utils.stage05_roundtrip import RoundTripAccumulator, mathematical_in_range_roundtrip
from utils.stage05_sidecar import canonical_json_hash, load_stage05_sidecar, load_stage05_stats


DATASETS = OrderedDict(
    (
        ("droid", ("/opt/data/private/lq/datasets/droid_1.0.1_stage05_full_95658_20260831", "droid", False)),
        ("household", ("/opt/data/private/lq/datasets/molmoact_dataset_household-v3_stage05", "molmo", True)),
        ("tabletop", ("/opt/data/private/lq/datasets/molmoact_dataset_tabletop-v3_stage05", "molmo", True)),
        ("rh20t", ("/opt/data/private/lq/datasets/RH20T-v30_stage05", "rh20t", False)),
    )
)


def _source_episode_metadata(root: Path) -> dict[int, dict]:
    result = {}
    for path in sorted((root / "meta/episodes").glob("**/*.parquet")):
        for row in pq.read_table(
            path, columns=["episode_index", "data/chunk_index", "data/file_index"]
        ).to_pylist():
            result[int(row["episode_index"])] = row
    return result


def _data_path(root: Path, metadata: dict) -> Path:
    return (
        root
        / "data"
        / f"chunk-{int(metadata['data/chunk_index']):03d}"
        / f"file-{int(metadata['data/file_index']):03d}.parquet"
    )


def _action_columns(kind: str) -> list[str]:
    common = ["episode_index", "frame_index", "timestamp"]
    if kind == "molmo":
        return common[:2] + ["state", "actions"]
    if kind == "droid":
        return common + [
            "observation.state.cartesian_position",
            "observation.state.gripper_position",
            "action.cartesian_position",
            "action.gripper_position",
        ]
    return common + ["observation.state", "action", "action.valid"]


def _production_adapter(kind: str, horizon=32) -> Stage05MixedPretrainingDataset:
    adapter = object.__new__(Stage05MixedPretrainingDataset)
    adapter.kind = kind
    adapter.action_horizon = horizon
    adapter._canonical_cache = OrderedDict()
    return adapter


def _eligible_bases(state_valid: np.ndarray, action_valid: np.ndarray, horizon=32) -> np.ndarray:
    future_valid = np.convolve(
        action_valid[::-1].astype(np.int16), np.ones(horizon, dtype=np.int16), mode="full"
    )[: len(action_valid)][::-1]
    return np.flatnonzero(state_valid & (future_valid > 0))


def _identities(name, episode, rows, bases, fps):
    result = []
    for base in bases:
        row = rows[int(base)]
        frame = int(row["frame_index"])
        timestamp = row.get("timestamp")
        if timestamp is None:
            timestamp = frame / fps
        result.append(
            {
                "dataset": name,
                "episode": int(episode),
                "frame": frame,
                "time": float(timestamp),
            }
        )
    return result


def _stratified_bases(actions, action_valid, eligible, q01, q99) -> list[int]:
    bases = {int(eligible[0]), int(eligible[len(eligible) // 2]), int(eligible[-1])}
    valid_rows = np.flatnonzero(action_valid)
    values = actions[valid_rows]
    for dimension in range(7):
        for target in (q01[dimension], q99[dimension]):
            bases.add(int(valid_rows[np.argmin(np.abs(values[:, dimension] - target))]))
        bases.add(int(valid_rows[np.argmin(values[:, dimension])]))
        bases.add(int(valid_rows[np.argmax(values[:, dimension])]))
    eligible_set = set(int(value) for value in eligible)
    return sorted(base for base in bases if base in eligible_set)


def _audit_dataset(name, root, kind, sidecar, episodes_per_dataset, full_rh20t, horizon=32):
    manifest = load_stage05_sidecar(sidecar, verify_source=True)
    stats_payload = load_stage05_stats(
        sidecar / "stats.json", expected_stats_key=f"stage05_{name}"
    )
    stats = {
        key: np.asarray(stats_payload["statistics"]["actions"][key], dtype=np.float32)
        for key in ("q01", "q99")
    }
    episode_records = [
        row
        for row in pq.read_table(sidecar / "episodes.parquet").to_pylist()
        if int(row["joint_action_eligible_frames"]) > 0
    ]
    full = name == "rh20t" and full_rh20t
    if full:
        selected = episode_records
    else:
        positions = np.unique(
            np.linspace(
                0,
                len(episode_records) - 1,
                min(episodes_per_dataset, len(episode_records)),
                dtype=np.int64,
            )
        )
        selected = [episode_records[int(position)] for position in positions]

    source_metadata = _source_episode_metadata(root)
    by_file = {}
    for record in selected:
        episode = int(record["episode_index"])
        by_file.setdefault(_data_path(root, source_metadata[episode]), []).append(record)

    adapter = _production_adapter(kind, horizon)
    accumulator = RoundTripAccumulator()
    audited_chunks = 0
    ordered_files = sorted(by_file.items(), key=lambda item: str(item[0]))
    fps = float(manifest["generation"]["native_fps"])
    for file_number, (path, records) in enumerate(ordered_files, start=1):
        print(
            f"[{name}] real round-trip parquet {file_number}/{len(ordered_files)}",
            file=sys.stderr,
            flush=True,
        )
        table = pq.read_table(path, columns=_action_columns(kind))
        grouped = {}
        for row in table.to_pylist():
            grouped.setdefault(int(row["episode_index"]), []).append(row)
        for record in records:
            episode = int(record["episode_index"])
            rows = sorted(grouped[episode], key=lambda row: int(row["frame_index"]))
            # This production method is the canonicalization source used by training.
            adapter._canonical_chunk(episode, rows, 0)
            states, actions, state_valid, action_valid = adapter._canonical_cache[episode]
            eligible = _eligible_bases(state_valid, action_valid, horizon)
            if not len(eligible):
                raise ValueError(f"{name} episode {episode} has no production-eligible base")
            if full:
                referenced = np.convolve(
                    np.isin(np.arange(len(rows)), eligible).astype(np.int16),
                    np.ones(horizon, dtype=np.int16),
                    mode="full",
                )[: len(rows)] > 0
                selected_steps = np.flatnonzero(action_valid & referenced)
                valid = np.ones((len(selected_steps), 7), dtype=bool)
                accumulator.update(
                    actions[selected_steps],
                    valid,
                    stats,
                    _identities(name, episode, rows, selected_steps, fps),
                )
                audited_chunks += len(eligible)
                continue

            bases = _stratified_bases(
                actions, action_valid, eligible, stats["q01"], stats["q99"]
            )
            for base in bases:
                chunk = adapter._canonical_chunk(episode, rows, base)
                raw_indices = base + np.arange(horizon, dtype=np.int64)
                source_indices = np.minimum(raw_indices, len(rows) - 1)
                valid = chunk.temporal_mask[:, None] & chunk.dimension_mask[None, :]
                accumulator.update(
                    chunk.action,
                    valid,
                    stats,
                    _identities(name, episode, rows, source_indices, fps),
                )
                audited_chunks += 1

    per_dimension = accumulator.report()
    extrema = np.stack(
        [
            np.asarray(stats_payload["statistics"]["actions"]["min"], dtype=np.float32),
            np.asarray(stats_payload["statistics"]["actions"]["max"], dtype=np.float32),
        ]
    )
    extrema_pre = min_max_norm_unclipped(torch.from_numpy(extrema), stats, True)
    low, high = NORMALIZED_VALUE_CLIP
    no_full_clip_from_extrema = bool(
        ((extrema_pre >= low) & (extrema_pre <= high)).all()
    )
    full_population = {
        "scope": (
            "all canonical valid action steps referenced by Joint chunks"
            if full
            else "inferred from full-population min/max in sidecar stats"
        ),
        "complete": full or no_full_clip_from_extrema,
        "clipped_count_by_dimension": (
            [item["clipped_count"] for item in per_dimension]
            if full
            else ([0] * 7 if no_full_clip_from_extrema else None)
        ),
        "clipped_ratio_by_dimension": (
            [item["clipped_ratio"] for item in per_dimension]
            if full
            else ([0.0] * 7 if no_full_clip_from_extrema else None)
        ),
    }
    return {
        "sidecar_content_hash": manifest["content_hash"],
        "stats_content_hash": stats_payload["content_hash"],
        "sampling": {
            "mode": "full_valid_action_steps" if full else f"deterministic_stratified_h{horizon}_chunks",
            "eligible_episode_count": len(episode_records),
            "sampled_episode_count": len(selected),
            "audited_chunk_count": audited_chunks,
            "coverage": "episode first/middle/tail plus q01/q99-nearest and extrema candidates",
        },
        "mathematical_in_range_roundtrip_max_abs_error": mathematical_in_range_roundtrip(stats),
        "real_production_clipped_roundtrip_by_dimension": per_dimension,
        "full_population_clipping": full_population,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sidecar-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes-per-dataset", type=int, default=64)
    parser.add_argument(
        "--full-rh20t",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fully audit RH20T valid action steps so known extremes are included.",
    )
    args = parser.parse_args()
    if args.episodes_per_dataset < 3:
        raise ValueError("episodes-per-dataset must be at least three")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    report = {
        "format_version": 1,
        "method": (
            "real production Stage05MixedPretrainingDataset._canonical_chunk plus "
            "utils.normalization min_max_norm_unclipped/min_max_norm/min_max_denorm"
        ),
        "normalized_clip": list(NORMALIZED_VALUE_CLIP),
        "datasets": OrderedDict(),
        "limitations": (
            "DROID/Household/Tabletop error percentiles use deterministic stratified "
            "real H=32 chunks; their full-population zero-clipping result is exact because "
            "the full valid-action min/max remain inside the clip bounds."
        ),
    }
    for name, (path, kind, _embedded) in DATASETS.items():
        report["datasets"][name] = _audit_dataset(
            name,
            Path(path),
            kind,
            args.sidecar_root / "joint" / name,
            args.episodes_per_dataset,
            args.full_rh20t,
        )
    report["content_hash"] = canonical_json_hash(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(args.output), "content_hash": report["content_hash"]}))


if __name__ == "__main__":
    main()
