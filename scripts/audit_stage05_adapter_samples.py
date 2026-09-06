#!/usr/bin/env python3
"""Run real Stage05 adapter samples through decode, processor, and canonicalization."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.stage05_dataset import Stage05MixedPretrainingDataset
from utils.load_training_dataset import EpochGroupedSampler
from utils.normalization import min_max_denorm, min_max_norm


ENTRIES = (
    "stage05_droid_mixed",
    "stage05_household_mixed",
    "stage05_tabletop_mixed",
    "stage05_rh20t_mixed",
)


def _json_shape(value):
    return list(value.shape) if hasattr(value, "shape") else None


def _record(dataset, position: int) -> dict:
    sample = dataset[position]
    global_index = int(dataset.indices[position])
    episode, frame = dataset._episode_for_global(global_index)
    action_mask = sample.get("action_mask")
    image_grid = sample.get("image_grid_thw")
    return {
        "dataset_entry": dataset.spec.dataset_entry,
        "loss_type": dataset.loss_type,
        "dataset_position": int(position),
        "global_index": global_index,
        "episode_id": int(episode["episode_index"]),
        "frame_id": int(frame),
        "stats_key": sample["stats_key"],
        "image_count": int(image_grid.shape[0]),
        "input_ids_shape": _json_shape(sample["input_ids"]),
        "pixel_values_shape": _json_shape(sample["pixel_values"]),
        "image_grid_thw": image_grid.tolist(),
        "ar_eligible": bool(sample["ar_eligible"]),
        "ar_token_count": int(sample["labels"][1:].ne(-100).sum()),
        "fm_eligible": bool(sample["fm_eligible"]),
        "fm_count": int(action_mask.sum()) if action_mask is not None else 0,
        "action_shape": _json_shape(sample.get("action")),
        "state_shape": _json_shape(sample.get("observation.state")),
        "task_preserved": bool(sample["task"].strip()),
        "train_data_preserved": sample["train_data"] is not None,
        "slot_data_preserved": "slot_data" in sample,
    }


def _roundtrip(dataset, positions) -> dict:
    result = {}
    real_values = {"observation.state": [], "action": []}
    for position in positions:
        global_index = int(dataset.indices[position])
        episode_meta, base = dataset._episode_for_global(global_index)
        episode = int(episode_meta["episode_index"])
        chunk = dataset._canonical_chunk(episode, dataset._episode_rows(episode), base)
        real_values["observation.state"].append(chunk.state[None])
        real_values["action"].append(chunk.action[chunk.temporal_mask])
    for key, chunks in real_values.items():
        stats = dataset.stats[key]
        values = torch.from_numpy(np.concatenate(chunks)).float()
        restored = min_max_denorm(min_max_norm(values, stats, True), stats, True)
        result[key] = float((restored - values).abs().max())
    return result


def _first_joint_without_ar(dataset) -> int | None:
    ar = np.load(dataset.sidecar_root.parent.parent / "ar" / dataset.sidecar_root.name / "ar_indices.npy", mmap_mode="r")
    joint = dataset.indices
    for start in range(0, len(joint), 100_000):
        candidates = np.asarray(joint[start : start + 100_000])
        positions = np.searchsorted(ar, candidates)
        matches = (positions < len(ar)) & (ar[np.minimum(positions, len(ar) - 1)] == candidates)
        missing = np.flatnonzero(~matches)
        if len(missing):
            return start + int(missing[0])
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processor-path", required=True)
    parser.add_argument("--max-length", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(args.processor_path)
    registry = yaml.safe_load((ROOT / "dataset2feature.yaml").read_text(encoding="utf-8"))
    records = []
    stats_roundtrip_max_abs_error = {}
    datasets_by_loss = {"vlm": [], "vlm_and_action": []}
    for dataset_id, entry_name in enumerate(ENTRIES):
        entry = dict(registry[entry_name])
        entry["dataset_entry"] = entry_name
        for loss_type in ("vlm", "vlm_and_action"):
            dataset = Stage05MixedPretrainingDataset(
                entry=entry,
                processor=processor,
                loss_type=loss_type,
                max_length=args.max_length,
                action_horizon=32,
                max_pad_state_and_action_length=64,
                dataset_id=dataset_id,
            )
            datasets_by_loss[loss_type].append(dataset)
            records.append(_record(dataset, len(dataset) // 2))
            if loss_type == "vlm_and_action":
                positions = sorted({0, len(dataset) // 2, len(dataset) - 1})
                stats_roundtrip_max_abs_error[entry_name] = _roundtrip(dataset, positions)
            if entry_name == "stage05_droid_mixed" and loss_type == "vlm_and_action":
                no_ar_position = _first_joint_without_ar(dataset)
                if no_ar_position is None:
                    raise ValueError("DROID Joint audit could not find an FM-only sample")
                fm_only = _record(dataset, no_ar_position)
                if fm_only["ar_eligible"] or fm_only["ar_token_count"]:
                    raise ValueError("DROID FM-only sample unexpectedly contributes AR")
                records.append(fm_only)

    sampler_prefixes = {}
    for loss_type, datasets in datasets_by_loss.items():
        concat = torch.utils.data.ConcatDataset(datasets)
        sampler = EpochGroupedSampler(concat, seed=42)
        boundaries = np.cumsum([0, *[len(dataset) for dataset in datasets]])
        samples = list(itertools.islice(iter(sampler), 12_800))
        prefix_report = {}
        for limit in (128, 12_800):
            prefix = samples[:limit]
            counts = [
                int(sum(boundaries[index] <= value < boundaries[index + 1] for value in prefix))
                for index in range(len(datasets))
            ]
            prefix_report[str(limit)] = {
                "counts": dict(zip(ENTRIES, counts)),
                "ratios": {
                    entry: count / len(prefix)
                    for entry, count in zip(ENTRIES, counts)
                },
            }
        sampler_prefixes[loss_type] = prefix_report

    report = {
        "format_version": 1,
        "processor_path": str(Path(args.processor_path).resolve()),
        "max_length": args.max_length,
        "records": records,
        "stats_roundtrip_max_abs_error": stats_roundtrip_max_abs_error,
        "sampler_prefixes_seed42": sampler_prefixes,
    }
    report["content_hash"] = hashlib.sha256(
        json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "content_hash": report["content_hash"]}))


if __name__ == "__main__":
    main()
