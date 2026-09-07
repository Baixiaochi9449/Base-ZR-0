#!/usr/bin/env python3
"""Publish independent four-source sidecars, Slot audits and light Flow indexes."""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import h5py
import numpy as np
import pyarrow.parquet as pq
import yaml

from utils.aux_data_contract import digest_file, flow_contract, public_flow_contract
from utils.slot_config import SlotConfig
from utils.slot_supervision import audit_slots, slot_index_implementation_identity
from utils.optical_flow_checkpoint import json_hash
from utils.future_difference_audit import _resolve_annotation_root
from utils.stage05_sidecar import build_stage05_sidecar

ENTRIES = {"droid": "stage05_droid_partial_mixed", "household": "stage05_household_mixed",
           "tabletop": "stage05_tabletop_mixed", "rh20t": "stage05_rh20t_mixed"}
IDENTITIES = {"droid": "droid", "household": "molmoact_household", "tabletop": "molmoact_tabletop", "rh20t": "rh20t"}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def candidates(entry, output):
    contract = flow_contract(entry)
    root = Path(entry["dataset_path"])
    flow_root = Path(entry["optical_flow_data_root"])
    records = [json.loads(line) for line in (flow_root / entry["optical_flow_manifest"]).read_text().splitlines()]
    source_paths = sorted({root / row["source_data_uri"] for row in contract["mapping"].values()})
    global_by_frame = {}
    for path in source_paths:
        for row in pq.read_table(path, columns=["episode_index", "frame_index", "index"]).to_pylist():
            global_by_frame[(row["episode_index"], row["frame_index"])] = row["index"]
    selected = []
    for i, record in enumerate(records):
        ep = record["merged_episode_index"]
        if record["dataset_id"] != contract["dataset_id"] or record["source_episode_index"] != contract["mapping"][str(ep)]["old_episode_index"]:
            raise ValueError("Flow candidate mapping conflict")
        with h5py.File(flow_root / record["hdf5_path"], "r") as handle:
            if handle.attrs["fps"] != contract["fps"] or handle.attrs["nominal_delta_frames"] != contract["nominal_delta_frames"]:
                raise ValueError("Flow candidate time contract conflict")
            frames, fraction = handle["frame_index"][:], handle["valid_fraction"][:]
            if len(np.unique(frames)) != len(frames) or not np.isfinite(fraction).all() or ((fraction < 0) | (fraction > 1)).any():
                raise ValueError("Flow candidate frame/fraction corruption")
            valid = (handle["actual_delta_frames"][:] == contract["nominal_delta_frames"]) & (handle["label_source"][:] == 1) & (fraction > 0)
            selected.extend(global_by_frame[(ep, int(frame))] for frame in frames[valid])
        if i % 1000 == 0:
            print(f"Flow metadata {entry['dataset_entry']}: {i}/{len(records)}", flush=True)
    index = output / "flow_candidates.npy"
    np.save(index, np.asarray(sorted(selected), dtype=np.int64), allow_pickle=False)
    write_json(index.with_suffix(".json"), {"version": 1, "flow_contract": public_flow_contract(contract),
        "sha256": digest_file(index), "count": len(selected),
        "rule": "nominal_delta_and_source1_and_positive_valid_fraction; conservative candidates; exact pooled masks checked per sample"})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=["sidecars", "slot", "slot-index", "flow", "tokens"], required=True)
    parser.add_argument("--index-version", type=int, default=2)
    parser.add_argument("--datasets", nargs="+", choices=list(ENTRIES), default=list(ENTRIES))
    args = parser.parse_args()
    registry = yaml.safe_load((ROOT / "dataset2feature.yaml").read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    for name in args.datasets:
        entry = dict(registry[ENTRIES[name]], dataset_entry=ENTRIES[name], aux_dataset_identity=IDENTITIES[name])
        output = args.output / name
        output.mkdir(exist_ok=True)
        if args.phase == "sidecars":
            for mode in ("aux", "joint"):
                path = output / mode
                if path.exists():
                    raise FileExistsError(f"refusing to overwrite {path}")
                print(f"Building {name}/{mode}", flush=True)
                result = build_stage05_sidecar(root=entry["dataset_path"], output=path, dataset_id=entry["stats_key"],
                    kind=entry["stage05_kind"], embedded_images=entry["embedded_images"], horizon=32, build_joint=mode == "joint")
                print(json.dumps({"name": name, "mode": mode, "counts": result["counts"]}), flush=True)
        elif args.phase == "slot":
            path = output / "slot"
            path.mkdir(exist_ok=False)
            report, stats, index = audit_slots(entry["dataset_path"], SlotConfig(), dataset_identity=IDENTITIES[name],
                camera=entry["camera_keys"][0], progress=lambda count: print(f"Slot {name}: {count}", flush=True))
            write_json(path / "slot_audit.json", report)
            write_json(path / "slot_supervision_stats.json", stats)
            write_json(path / "slot_anchor_index.json", index)
        elif args.phase == "slot-index":
            source = output / "slot"
            path = output / f"slot_v{args.index_version}"
            path.mkdir(exist_ok=False)
            stats = json.loads((source / "slot_supervision_stats.json").read_text())
            index = json.loads((source / "slot_anchor_index.json").read_text())
            if index["stats_sha256"] != json_hash(stats) or json_hash(index["source_hashes"]) != stats["source_identity"] or json_hash(index["data_hashes"]) != stats["data_identity"]:
                raise ValueError("cannot derive index from corrupt audit")
            source_root, _ = _resolve_annotation_root(Path(entry["dataset_path"]), None)
            stats["anchor_contract"] = {"anchors_sha256": json_hash(index["anchors"]),
                "mapping_sha256": digest_file(Path(entry["dataset_path"]) / "meta/stage05_episode_mapping.jsonl"),
                "annotation_root": str(source_root)}
            stats["index_implementation_identity"] = slot_index_implementation_identity()
            stats["audit_derivation"] = {name: digest_file(source / name)
                for name in ("slot_supervision_stats.json", "slot_anchor_index.json")}
            index.update(version=2, stats_sha256=json_hash(stats))
            write_json(path / "slot_supervision_stats.json", stats)
            write_json(path / "slot_anchor_index.json", index)
        elif args.phase == "flow":
            if (output / "flow_candidates.npy").exists():
                raise FileExistsError("Flow candidates already exist")
            candidates(entry, output)
        else:
            from transformers import AutoProcessor
            from scripts.audit_stage05_token_lengths import ExactLengthMeasurer, _audit_dataset, _processor_runtime_identity, _repository_implementation_identity
            path = output / "token_audit.json"
            if path.exists():
                raise FileExistsError(path)
            processor_path = Path("/opt/data/private/lq/models/Qwen3-VL-2B-Instruct")
            processor = AutoProcessor.from_pretrained(processor_path)
            result = _audit_dataset(name, Path(entry["dataset_path"]), output / "joint", ExactLengthMeasurer(processor))
            write_json(path, {"version": 1, "dataset": entry["dataset_entry"], "counts": result,
                "sidecar_sha256": digest_file(output / "joint/manifest.json"),
                "processor_identity": _processor_runtime_identity(processor, processor_path),
                "implementation_identity": _repository_implementation_identity()})
    if args.phase == "slot":
        write_json(args.output / "slot_routes.json", {"version": 1,
            "datasets": {identity: f"{name}/slot" for name, identity in IDENTITIES.items()}})
    if args.phase == "slot-index":
        routes = args.output / f"slot_sources_v{args.index_version}"
        routes.mkdir(exist_ok=False)
        write_json(routes / "slot_routes.json", {"version": 1,
            "datasets": {identity: f"../{name}/slot_v{args.index_version}" for name, identity in IDENTITIES.items()}})


if __name__ == "__main__":
    main()
