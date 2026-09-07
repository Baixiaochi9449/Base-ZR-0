"""Stage05 auxiliary provenance and strict lazy Flow file validation."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from utils.future_difference_audit import _load_episode_mapping


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def flow_contract(entry):
    root = Path(entry["dataset_path"]).resolve()
    info = json.loads((root / "meta/info.json").read_text())
    mapping = _load_episode_mapping(root)
    manifest = Path(entry["optical_flow_manifest"])
    if not manifest.is_absolute():
        manifest = Path(entry["optical_flow_data_root"]) / manifest
    if not entry.get("aux_dataset_identity"):
        raise ValueError("Flow registration requires aux_dataset_identity")
    return {"version": 1, "dataset_root": str(root), "dataset_id": entry["aux_dataset_identity"],
            "fps": float(info["fps"]), "camera": entry["camera_keys"][0],
            "nominal_delta_frames": int(entry["flow_delta_frames"]),
            "units": "normalized_source_image_extent", "geometry": "resize_224x224_no_crop",
            "manifest_sha256": digest_file(manifest),
            "mapping_sha256": digest_file(root / "meta/stage05_episode_mapping.jsonl"),
            "mapping": {str(ep): row for ep, row in mapping.items()}}


def public_flow_contract(contract):
    return {key: value for key, value in contract.items() if key != "mapping"}


def validate_flow_file(handle, entry, contract, frames, targets):
    expected = {"schema_version": "stage06_flow_v2", "dataset_id": contract["dataset_id"],
                "camera_key": contract["camera"], "output_height": 224, "output_width": 224}
    for key in ("artifact_identity", "generation_identity", "label_identity", "source_fingerprint",
                "checkpoint_sha256", "model_revision"):
        expected[key] = entry[key]
    for key, value in expected.items():
        if handle.attrs.get(key) != value:
            raise ValueError(f"flow metadata mismatch: {key}")
    if digest_file(handle.filename) != entry["sha256"]:
        raise ValueError("Flow file content differs from manifest sha256")
    ep = entry["merged_episode_index"]
    path = Path(contract["dataset_root"]) / contract["mapping"][str(ep)]["source_data_uri"]
    table = pq.read_table(path, columns=["episode_index", "frame_index", "timestamp"])
    rows = [row for row in table.to_pylist() if row["episode_index"] == ep]
    times = {row["frame_index"]: row["timestamp"] for row in rows}
    if len(times) != len(rows):
        raise ValueError("duplicate source frame timestamp")
    source = handle["source_timestamp_s"][:]
    target = handle["target_timestamp_s"][:]
    actual = handle["actual_delta_s"][:]
    for values in (source, target, actual):
        if values.shape != frames.shape or not np.isfinite(values).all():
            raise ValueError("invalid Flow timestamp shape/value")
    if (not np.allclose(source, [times[int(f)] for f in frames], rtol=0, atol=2e-6)
            or not np.allclose(target, [times[int(f)] for f in targets], rtol=0, atol=2e-6)
            or not np.allclose(actual, target - source, rtol=0, atol=2e-6)
            or not np.allclose(actual, (targets - frames) / contract["fps"], rtol=0, atol=2e-5)):
        raise ValueError("Flow timestamp/source/FPS interval mismatch")
