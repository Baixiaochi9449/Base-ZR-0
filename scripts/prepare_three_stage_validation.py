#!/usr/bin/env python3
"""Independent CPU preparation; this program never launches training."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "lerobot")]

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from utils.stage05_sidecar import build_stage05_sidecar, load_stage05_sidecar, load_stage05_stats, sha256_file, canonical_json_hash

NAMES = ("droid", "household", "tabletop", "rh20t")
IDENTITIES = ("droid", "molmoact_household", "molmoact_tabletop", "rh20t")


def write_json(path, payload):
    with Path(path).open("x") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")


def entries(config):
    registry = yaml.safe_load((ROOT / "dataset2feature.yaml").read_text())
    output = Path(config["output_root"])
    result = {}
    for name, identity, key in zip(NAMES, IDENTITIES, config["dataset_entries"]):
        entry = dict(registry[key], dataset_entry=key, aux_dataset_identity=identity,
            ar_sidecar_path=str(output / name / "ar"), joint_sidecar_path=str(output / name / "joint"),
            frozen_stage_index=str(output / name / "frozen_index.json"), max_transient_retries=0)
        if name == "droid":
            if "full_95658" not in entry["dataset_path"]:
                raise ValueError("full DROID registration required")
            entry.update(optical_flow_data_root=str(Path(entry["dataset_path"]) / "stage06_flow/droid"),
                optical_flow_manifest="manifest.d76a6dd745045034.jsonl", flow_delta_frames=20,
                flow_episode_map=str(output / name / "flow_episode_map.json"))
        exclusion_path = output / name / "flow_exclusions.json"
        if exclusion_path.is_file():
            exclusion = json.loads(exclusion_path.read_text())
            if exclusion.get("version") != 1:
                raise ValueError(f"invalid Flow exclusion record: {exclusion_path}")
            entry["flow_excluded_frames"] = {str(ep): frames for ep, frames in exclusion["episodes"].items()}
        result[name] = entry
    return result


def inventory(config):
    from utils.three_stage_sources import audit_base, file_identity
    output = Path(config["output_root"])
    output.mkdir(parents=True, exist_ok=True)
    doc = (ROOT / "docs/experiments/three_stage_validation_20260908/experiment.md").read_text()
    if not (output / "requested_config.json").exists():
        (output / "experiment.md").write_text(doc)
        write_json(output / "requested_config.json", config)
        write_json(output / "git_identity.json", {
            "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "status": subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True)})
        (output / "worktree.patch").write_bytes(subprocess.check_output(["git", "diff", "HEAD", "--binary"], cwd=ROOT))
    elif json.loads((output / "requested_config.json").read_text()) != config:
        raise ValueError("preparation configuration changed")
    write_json(output / "base_identity.json", audit_base(config["base_model"]))
    summary = {}
    for name, entry in entries(config).items():
        directory = output / name
        directory.mkdir()
        (directory / "experiment.md").write_text(doc)
        root = Path(entry["dataset_path"])
        info = json.loads((root / "meta/info.json").read_text())
        files = [root / "meta" / filename for filename in ("info.json", "stage05_merge.json", "stage05_episode_mapping.jsonl")]
        summary[name] = {"root": str(root), "frames": info["total_frames"], "episodes": info["total_episodes"],
            "fps": info["fps"], "splits": info["splits"], "cameras": entry["camera_keys"],
            "images": {k: info["features"][k] for k in entry["camera_keys"]},
            "metadata": [file_identity(p) for p in files],
            "raw_statistics": [file_identity(p) for p in (root / "meta").glob("*stats*.json")],
            "canonical_statistics_status": "pending independent full-train audit"}
    write_json(output / "dataset_identity.json", summary)


def sidecars(config):
    for name, entry in entries(config).items():
        for phase in ("ar", "joint"):
            print(f"Building {name}/{phase}", flush=True)
            build_stage05_sidecar(root=entry["dataset_path"], output=entry[phase + "_sidecar_path"],
                dataset_id=entry["stats_key"], kind=entry["stage05_kind"], embedded_images=entry["embedded_images"],
                video_backend=entry.get("video_backend", "pyav"), horizon=10, build_joint=phase == "joint")


def sources(config):
    from utils.future_difference_audit import _load_episode_mapping
    from utils.three_stage_sources import file_identity
    output = Path(config["output_root"])
    for name, entry in entries(config).items():
        root = Path(entry["dataset_path"])
        mapping = _load_episode_mapping(root)
        metadata = [row for path in sorted((root / "meta/episodes").rglob("*.parquet"))
                    for row in pq.read_table(path).to_pylist()]
        paths = set((root / "meta").rglob("*.parquet")) | set((root / "meta").glob("*.json*"))
        episodes = []
        for row in metadata:
            episode = int(row["episode_index"])
            data = root / "data" / f"chunk-{int(row['data/chunk_index']):03d}" / f"file-{int(row['data/file_index']):03d}.parquet"
            cameras = {}
            for camera in entry["camera_keys"]:
                prefix = "videos/" + camera
                cameras[camera] = str(data if entry["embedded_images"] else root / prefix /
                    f"chunk-{int(row[prefix + '/chunk_index']):03d}" / f"file-{int(row[prefix + '/file_index']):03d}.mp4")
            paths.update([data, *map(Path, cameras.values())])
            episodes.append({"episode_index": episode, "original_identity": mapping[episode],
                "frames": int(row["length"]), "split": "train", "data": str(data), "cameras": cameras})
        records = {}
        def identify(path):
            before = path.stat()
            record = file_identity(path)
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValueError(f"source changed while hashing: {path}")
            return str(path), dict(record, mtime_ns=after.st_mtime_ns)
        with ThreadPoolExecutor(max_workers=4) as executor:
            for i, (path, record) in enumerate(executor.map(identify, sorted(paths))):
                records[path] = record
                if i % 100 == 0:
                    print(f"Source SHA256 {name}: {i + 1}/{len(paths)}", flush=True)
        write_json(output / name / "source_audit.json", {"status": "passed", "dataset_root": str(root),
            "files": records, "episodes": sorted(episodes, key=lambda row: row["episode_index"]),
            "camera_order": entry["camera_keys"], "source_hash_algorithm": "sha256_entire_file"})


def slots(config):
    from utils.slot_config import SlotConfig
    from utils.slot_supervision import audit_slots, slot_index_implementation_identity
    from utils.future_difference_audit import _resolve_annotation_root
    output = Path(config["output_root"])
    routes = {}
    for name, entry in entries(config).items():
        root = Path(entry["dataset_path"])
        directory = output / name / "slot"
        directory.mkdir(exist_ok=False)
        (directory / "experiment.md").write_text((output / "experiment.md").read_text())
        report, stats, index = audit_slots(root, SlotConfig(), dataset_identity=entry["aux_dataset_identity"],
            camera=entry["camera_keys"][0], allow_unannotated_episodes=name == "droid",
            progress=lambda n: print(f"Slot {name}: {n}", flush=True))
        source_root, _ = _resolve_annotation_root(root, None)
        stats.update(anchor_contract={"anchors_sha256": canonical_json_hash(index["anchors"]),
            "mapping_sha256": sha256_file(root / "meta/stage05_episode_mapping.jsonl"), "annotation_root": str(source_root)},
            index_implementation_identity=slot_index_implementation_identity())
        index.update(version=2, stats_sha256=canonical_json_hash(stats))
        write_json(directory / "slot_audit.json", report)
        write_json(directory / "slot_supervision_stats.json", stats)
        write_json(directory / "slot_anchor_index.json", index)
        routes[entry["aux_dataset_identity"]] = f"{name}/slot"
    write_json(output / "slot_routes.json", {"version": 1, "datasets": routes})


def freeze(config):
    output = Path(config["output_root"])
    counts = {}
    for name, entry in entries(config).items():
        directory = output / name
        manifests = {phase: load_stage05_sidecar(entry[phase + "_sidecar_path"], verify_source=True)
                     for phase in ("ar", "joint")}
        source_audit = json.loads((directory / "source_audit.json").read_text())
        if (source_audit["dataset_root"] != entry["dataset_path"]
                or any(manifest["dataset_root"] != entry["dataset_path"] for manifest in manifests.values())):
            raise ValueError("source/sidecar dataset root changed during preparation")
        manifest = manifests["joint"]
        n = manifest["counts"]["source_frames"]
        packed = np.load(directory / "joint/validity_packed.npy", mmap_mode="r")
        flags = np.unpackbits(packed, axis=0)[:n].astype(bool)
        quality = np.zeros(n, dtype=bool)
        reasons = {}
        episode_rows = pq.read_table(directory / "joint/episodes.parquet").to_pylist()
        for row in episode_rows:
            quality[row["dataset_from_index"]:row["dataset_to_index"]] = row["episode_allowed"]
            if not row["episode_allowed"]:
                key = row["exclusion_reason"]
                reasons[key] = reasons.get(key, 0) + row["length"]
        slot_index = json.loads((directory / "slot/slot_anchor_index.json").read_text())
        anchors = np.zeros(n, dtype=bool)
        for _, _, value in slot_index["anchors"]:
            if value["valid"]:
                anchors[value["global_index"]] = True
        dual = flags[:, 0] & flags[:, 1]
        selections = {"stage12": quality & dual & flags[:, 2] & anchors, "stage3": quality & dual}
        admissions = []
        for row in episode_rows:
            start, stop = row["dataset_from_index"], row["dataset_to_index"]
            admissions.append({"episode_index": row["episode_index"], "source_frames": row["length"],
                "dataset_from_index": start, "dataset_to_index": stop, "quality_allowed": row["episode_allowed"],
                "exclusion_reason": row["exclusion_reason"], "split": "train",
                **{phase + "_valid_samples": int(valid[start:stop].sum()) for phase, valid in selections.items()}})
        admission_path = directory / "episode_admission.parquet"
        if admission_path.exists():
            raise ValueError(f"refusing overwritten episode admission audit: {admission_path}")
        pq.write_table(pa.Table.from_pylist(admissions), admission_path)
        stat_path = directory / "joint/stats.json"
        stats = load_stage05_stats(stat_path, expected_stats_key=entry["stats_key"])
        report = {"version": 1, "dataset_root": entry["dataset_path"], "source_frames": n,
            "source_audit": {"path": str(directory / "source_audit.json"), "sha256": sha256_file(directory / "source_audit.json")},
            "episode_admission": {"path": str(admission_path), "sha256": sha256_file(admission_path)},
            "quality": int(quality.sum()), "dual_camera": int(dual.sum()), "legal_ar": int(flags[:, 2].sum()),
            "slot_anchor_valid": int(anchors.sum()), "fm_eligible": int(flags[:, 4].sum()),
            "independent_exclusions": {"quality": int((~quality).sum()), "dual_camera": int((~dual).sum()),
                "legal_ar": int((~flags[:, 2]).sum()), "valid_slot_anchor": int((~anchors).sum())},
            "stage12_sequential_exclusions": {"quality": int((~quality).sum()),
                "dual_camera_after_quality": int((quality & ~dual).sum()),
                "ar_after_quality_dual_camera": int((quality & dual & ~flags[:, 2]).sum()),
                "slot_after_quality_dual_camera_ar": int((quality & dual & flags[:, 2] & ~anchors).sum())},
            "excluded_quality": reasons, "missing_wrist_after_quality": int((quality & flags[:, 0] & ~flags[:, 1]).sum()),
            "sidecar_manifest_sha256": {phase: sha256_file(directory / phase / "manifest.json") for phase in manifests},
            "slot_index_sha256": sha256_file(directory / "slot/slot_anchor_index.json"),
            "statistics": {"path": str(stat_path), "sha256": sha256_file(stat_path),
                "stats_key": entry["stats_key"], "content_hash": stats["content_hash"], "dataset_root": entry["dataset_path"]}}
        for phase, valid in selections.items():
            values = np.flatnonzero(valid).astype(np.int64)
            path = directory / (phase + "_indices.npy")
            if path.exists() or not len(values):
                raise ValueError(f"refusing empty or overwritten frozen index: {path}")
            np.save(path, values, allow_pickle=False)
            report[phase] = {"file": path.name, "sha256": sha256_file(path), "count": len(values)}
        report["content_hash"] = canonical_json_hash(report)
        write_json(directory / "frozen_index.json", report)
        counts[name] = {phase: report[phase]["count"] for phase in selections}
    summary = {"counts": counts, "expected_stage3_crosscheck_only": 24059815}
    for phase in ("stage12", "stage3"):
        total = sum(row[phase] for row in counts.values())
        summary[phase] = {"total": total, "natural_frame_ratios": {name: row[phase] / total for name, row in counts.items()}}
    write_json(output / "frozen_counts.json", summary)
    write_json(output / "aux_dataset_routes.json", {"version": 1,
        "datasets": {entry["dataset_entry"]: entry for entry in entries(config).values()}})


def flow(config):
    from utils.future_difference_audit import _load_episode_mapping
    from utils.aux_data_contract import flow_contract, public_flow_contract
    from utils.optical_flow_reader import OpticalFlowReader
    output = Path(config["output_root"])
    for name, entry in entries(config).items():
        manifest_path = Path(entry["optical_flow_data_root"]) / entry["optical_flow_manifest"]
        records = [json.loads(line) for line in manifest_path.read_text().splitlines() if line.strip()]
        if name == "droid":
            mapping = _load_episode_mapping(Path(entry["dataset_path"]))
            source_to_full = {row["old_episode_index"]: episode for episode, row in mapping.items()}
            if len(source_to_full) != len(mapping):
                raise ValueError("full DROID original source identity is not unique")
            matches = [{"full_episode": source_to_full[row["source_episode_index"]],
                "flow_episode": row["merged_episode_index"], "source_episode": row["source_episode_index"]} for row in records]
            write_json(output / name / "flow_episode_map.json", {"version": 1,
                "dataset_root": entry["dataset_path"], "source_manifest": str(manifest_path),
                "source_manifest_sha256": sha256_file(manifest_path),
                "target_mapping_sha256": sha256_file(Path(entry["dataset_path"]) / "meta/stage05_episode_mapping.jsonl"),
                "matches": matches})
        contract = flow_contract(entry)
        reader = OpticalFlowReader(entry["optical_flow_data_root"], entry["optical_flow_manifest"],
            delta_frames=entry["flow_delta_frames"], contract=contract)
        frames, nominal = 0, 0
        try:
            for i, episode in enumerate(reader.episodes):
                handle = reader._handle(episode)
                frames += len(handle["frame_index"])
                nominal += int(((handle["actual_delta_frames"][:] == reader.delta_frames) &
                    (handle["label_source"][:] == 1)).sum())
                if i % 100 == 0:
                    print(f"Flow {name}: {i + 1}/{len(reader.episodes)}", flush=True)
        finally:
            reader.close()
        write_json(output / name / "flow_audit.json", {"status": "passed", "contract": public_flow_contract(contract),
            "episodes": len(reader.episodes), "frames": frames, "nominal_delta_rows": nominal,
            "hdf5_integrity": "all original manifest SHA256, metadata, frames, timestamps, FPS, units and source joins verified",
            "pixel_masks": "original artifact integrity verified; finite/binary and pooled masks checked lazily per training sample"})


def flow_origin(config):
    from utils.future_difference_audit import _load_episode_mapping
    from utils.three_stage_sources import file_identity
    entry = entries(config)["droid"]
    root = Path(entry["dataset_path"])
    flow_root = Path(entry["optical_flow_data_root"])
    records = [json.loads(line) for line in (flow_root / entry["optical_flow_manifest"]).read_text().splitlines() if line.strip()]
    generations = {row["generation_identity"] for row in records}
    if len(generations) != 1:
        raise ValueError("DROID Flow has ambiguous generation identities")
    paths = sorted(flow_root.glob(f"stage06_config.{next(iter(generations))[:16]}.*.json"))
    if not paths:
        raise ValueError("original DROID Flow generation configuration is unavailable")
    full_merge = json.loads((root / "meta/stage05_merge.json").read_text())
    identities = []
    for path in paths:
        original = json.loads(path.read_text())
        previous = Path(original["merged_dataset_root"])
        previous_merge = json.loads((previous / "meta/stage05_merge.json").read_text())
        if (original["source_dataset_root"] != full_merge["source_dataset"]
                or any(previous_merge[key] != full_merge[key] for key in ("source_dataset", "source_identity_sha256"))
                or original["camera_key"] != entry["camera_keys"][0] or original["delta_frames"] != entry["flow_delta_frames"]
                or (original["output_height"], original["output_width"]) != (224, 224)):
            raise ValueError("original/full DROID Flow source identity or camera/delta/geometry differs")
        old_mapping = _load_episode_mapping(previous)
        if any(old_mapping[row["merged_episode_index"]]["old_episode_index"] != row["source_episode_index"] for row in records):
            raise ValueError("original DROID Flow manifest/episode mapping differs")
        identities.extend(file_identity(p) for p in (path, previous / "meta/stage05_merge.json", previous / "meta/stage05_episode_mapping.jsonl"))
    write_json(Path(config["output_root"]) / "droid/flow_source_identity.json", {"status": "passed",
        "original_source_dataset": full_merge["source_dataset"], "original_source_identity_sha256": full_merge["source_identity_sha256"],
        "original_metadata": identities, "manifest": file_identity(flow_root / entry["optical_flow_manifest"]),
        "episodes_verified": len(records), "partial_statistics_read_or_reused": False})


def tokens(config):
    from transformers import AutoProcessor
    from scripts.audit_stage05_token_lengths import ExactLengthMeasurer, _audit_dataset, _processor_runtime_identity
    output = Path(config["output_root"])
    processor = AutoProcessor.from_pretrained(config["base_model"], local_files_only=True)
    measurer = ExactLengthMeasurer(processor)
    reports = {}
    for name, entry in entries(config).items():
        reports[name] = _audit_dataset(name, Path(entry["dataset_path"]), output / name / "ar", measurer)
        direct_max = max(measurer._rendered_context(str(row["task"]), 2) for row in
            pq.read_table(output / name / "ar/episodes.parquet").to_pylist() if row["episode_allowed"])
        reports[name]["all_quality_task_context_max"] = direct_max
        if max(reports[name]["maximum_sample"]["tokens"], direct_max) > config["max_length"]:
            raise ValueError(f"base processor truncates prepared data: {name}")
    write_json(output / "token_audit.json", {"status": "passed", "datasets": reports,
        "processor": _processor_runtime_identity(processor, Path(config["base_model"])), "vision": measurer.vision})


def normalization(config):
    import torch
    from scripts.audit_stage05_roundtrip import _audit_dataset
    from utils.normalization import min_max_norm, min_max_denorm
    output = Path(config["output_root"])
    reports = {}
    for name, entry in entries(config).items():
        reports[name] = _audit_dataset(name, Path(entry["dataset_path"]), entry["stage05_kind"],
            output / name / "joint", 8, False, horizon=10)
        payload = load_stage05_stats(output / name / "joint/stats.json", expected_stats_key=entry["stats_key"])
        # Exercise both state/action routes with each dataset's own saved values.
        for key in ("state", "actions"):
            stats = {q: np.asarray(payload["statistics"][key][q], dtype=np.float32) for q in ("q01", "q99")}
            values = torch.tensor(np.stack([stats["q01"], stats["q99"], (stats["q01"] + stats["q99"]) / 2]))
            error = float((min_max_denorm(min_max_norm(values, stats, True), stats) - values).abs().max())
            if error > 1e-5:
                raise ValueError(f"{name}/{key}: normalization roundtrip failed")
            reports[name][key + "_roundtrip_max_error"] = error
    constant = {"q01": np.zeros(7, dtype=np.float32), "q99": np.zeros(7, dtype=np.float32)}
    actual = min_max_norm(torch.zeros(1, 7), constant, True)
    if not torch.isfinite(actual).all() or not torch.equal(min_max_denorm(actual, constant), torch.zeros(1, 7)):
        raise ValueError("constant-dimension normalization failed")
    write_json(output / "normalization_audit.json", {"status": "passed", "datasets": reports,
        "constant_dimension_finite_and_exact": True, "clip_range": [-15, 15],
        "limits": "real-data stratified H10 action roundtrip; full clipping frequency is reported only where proven"})


def commands(config):
    import importlib.metadata
    from scripts.run_three_stage_validation import training_command, STAGES, validate_config
    output = Path(config["output_root"])
    validate_config(config)
    base = json.loads((output / "base_identity.json").read_text())
    write_json(output / "action_expert_config_h10.json", base["action_expert_config"])
    accelerate = yaml.safe_load((ROOT / "accelerate_configs/structured_slots_zero2_bf16.yaml").read_text())
    accelerate["deepspeed_config"].update(gradient_accumulation_steps=2, train_micro_batch_size_per_gpu=16,
        train_batch_size=128, gradient_clipping=1.)
    with (output / "accelerate.yaml").open("x") as stream:
        yaml.safe_dump(accelerate, stream)
    write_json(output / "validation_commands.json", [training_command(config, stage, step)[1] for stage in STAGES for step in (50, 100)])
    write_json(output / "formal_commands_not_executed.json", [training_command(config, stage, None, formal=True)[1] for stage in STAGES])
    write_json(output / "runtime_environment.json", {"python": sys.executable, "python_version": sys.version,
        "packages": {name: importlib.metadata.version(name) for name in
            ("torch", "transformers", "accelerate", "deepspeed", "numpy", "pyarrow", "scipy", "h5py", "Pillow", "av", "wandb")},
        "gradient_checkpointing": {"stage1_ar": True, "stage2_aux": False, "stage3_joint": True},
        "processor_secondary_resize": False, "geometric_augmentation": False, "training_updates": [0, 0, 0]})
    routes = json.loads((output / "aux_dataset_routes.json").read_text())
    for entry in routes["datasets"].values():
        entry["preparation_audit_cache"] = str(output / "audit_snapshot.json")
    write_json(output / "cached_aux_dataset_routes.json", routes)


def archive(config):
    """Persist existing pass/fail evidence; never read original payloads again."""
    from utils.preparation_audit_cache import stat_identity
    from utils.three_stage_preflight import implementation_identity
    output = Path(config["output_root"])
    files, blockers, gates = {}, [], {}

    def remember(path, digest, evidence, *, size=None, mtime_ns=None):
        path, evidence = Path(path).absolute(), Path(evidence)
        identity = stat_identity(path)
        if (identity["ctime_ns"] > evidence.stat().st_mtime_ns
                or (size is not None and identity["size"] != size)
                or (mtime_ns is not None and identity["mtime_ns"] != mtime_ns)):
            raise ValueError(f"source changed after its audit; refusing cached reuse: {path}")
        record = {"sha256": digest, "stat": identity, "evidence": str(evidence)}
        if str(path) in files and files[str(path)]["sha256"] != digest:
            raise ValueError(f"conflicting saved SHA256 evidence: {path}")
        files[str(path)] = record

    base_path = output / "base_identity.json"
    for record in json.loads(base_path.read_text())["files"]:
        remember(record["path"], record["sha256"], base_path, size=record["size"])
    known_artifact_hashes, artifact_evidence = {}, {}
    for name, entry in entries(config).items():
        directory = output / name
        source_path = directory / "source_audit.json"
        source = json.loads(source_path.read_text())
        if source["status"] != "passed":
            raise ValueError(f"source audit is incomplete: {source_path}")
        for path, record in source["files"].items():
            if str(Path(path).resolve()) != record["path"]:
                raise ValueError(f"audited source resolution changed: {path}")
            remember(path, record["sha256"], source_path, size=record["size"], mtime_ns=record["mtime_ns"])
        index_path = directory / "slot/slot_anchor_index.json"
        slot = json.loads(index_path.read_text())
        for path, digest in slot["source_hashes"].items():
            remember(path, digest, index_path)
        frozen = json.loads((directory / "frozen_index.json").read_text())
        known_artifact_hashes.update({str(index_path): frozen["slot_index_sha256"],
            str(source_path): frozen["source_audit"]["sha256"],
            frozen["statistics"]["path"]: frozen["statistics"]["sha256"],
            frozen["episode_admission"]["path"]: frozen["episode_admission"]["sha256"]})
        for phase in ("ar", "joint"):
            manifest_path = directory / phase / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            known_artifact_hashes[str(manifest_path)] = frozen["sidecar_manifest_sha256"][phase]
            for filename, digest in manifest["files"].items():
                known_artifact_hashes[str(directory / phase / filename)] = digest
        for selection in ("stage12", "stage3"):
            known_artifact_hashes[str(directory / frozen[selection]["file"])] = frozen[selection]["sha256"]
        for path in known_artifact_hashes:
            if str(Path(path).parent).startswith(str(directory)):
                artifact_evidence[path] = directory / "frozen_index.json"
        flow_path = directory / "flow_audit.json"
        flow = json.loads(flow_path.read_text())
        gates[name] = {"source": "passed", "slot": "passed", "frozen_index": "passed", "flow": flow["status"]}
        if flow["status"] != "passed":
            blockers.append({"dataset": name, "audit": str(flow_path), "reason": flow.get("error", "Flow audit did not pass")})
            continue
        manifest_path = Path(entry["optical_flow_data_root"]) / entry["optical_flow_manifest"]
        remember(manifest_path, flow["contract"]["manifest_sha256"], flow_path)
        for line in manifest_path.read_text().splitlines():
            row = json.loads(line)
            remember(Path(entry["optical_flow_data_root"]) / row["hdf5_path"], row["sha256"], flow_path)
    for filename in ("token_audit.json", "normalization_audit.json", "droid/flow_source_identity.json"):
        report = json.loads((output / filename).read_text())
        gates[filename] = report["status"]
        if report["status"] != "passed":
            blockers.append({"audit": str(output / filename), "reason": "saved audit did not pass"})
    for path in sorted(output.rglob("*")):
        if (path.is_file() and path.name != "preparation_events.jsonl"
                and path.suffix in {".json", ".jsonl", ".npy", ".yaml", ".parquet", ".log"}):
            digest = known_artifact_hashes.get(str(path)) or sha256_file(path)
            if str(path) in artifact_evidence:
                remember(path, digest, artifact_evidence[str(path)])
                continue
            files[str(path.absolute())] = {"sha256": digest, "stat": stat_identity(path), "evidence": "prepared artifact"}
    snapshot = {"version": 1, "status": "blocked" if blockers else "passed", "config": config,
        "created_at": datetime.now(timezone.utc).isoformat(), "files": files, "gates": gates, "blockers": blockers,
        "runtime_implementation": implementation_identity(), "training_updates": [0, 0, 0],
        "reuse_policy": "saved audit only; stat identity checks; changed identity stops without re-audit",
        "producer_compatibility": "original Slot audit/index/statistics preserved; runtime-only cache adaptation does not change audit_slots"}
    write_json(output / "audit_snapshot.json", snapshot)
    if blockers:
        write_json(output / "preparation_blocked.json", {"status": "blocked", "blockers": blockers,
            "audit_snapshot": str(output / "audit_snapshot.json"), "training_updates": [0, 0, 0],
            "automatic_retry": False, "formal_training_started": False})
    print(json.dumps({"status": snapshot["status"], "cached_files": len(files), "blockers": blockers}), flush=True)


def bind_runtime(config):
    """Record a narrow reader-cache adaptation without regenerating any audit."""
    from utils.preparation_audit_cache import PreparationAuditCache
    from utils.three_stage_preflight import implementation_identity
    output = Path(config["output_root"])
    cache = PreparationAuditCache(output / "audit_snapshot.json", require_passed=False)
    if cache.snapshot["config"] != config:
        raise ValueError("runtime binding configuration differs from saved audit")
    implementation = implementation_identity()
    before = cache.snapshot["runtime_implementation"]
    changed = {path for path in set(before) | set(implementation) if before.get(path) != implementation.get(path)}
    if not changed <= {"utils/dataset_spec.py", "utils/preparation_audit_cache.py", "scripts/prepare_three_stage_validation.py"}:
        raise ValueError(f"runtime binding exceeds the reviewed cache adapter scope: {sorted(changed)}")
    write_json(output / "audit_runtime_binding.json", {"version": 1, "audit_snapshot_sha256": cache.sha256,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "runtime_implementation": implementation, "changed_files": sorted(changed),
        "audit_status_unchanged": cache.snapshot["status"], "original_payloads_reaudited": False,
        "purpose": "cached DatasetSpec resolution; original audit producers, results and indexes unchanged"})


def seal(config):
    from utils.three_stage_preflight import implementation_identity, preparation_directory
    from scripts.run_three_stage_validation import validate_config
    from utils.slot_routing import load_slot_supervision
    from utils.frozen_stage_index import load_frozen_stage_index
    from utils.preparation_audit_cache import load_preparation_audit_cache, stat_identity
    output = Path(config["output_root"])
    preparation = preparation_directory(config)
    validate_config(config)
    cache = load_preparation_audit_cache(str(preparation / "audit_snapshot.json"))
    cache.check_all()
    load_slot_supervision(output, audit_cache=cache)
    frozen_counts = json.loads((output / "frozen_counts.json").read_text())
    routes = json.loads((preparation / "cached_aux_dataset_routes.json").read_text())
    expected_entries = entries(config)
    if routes != {"version": 1, "datasets": {entry["dataset_entry"]: dict(entry,
            preparation_audit_cache=str(preparation / "audit_snapshot.json")) for entry in expected_entries.values()}}:
        raise ValueError("prepared dataset routing differs from the fixed contract")
    for name, entry in expected_entries.items():
        for phase, selection in (("ar", "stage12"), ("joint", "stage3")):
            indices, identity = load_frozen_stage_index(entry["frozen_stage_index"],
                dataset_root=entry["dataset_path"], sidecar_root=entry[phase + "_sidecar_path"], phase=phase, audit_cache=cache)
            if len(indices) != frozen_counts["counts"][name][selection]:
                raise ValueError("frozen index count differs from the preparation audit")
            stats = identity["statistics"]
            if stats["dataset_root"] != entry["dataset_path"] or cache.check(stats["path"]) != stats["sha256"]:
                raise ValueError("per-dataset canonical statistics identity changed")
            load_stage05_stats(stats["path"], expected_stats_key=entry["stats_key"])
    base = json.loads((output / "base_identity.json").read_text())
    for record in base["files"]:
        if cache.check(record["path"]) != record["sha256"]:
            raise ValueError("base source changed since inventory")
    original_datasets = json.loads((output / "dataset_identity.json").read_text())
    for name, original in original_datasets.items():
        if original["root"] != expected_entries[name]["dataset_path"]:
            raise ValueError("dataset root changed since the initial inventory")
        for record in original["metadata"] + original["raw_statistics"]:
            if cache.check(record["path"]) != record["sha256"]:
                raise ValueError(f"dataset metadata/statistics changed since inventory: {record['path']}")
    for path in [output / "token_audit.json", output / "normalization_audit.json", output / "droid/flow_source_identity.json",
            *[output / name / filename for name in NAMES for filename in ("flow_audit.json", "source_audit.json")]]:
        if json.loads(path.read_text())["status"] != "passed":
            raise ValueError(f"preparation gate failed: {path}")
    write_json(preparation / "preparation_complete.json", {"version": 2, "status": "passed", "config": config,
        "implementation": implementation_identity(), "audit_snapshot_stat": stat_identity(preparation / "audit_snapshot.json")})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/three_stage_validation_20260908.json")
    parser.add_argument("--phase", choices=("inventory", "sidecars", "sources", "slots", "freeze", "flow", "flow_origin", "tokens", "normalization", "commands", "archive", "bind_runtime", "resolve_flow_exclusions", "seal"), required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if sys.executable != config["python"]:
        raise ValueError("use the specified ZR-0 interpreter")
    from utils.three_stage_preflight import preparation_directory
    snapshot = preparation_directory(config) / "audit_snapshot.json"
    if not snapshot.is_file():
        snapshot = Path(config["output_root"]) / "audit_snapshot.json"
    if snapshot.is_file() and args.phase not in {"seal", "bind_runtime", "resolve_flow_exclusions"}:
        saved = json.loads(snapshot.read_text())
        if saved["config"] != config:
            raise ValueError("saved audit configuration changed; no automatic re-audit")
        print(json.dumps({"audit_snapshot": str(snapshot), "status": saved["status"], "reused": True,
                          "audit_executed": False, "blockers": saved["blockers"]}))
        return
    try:
        if args.phase == "resolve_flow_exclusions":
            from utils.flow_exclusion_resolution import resolve_preparation
            resolve_preparation(config)
        else:
            globals()[args.phase](config)
    except Exception as error:
        output = Path(config["output_root"])
        if output.is_dir():
            record = {"phase": args.phase, "status": "failed", "error": str(error),
                      "time": datetime.now(timezone.utc).isoformat(), "training_updates": [0, 0, 0]}
            with (output / "preparation_events.jsonl").open("a") as stream:
                stream.write(json.dumps(record, sort_keys=True) + "\n")
        raise


if __name__ == "__main__":
    main()
