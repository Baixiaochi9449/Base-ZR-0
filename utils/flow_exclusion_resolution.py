"""Append an authorized Flow-only resolution to the existing one-time audit."""

import json
from datetime import datetime, timezone
from pathlib import Path
import subprocess

import h5py
import numpy as np
import pyarrow.parquet as pq

from utils.preparation_audit_cache import PreparationAuditCache, stat_identity
from utils.stage05_sidecar import sha256_file


def resolve_preparation(config):
    from scripts.prepare_three_stage_validation import entries, write_json
    from scripts.run_three_stage_validation import STAGES, training_command
    from utils.three_stage_preflight import implementation_identity, preparation_directory
    from utils.future_difference_audit import _load_episode_mapping
    from utils.aux_data_contract import flow_contract, public_flow_contract

    root, output = Path(config["output_root"]), preparation_directory(config)
    if output == root:
        raise ValueError("Flow exclusion resolution requires a separate preparation revision")
    previous = PreparationAuditCache(root / "audit_snapshot.json", require_passed=False)
    original_config = {k: v for k, v in config.items() if k != "preparation_revision"}
    if original_config != previous.snapshot["config"]:
        raise ValueError("Flow resolution changed the training or source configuration")
    entry = entries(config)["rh20t"]
    exclusions = entry["flow_excluded_frames"]
    scan_path = root / "rh20t/flow_quantization_scan.log"
    lines = scan_path.read_text().splitlines()
    scan = json.loads("\n".join(lines[lines.index("{"):]))
    expected = {str(row["episode"]): row["frames"] for row in scan["failures"]}
    if exclusions != expected or len(exclusions) != 12 or sum(map(len, exclusions.values())) != 108:
        raise ValueError("Flow exclusions differ from the authorized 12-episode/108-row scan")
    completed_path = root / "rh20t/flow_audit.json"
    completed = json.loads(completed_path.read_text())
    contract = flow_contract(entry)
    if (completed["status"] != "passed" or completed["contract"] != public_flow_contract(contract)
            or completed["excluded_flow_rows"] != 108 or completed["prefix_records_reused"] != 491):
        raise ValueError("RH20T continuation has not completed under the authorized exclusions")
    manifest_path = Path(entry["optical_flow_data_root"]) / entry["optical_flow_manifest"]
    manifest = [json.loads(line) for line in manifest_path.read_text().splitlines() if line.strip()]
    if len(manifest) != scan["total_records"] or len(manifest) != 8142:
        raise ValueError("RH20T manifest count changed")
    mapping = _load_episode_mapping(Path(entry["dataset_path"]))
    prefix_nominal = excluded_nominal = 0
    source_checks = []
    # Only missing prefix counts and the approved exception rows are read here.
    # The completed payload SHA256 audits are reused for all original HDF5 files.
    for position, record in enumerate(manifest):
        episode = record["merged_episode_index"]
        excluded = exclusions.get(str(episode), [])
        if position >= 491 and not excluded:
            continue
        path = Path(entry["optical_flow_data_root"]) / record["hdf5_path"]
        with h5py.File(path, "r") as handle:
            nominal = (handle["actual_delta_frames"][:] == entry["flow_delta_frames"]) & (handle["label_source"][:] == 1)
            if position < 491:
                prefix_nominal += int(nominal.sum())
            if not excluded:
                continue
            frames, targets = handle["frame_index"][:], handle["target_frame_index"][:]
            selected = np.flatnonzero(np.isin(frames, excluded))
            if len(selected) != len(excluded):
                raise ValueError("approved Flow exclusion names an absent frame")
            excluded_nominal += int(nominal[selected].sum())
            source_path = Path(entry["dataset_path"]) / mapping[episode]["source_data_uri"]
            table = pq.read_table(source_path, columns=["frame_index", "timestamp"], filters=[("episode_index", "=", episode)])
            times = {row["frame_index"]: row["timestamp"] for row in table.to_pylist()}
            source = handle["source_timestamp_s"][:][selected]
            target = handle["target_timestamp_s"][:][selected]
            actual = handle["actual_delta_s"][:][selected]
            if (not np.allclose(source, [times[int(f)] for f in frames[selected]], rtol=0, atol=2e-6)
                    or not np.allclose(target, [times[int(f)] for f in targets[selected]], rtol=0, atol=2e-6)
                    or not np.allclose(actual, target - source, rtol=0, atol=2e-6)):
                raise ValueError("excluded Flow rows violate a contract beyond the approved FPS residual")
            source_checks.append({"episode": episode, "frames": excluded, "source_alignment": "passed",
                "excluded_nominal_rows": int(nominal[selected].sum())})

    output.mkdir(parents=True, exist_ok=False)
    (output / "experiment.md").write_text((root / "experiment.md").read_text() +
        "\nFlow resolution: explicit user-authorized 108 Flow rows; original data and all other objectives retained.\n")
    write_json(output / "requested_config.json", config)
    repository = Path(__file__).resolve().parents[1]
    write_json(output / "git_identity.json", {
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repository, text=True).strip(),
        "status": subprocess.check_output(["git", "status", "--short"], cwd=repository, text=True),
        "automatic_staging": False, "commit_created": False})
    (output / "worktree.patch").write_bytes(subprocess.check_output(
        ["git", "diff", "HEAD", "--binary"], cwd=repository))
    routes = {"version": 1, "datasets": {item["dataset_entry"]: dict(item,
        preparation_audit_cache=str(output / "audit_snapshot.json")) for item in entries(config).values()}}
    write_json(output / "cached_aux_dataset_routes.json", routes)
    write_json(output / "validation_commands.json", [training_command(config, stage, step)[1] for stage in STAGES for step in (50, 100)])
    write_json(output / "formal_commands_not_executed.json", [training_command(config, stage, None, formal=True)[1] for stage in STAGES])
    report = dict(completed, frames=sum(row["frame_count"] for row in manifest),
        nominal_delta_rows=prefix_nominal + completed["nominal_delta_rows"] + 108 - excluded_nominal,
        excluded_nominal_delta_rows=excluded_nominal, excluded_non_nominal_rows=108 - excluded_nominal,
        approved_exception_source_alignment=source_checks, previous_audit_snapshot_sha256=previous.sha256,
        sample_indices_changed=False, state_action_statistics_changed=False,
        counts_note="total across all 8142 episodes; exception rows counted by their actual nominal-delta mask")
    write_json(output / "rh20t_flow_audit.json", report)
    old_failure_path = str(root / "rh20t/flow_audit.json")
    archived_failure_path = root / "rh20t/flow_audit_failed_original.json"
    if sha256_file(archived_failure_path) != previous.files[old_failure_path]["sha256"]:
        raise ValueError("original failed audit was not retained exactly")
    files = {}
    for i, (path, record) in enumerate(previous.files.items()):
        if path == old_failure_path:
            continue
        previous.check(path)
        files[path] = record
        if i % 10000 == 0:
            print(f"Reused saved identities: {i}/{len(previous.files)}", flush=True)
    prefix_cutoff = (root / "flow_preparation.log").stat().st_mtime_ns
    suffix_cutoff = completed_path.stat().st_mtime_ns
    for position, row in enumerate(manifest):
        path = Path(entry["optical_flow_data_root"]) / row["hdf5_path"]
        identity = stat_identity(path)
        if identity["size"] != row["size_bytes"] or identity["ctime_ns"] > (prefix_cutoff if position < 491 else suffix_cutoff):
            raise ValueError(f"original RH20T Flow file changed since its audit: {path}")
        files[str(path)] = {"sha256": row["sha256"], "stat": identity,
            "evidence": str(root / "flow_preparation.log") if position < 491 else str(completed_path)}
    additions = [previous.path, manifest_path, completed_path, archived_failure_path, scan_path,
        root / "rh20t/flow_exclusions.json", root / "rh20t/flow_resume_after_all_exclusions.log",
        output / "worktree.patch",
        *[path for path in output.iterdir() if path.suffix == ".json"]]
    for path in additions:
        files[str(path)] = {"sha256": sha256_file(path), "stat": stat_identity(path), "evidence": "authorized Flow resolution"}
    gates = dict(previous.snapshot["gates"])
    gates["rh20t"] = dict(gates["rh20t"], flow="passed_with_explicit_exclusions")
    write_json(output / "audit_snapshot.json", dict(previous.snapshot, status="passed", config=config,
        files=files, gates=gates, blockers=[], runtime_implementation=implementation_identity(),
        created_at=datetime.now(timezone.utc).isoformat(),
        runtime_environment_path=str(root / "runtime_environment.json"),
        previous_audit_snapshot_sha256=previous.sha256, flow_exclusion_resolution=str(output / "rh20t_flow_audit.json")))
    print(json.dumps({"status": "resolved", "excluded_rows": 108, "excluded_nominal_rows": excluded_nominal,
        "flow_frames": report["frames"], "nominal_delta_rows": report["nominal_delta_rows"], "revision": str(output)}), flush=True)
