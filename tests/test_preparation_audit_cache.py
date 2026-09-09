"""Saved audit reuse must avoid payload scans and reject changed sources."""

import json
import os
from pathlib import Path

import numpy as np
import pytest

from utils.preparation_audit_cache import PreparationAuditCache, stat_identity, load_cached_sidecar
from utils.stage05_sidecar import sha256_file


def snapshot(tmp_path, paths, *, status="passed"):
    path = tmp_path / "audit_snapshot.json"
    path.write_text(json.dumps({"version": 1, "status": status, "blockers": ["timestamp mismatch"] if status != "passed" else [],
        "files": {str(p.absolute()): {"sha256": sha256_file(p), "stat": stat_identity(p)} for p in paths}}))
    return path


def test_cache_never_reads_source_payload_and_detects_same_size_replacement(tmp_path, monkeypatch):
    source = tmp_path / "source.bin"
    source.write_bytes(b"original")
    cache = PreparationAuditCache(snapshot(tmp_path, [source]))
    original_open = Path.open
    with monkeypatch.context() as patch:
        def no_payload(path, *args, **kwargs):
            if path == source:
                pytest.fail("cached startup reread the original payload")
            return original_open(path, *args, **kwargs)
        patch.setattr(Path, "open", no_payload)
        cache.check_all()
        cache.check_all()
    timestamp = source.stat().st_mtime_ns
    replacement = tmp_path / "replacement.bin"
    replacement.write_bytes(b"modified")
    os.utime(replacement, ns=(timestamp, timestamp))
    replacement.replace(source)
    with pytest.raises(ValueError, match="changed since the saved audit"):
        cache.check(source)


def test_cache_rejects_symlink_retarget_missing_identity_and_failed_audit(tmp_path):
    first, second, link = (tmp_path / name for name in ("first", "second", "link"))
    first.write_bytes(b"same")
    second.write_bytes(b"same")
    link.symlink_to(first)
    cache = PreparationAuditCache(snapshot(tmp_path, [link]))
    with pytest.raises(ValueError, match="absent from the saved audit"):
        cache.check(second)
    link.unlink()
    link.symlink_to(second)
    with pytest.raises(ValueError, match="changed since the saved audit"):
        cache.check(link)
    path = snapshot(tmp_path, [first], status="blocked")
    with pytest.raises(ValueError, match="timestamp mismatch"):
        PreparationAuditCache(path)


def test_cached_sidecar_reuses_audited_arrays_and_checks_requested_generation(tmp_path, monkeypatch):
    from test_stage05_mixed_pretraining import _toy_molmo
    from utils.stage05_sidecar import build_stage05_sidecar
    root, sidecar = tmp_path / "source", tmp_path / "ar"
    _toy_molmo(root)
    build_stage05_sidecar(root=root, output=sidecar, dataset_id="own", kind="molmo",
        embedded_images=True, horizon=10, build_joint=False)
    paths = [p for p in sidecar.rglob("*") if p.is_file()]
    cache = PreparationAuditCache(snapshot(tmp_path, paths))
    monkeypatch.setattr("utils.stage05_sidecar._source_inventory", lambda *a, **k: pytest.fail("source inventory repeated"))
    monkeypatch.setattr("utils.stage05_sidecar._validate_sidecar_shapes", lambda *a, **k: pytest.fail("array audit repeated"))
    monkeypatch.setattr("utils.stage05_sidecar.sha256_file", lambda *a: pytest.fail("array hashing repeated"))
    manifest = load_cached_sidecar(sidecar, audit_cache=cache, expected_generation={"horizon": 10})
    assert manifest["counts"]["source_frames"] == 3
    with pytest.raises(ValueError, match="generation mismatch"):
        load_cached_sidecar(sidecar, audit_cache=cache, expected_generation={"horizon": 50})


def test_cached_flow_reopen_skips_file_audit_but_keeps_sample_validation(tmp_path, monkeypatch):
    import h5py
    from test_optical_flow_aux import fixture_manifest
    from utils.optical_flow_reader import OpticalFlowReader
    from utils.training_tokenization import DatasetIntegrityError
    record = fixture_manifest(tmp_path)
    manifest, hdf5 = tmp_path / "manifest.jsonl", tmp_path / "0.h5"
    manifest.write_text(json.dumps(record))
    reader = OpticalFlowReader(tmp_path, manifest, max_handles=1)
    assert reader.read(0, 3)["flow_supervision_available"]
    reader.close()
    record["sha256"] = sha256_file(hdf5)
    reader.episodes[0][1]["sha256"] = record["sha256"]
    cache = PreparationAuditCache(snapshot(tmp_path, [hdf5]))
    reader.contract = {"audit_cache": cache}
    monkeypatch.setattr(reader, "_validate_structure", lambda *a: pytest.fail("full Flow file audit repeated"))
    for _ in range(2):
        assert reader.read(0, 3)["flow_supervision_available"]
        assert len(reader._handles) == 1
        reader.close()
    with h5py.File(hdf5, "a") as stream:
        stream["flow"][0, 0, 0, 0] = np.nan
    with pytest.raises(DatasetIntegrityError, match="changed since the saved audit"):
        reader.read(0, 3)
    assert not reader._handles


def test_cached_slot_checks_source_identity_without_reopening_annotation_payload(tmp_path, monkeypatch):
    from collections import OrderedDict
    from utils.slot_supervision import SlotSupervisionReader
    source = tmp_path / "annotations.jsonl"
    source.write_text('{"audited": true}')
    cache = PreparationAuditCache(snapshot(tmp_path, [source]))
    reader = SlotSupervisionReader.__new__(SlotSupervisionReader)
    reader.audit_cache, reader.source_root = cache, tmp_path
    reader.source_hashes = {str(source): sha256_file(source)}
    reader.mapping = {0: {"old_episode_index": 0}}
    reader.unannotated, reader.verified_episodes = set(), OrderedDict()
    monkeypatch.setattr("utils.slot_supervision._active_training_samples", lambda *a: source)
    monkeypatch.setattr(Path, "read_bytes", lambda *a: pytest.fail("source Slot labels audited again"))
    reader._verify_source_episode(0)
    reader._verify_source_episode(0)
    source.write_text('{"changed_annotation": true}')
    with pytest.raises(ValueError, match="changed since the saved audit"):
        reader._verify_source_episode(0)


def test_preparation_command_reuses_failed_snapshot_without_rerunning_audit(tmp_path, monkeypatch, capsys):
    import sys
    from scripts import prepare_three_stage_validation as preparation
    config = {"python": sys.executable, "output_root": str(tmp_path)}
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    (tmp_path / "audit_snapshot.json").write_text(json.dumps({"config": config, "status": "blocked", "blockers": ["Flow"]}))
    monkeypatch.setattr(preparation, "sources", lambda *a: pytest.fail("source audit reran"))
    monkeypatch.setattr(sys, "argv", ["prepare", "--config", str(config_path), "--phase", "sources"])
    preparation.main()
    result = json.loads(capsys.readouterr().out)
    assert result["reused"] and not result["audit_executed"] and result["status"] == "blocked"


def test_blocked_preparation_stops_before_payload_checks_or_gpu_start(tmp_path, monkeypatch):
    from test_three_stage_validation import configuration
    from utils.three_stage_preflight import validate_preparation
    config = configuration()
    config.pop("preparation_revision", None)
    config["output_root"] = str(tmp_path)
    (tmp_path / "preparation_blocked.json").write_text(json.dumps({"blockers": ["RH20T Flow timestamps"]}))
    monkeypatch.setattr("utils.three_stage_preflight.sha256_file", lambda *a: pytest.fail("blocked startup audited files"))
    with pytest.raises(ValueError, match="saved preparation audit is blocked"):
        validate_preparation(config)


def test_runtime_binding_preserves_failed_snapshot_and_rejects_audit_algorithm_changes(tmp_path, monkeypatch):
    from scripts.prepare_three_stage_validation import bind_runtime
    config = {"output_root": str(tmp_path)}
    original = {"utils/dataset_spec.py": "before", "utils/slot_supervision.py": "unchanged"}
    path = tmp_path / "audit_snapshot.json"
    path.write_text(json.dumps({"version": 1, "status": "blocked", "config": config,
        "files": {}, "runtime_implementation": original}))
    digest = sha256_file(path)
    current = dict(original, **{"utils/dataset_spec.py": "cached"})
    monkeypatch.setattr("utils.three_stage_preflight.implementation_identity", lambda: current)
    bind_runtime(config)
    binding = json.loads((tmp_path / "audit_runtime_binding.json").read_text())
    assert binding["audit_status_unchanged"] == "blocked"
    assert sha256_file(path) == digest == binding["audit_snapshot_sha256"]
    assert not binding["original_payloads_reaudited"]
    current["utils/slot_supervision.py"] = "different audit producer"
    with pytest.raises(ValueError, match="exceeds the reviewed cache adapter scope"):
        bind_runtime(config)


def test_preparation_reuses_resolved_revision_without_repeating_source_audit(tmp_path, monkeypatch, capsys):
    import sys
    from scripts import prepare_three_stage_validation as preparation
    config = {"python": sys.executable, "output_root": str(tmp_path), "preparation_revision": "flow_exclusion_108"}
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    (tmp_path / "audit_snapshot.json").write_text(json.dumps({"status": "blocked"}))
    revision = tmp_path / "flow_exclusion_108"
    revision.mkdir()
    snapshot_path = revision / "audit_snapshot.json"
    snapshot_path.write_text(json.dumps({"config": config, "status": "passed", "blockers": []}))
    monkeypatch.setattr(preparation, "sources", lambda *a: pytest.fail("source audit reran"))
    monkeypatch.setattr(sys, "argv", ["prepare", "--config", str(config_path), "--phase", "sources"])
    preparation.main()
    result = json.loads(capsys.readouterr().out)
    assert result["reused"] and not result["audit_executed"] and result["status"] == "passed"
    assert result["audit_snapshot"] == str(snapshot_path)
