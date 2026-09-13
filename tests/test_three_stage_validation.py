"""Frozen identities, command boundaries and source tensor loading contracts."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from scripts.run_three_stage_validation import training_command, validate_config, STAGES
from utils.frozen_stage_index import load_frozen_stage_index
from utils.stage05_sidecar import canonical_json_hash, sha256_file
from utils.three_stage_sources import compare_component


def configuration():
    return json.loads((Path(__file__).resolve().parents[1] / "configs/three_stage_validation_20260908.json").read_text())


def test_commands_keep_formal_budgets_out_of_validation_and_expert_source_independent():
    config = configuration()
    for stage in STAGES:
        for stop in (50, 100):
            _, command = training_command(config, stage, stop)
            value = lambda name: command[command.index("--" + name) + 1]
            assert value("max_train_steps") == "100"
            assert value("save_and_exit_after_updates") == str(stop)
            assert "--bounded_three_stage_validation" in command
            assert value("component_update_diagnostics") == "full"
            assert "--fast_resume_data_skip" in command
            assert value("optical_flow_loss_weight") == ("0.0" if stage == "stage1_ar" else "1.0")
            assert value("num_flow_queries") == "16" and value("num_difference_queries") == "32"
            if stage == "stage3_joint" and stop == 50:
                assert value("action_expert_name_or_path") == config["base_model"]
                assert "/stage2_aux/to100/" in value("vlm_name_or_path")
                assert value("action_expert_loss_weight") == "5.0"
        _, command = training_command(config, stage, None, formal=True)
        assert "--bounded_three_stage_validation" not in command
        assert command[command.index("--max_train_steps") + 1] == str(config["formal_command_templates_only"][stage]["steps"])
        assert not any("/validation/" in word for word in command)
    config["validation_updates_per_stage"] = 10000
    with pytest.raises(ValueError, match="fixed configuration"):
        validate_config(config)


def test_bounded_runtime_rejects_cli_override_of_prepared_learning_rate():
    from utils.cli_options import parse_train_options
    from utils.three_stage_preflight import validate_runtime_options
    config = configuration()
    _, command = training_command(config, "stage1_ar", 50)
    script = next(word for word in command if word.endswith("/train_vla.py"))
    options = parse_train_options(command[command.index(script) + 1:])
    validate_runtime_options(options, config)
    options.peak_learning_rate = .1
    with pytest.raises(ValueError, match="peak_learning_rate"):
        validate_runtime_options(options, config)


def test_authorized_attempt_only_changes_validation_wandb_identity(monkeypatch):
    config = configuration()
    monkeypatch.delenv("ZR0_VALIDATION_ATTEMPT", raising=False)
    originals = {(stage, stop): training_command(config, stage, stop)
        for stage in STAGES for stop in (50, 100)}
    formal = {stage: training_command(config, stage, None, formal=True) for stage in STAGES}
    monkeypatch.setenv("ZR0_VALIDATION_ATTEMPT", "r1")
    for (stage, stop), (old_run, old_command) in originals.items():
        run, command = training_command(config, stage, stop)
        assert run == old_run
        expected = list(old_command)
        for option in ("--wandb_run_name", "--wandb_run_id"):
            expected[expected.index(option) + 1] += "-r1"
        assert command == expected
        assert training_command(config, stage, None, formal=True) == formal[stage]
    monkeypatch.setenv("ZR0_VALIDATION_ATTEMPT", "../invalid")
    with pytest.raises(ValueError, match="validation attempt"):
        training_command(config, "stage1_ar", 50)


def test_frozen_index_rejects_changed_sidecar_and_unsorted_content(tmp_path):
    sidecar = tmp_path / "ar"
    sidecar.mkdir()
    (sidecar / "manifest.json").write_text("{}")
    index = tmp_path / "stage12_indices.npy"
    np.save(index, np.array([1, 3, 4], dtype=np.int64))
    report = {"version": 1, "dataset_root": str(tmp_path), "source_frames": 8,
        "sidecar_manifest_sha256": {"ar": sha256_file(sidecar / "manifest.json")},
        "statistics": {"stats_key": "full_droid", "sha256": "independent"},
        "stage12": {"file": index.name, "sha256": sha256_file(index), "count": 3}}
    path = tmp_path / "frozen.json"
    def save():
        report.pop("content_hash", None)
        report["content_hash"] = canonical_json_hash(report)
        path.write_text(json.dumps(report))
    save()
    result, identity = load_frozen_stage_index(path, dataset_root=tmp_path, sidecar_root=sidecar, phase="ar")
    assert result.tolist() == [1, 3, 4] and identity["phase"] == "stage12"
    np.save(index, np.array([4, 3, 1], dtype=np.int64))
    report["stage12"]["sha256"] = sha256_file(index)
    save()
    with pytest.raises(ValueError, match="invalid frozen"):
        load_frozen_stage_index(path, dataset_root=tmp_path, sidecar_root=sidecar, phase="ar")
    (sidecar / "manifest.json").write_text('{"changed":true}')
    with pytest.raises(ValueError, match="sidecar identity"):
        load_frozen_stage_index(path, dataset_root=tmp_path, sidecar_root=sidecar, phase="ar")


def test_component_verification_is_exact_after_source_dtype_conversion(tmp_path):
    module = torch.nn.Linear(3, 2).to(torch.bfloat16)
    save_file({key: value.float() for key, value in module.state_dict().items()}, str(tmp_path / "source.safetensors"))
    assert compare_component(module, tmp_path, weights="source.safetensors")["tensors_exact"] == 2
    with torch.no_grad():
        module.bias[0] += 1
    with pytest.raises(RuntimeError, match="differs from source"):
        compare_component(module, tmp_path, weights="source.safetensors")


def test_three_stage_h50_source_h10_checkpoint_save_load_and_transition(tmp_path, monkeypatch):
    import hashlib
    from model.reasoning_vla_model import ZR0Model
    from utils.action_expert_config import load_action_expert_config
    from utils.optical_flow_config import OpticalFlowConfig
    from utils.optical_flow_checkpoint import module_checksum
    from utils.stage05_checkpoint_contract import validate_action_expert_config_provenance
    from test_query_ar_joint_checkpoint import (
        _TinyBackbone, _TinyActionExpert, _production_stage05_manifest, QueryArWarmStartCheckpointTest,
    )

    monkeypatch.setattr("model.reasoning_vla_model.QwenVLBackbone", _TinyBackbone)
    monkeypatch.setattr("model.reasoning_vla_model.FlowmatchingActionHead", _TinyActionExpert)
    monkeypatch.setattr("model.flow_matching_action_head.FlowmatchingActionHead", _TinyActionExpert)
    base = tmp_path / "base"
    base.mkdir()
    payload = QueryArWarmStartCheckpointTest.action_config().to_dict()
    payload.update(action_horizon=50, action_dim=64, state_dim=64, max_seq_len=64)
    payload["diffusion_transformer_cfg"]["max_num_positional_embeddings"] = 64
    source_bytes = json.dumps(payload, indent=2).encode()
    (base / "action_expert_config.json").write_bytes(source_bytes)
    (base / "config.json").write_text(json.dumps({"text_config": {"hidden_size": 3}}))
    original_expert = _TinyActionExpert(None, True)
    save_file(original_expert.state_dict(), base / "action_expert.safetensors")
    runtime = load_action_expert_config(base / "action_expert_config.json", action_horizon_override=10)
    source = base
    previous = None
    for stage in STAGES:
        flow = OpticalFlowConfig(num_flow_queries=16, flow_head_hidden_dim=16, flow_head_num_layers=1,
            optical_flow_loss_weight=0.0 if stage == "stage1_ar" else 1.0,
            optical_flow_aux_type="none" if stage == "stage1_ar" else "dense_regression_v1")
        model = ZR0Model(str(source), str(base) if stage == "stage3_joint" else None, runtime.config,
            training_stage=stage, tune_vlm=stage != "stage2_aux", tune_action_expert=stage == "stage3_joint",
            use_difference_query=True, num_difference_queries=32, optical_flow_config=flow,
            action_expert_config_path=str(base / "action_expert_config.json"),
            init_from_checkpoint=str(source) if previous is not None else None)
        model.action_expert_config_source_bytes = source_bytes
        model.action_expert_config_source_sha256 = hashlib.sha256(source_bytes).hexdigest()
        model.action_expert_source_action_horizon = 50
        model.resolved_dataset_manifest = _production_stage05_manifest(model.loss_type,
            action_horizon=10, state_dim=64, action_dim=64)
        if previous is not None:
            assert module_checksum(model.backbone) == module_checksum(previous.backbone)
        if stage == "stage3_joint":
            assert module_checksum(model.action_expert) == module_checksum(original_expert)
            assert module_checksum(model.optical_flow_aux) == module_checksum(previous.optical_flow_aux)
        saved = tmp_path / stage
        model.save_pretrained(saved)
        metadata = json.loads((saved / "zr0_checkpoint_metadata.json").read_text())
        provenance = validate_action_expert_config_provenance(saved, metadata)
        assert provenance["source_action_horizon"] == 50
        assert provenance["runtime_action_horizon"] == 10
        assert (saved / "action_expert_source_config.json").read_bytes() == source_bytes
        assert load_action_expert_config(saved / "action_expert_config.json").config.action_horizon == 10
        restored = ZR0Model.from_pretrained(saved, tune_vlm=stage != "stage2_aux",
            tune_action_expert=stage == "stage3_joint", action_horizon=10)
        assert module_checksum(restored) == module_checksum(model)
        restored.action_expert_config_source_bytes = (saved / "action_expert_config.json").read_bytes()
        restored.action_expert_config_source_sha256 = hashlib.sha256(restored.action_expert_config_source_bytes).hexdigest()
        restored.resolved_dataset_manifest = model.resolved_dataset_manifest
        resaved = tmp_path / (stage + "_resaved")
        restored.save_pretrained(resaved)
        assert (resaved / "action_expert_source_config.json").read_bytes() == source_bytes
        assert validate_action_expert_config_provenance(resaved,
            json.loads((resaved / "zr0_checkpoint_metadata.json").read_text()))["source_action_horizon"] == 50
        from types import SimpleNamespace
        from utils.three_stage_sources import verify_checkpoint_serialization
        report = verify_checkpoint_serialization(model, SimpleNamespace(output_ckpt_dir=str(resaved),
            training_stage=stage, action_horizon=10, num_difference_queries=32,
            max_pad_state_and_action_length=64))
        assert report["status"] == "passed" and report["optimizer_updates"] == 0
        assert not list(resaved.glob(".checkpoint-contract-*"))
        source, previous = saved, model
    inference = ZR0Model.from_pretrained(source, for_action_inference=True,
        checkpoint_load_purpose="inference", action_horizon=10)
    assert module_checksum(inference.action_expert) == module_checksum(original_expert)
    (source / "action_expert_source_config.json").write_bytes(source_bytes + b" ")
    with pytest.raises(ValueError, match="original source config hash"):
        ZR0Model.from_pretrained(source, tune_vlm=True, tune_action_expert=True, action_horizon=10)


def test_resume_forward_diagnostic_restores_rng_and_training_mode():
    from types import SimpleNamespace
    from utils.validation_resume import diagnostic_forward, fingerprint
    from utils.training_checkpoint import capture_rng_state
    class RandomForward(torch.nn.Module):
        def forward(self, batch, progress, **kwargs):
            return {"loss": batch["x"].sum() + torch.rand(())}
    model = RandomForward().train()
    before = fingerprint(capture_rng_state())
    options = SimpleNamespace(vlm_loss_weight=1., action_expert_loss_weight=5.)
    first = diagnostic_forward(model, [{"x": torch.ones(2)}], options)
    second = diagnostic_forward(model, [{"x": torch.ones(2)}], options)
    assert torch.equal(first[0]["loss"], second[0]["loss"])
    assert model.training and before == fingerprint(capture_rng_state())


def test_frozen_dataset_stage1_omits_slot_and_stage3_retains_missing_fm(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import pyarrow as pa
    import pyarrow.parquet as pq
    from test_stage05_mixed_pretraining import _toy_molmo
    from utils.stage05_sidecar import build_stage05_sidecar
    from utils.stage05_dataset import Stage05MixedPretrainingDataset
    root = tmp_path / "source"
    _toy_molmo(root)
    data_path = root / "data/chunk-000/file-000.parquet"
    rows = pq.read_table(data_path).to_pylist()
    for i, row in enumerate(rows):
        row["state"] = [value + i * .01 for value in row["state"]]
        row["actions"] = [value + i * .01 for value in row["actions"]]
    pq.write_table(pa.Table.from_pylist(rows), data_path)
    monkeypatch.setattr("utils.stage05_dataset.resolve_stage05_vision_contract", lambda *a, **k: {})
    monkeypatch.setattr("utils.stage05_dataset.tokenize_future_difference_message", lambda *a, **k:
        {"input_ids": torch.tensor([1, 2]), "labels": torch.tensor([-100, 2])})
    ar, joint = tmp_path / "ar", tmp_path / "joint"
    for directory in (ar, joint):
        build_stage05_sidecar(root=root, output=directory, dataset_id="own_stats", kind="molmo",
            embedded_images=True, horizon=10, build_joint=directory == joint)
    index = tmp_path / "indices.npy"
    np.save(index, np.array([0, 1, 2], dtype=np.int64))
    item = {"file": index.name, "sha256": sha256_file(index), "count": 3}
    report = {"version": 1, "dataset_root": str(root), "source_frames": 3,
        "sidecar_manifest_sha256": {"ar": sha256_file(ar / "manifest.json"), "joint": sha256_file(joint / "manifest.json")},
        "stage12": item, "stage3": item, "statistics": {"stats_key": "own_stats"}}
    report["content_hash"] = canonical_json_hash(report)
    frozen = tmp_path / "frozen.json"
    frozen.write_text(json.dumps(report))
    entry = dict(dataset_entry="toy", dataset_path=str(root), dataset_type="vla", dataset_adapter="stage05_mixed_pretraining",
        stage05_kind="molmo", embedded_images=True, stats_key="own_stats", action_representation="canonical_eef_delta_7d",
        camera_keys=["first_view", "wrist_image"], ar_sidecar_path=str(ar), joint_sidecar_path=str(joint),
        frozen_stage_index=str(frozen), state_key="state", action_key="actions", task_description_key="task",
        target_text_field="train_data", action_normalization="quantile", original_action_dim=7, original_state_dim=7)
    reads = []
    original_read = pq.read_table
    def read(path, *args, **kwargs):
        if "/data/" in str(path):
            reads.extend(kwargs.get("columns", []))
        return original_read(path, *args, **kwargs)
    monkeypatch.setattr(pq, "read_table", read)
    stage1 = Stage05MixedPretrainingDataset(entry=entry, processor=SimpleNamespace(), loss_type="vlm", max_length=1024, action_horizon=10)
    sample = stage1[0]
    assert "slot_data" not in reads and "actions" not in reads and "slot_data" not in sample
    assert stage1.flow_reader is None and sample["ar_eligible"]
    stage3 = Stage05MixedPretrainingDataset(entry=entry, processor=SimpleNamespace(), loss_type="vlm_and_action", max_length=1024, action_horizon=10)
    assert len(stage3) == 3
    assert stage3[0]["fm_eligible"]
    missing = stage3[2]
    assert missing["ar_eligible"] and not missing["fm_eligible"]
    assert not missing["action_mask"].any() and not missing["state_mask"].any()
    assert torch.isfinite(missing["action"]).all() and not missing["strict_joint_fm"]
    from utils.preparation_audit_cache import PreparationAuditCache
    from test_preparation_audit_cache import snapshot
    cached_path = snapshot(tmp_path, [path for path in tmp_path.rglob("*") if path.is_file()])
    cache = PreparationAuditCache(cached_path)
    monkeypatch.setattr("utils.preparation_audit_cache.load_preparation_audit_cache", lambda *a: cache)
    def repeated_audit(*args, **kwargs):
        pytest.fail("cached production dataset construction repeated its sidecar audit")
    monkeypatch.setattr("utils.stage05_sidecar.load_stage05_sidecar", repeated_audit)
    monkeypatch.setattr("utils.stage05_dataset.load_stage05_sidecar", repeated_audit)
    entry["preparation_audit_cache"] = str(cached_path)
    for phase in ("vlm", "vlm_and_action"):
        cached = Stage05MixedPretrainingDataset(entry=entry, processor=SimpleNamespace(), loss_type=phase,
            max_length=1024, action_horizon=10)
        assert len(cached) == 3 and cached[0]["ar_eligible"]
        assert cached.spec.auxiliary_contract["frozen_stage_index"]["preparation_audit_sha256"] == cache.sha256


def test_flow_remapping_preserves_original_hdf5_and_checks_full_source_timestamps(tmp_path):
    import h5py
    import pyarrow as pa
    import pyarrow.parquet as pq
    from test_optical_flow_aux import fixture_manifest
    from utils.aux_data_contract import flow_contract
    from utils.optical_flow_reader import OpticalFlowReader
    from utils.training_tokenization import DatasetIntegrityError
    root = tmp_path / "full"
    (root / "meta").mkdir(parents=True)
    (root / "meta/info.json").write_text(json.dumps({"fps": 10}))
    mapping = root / "meta/stage05_episode_mapping.jsonl"
    mapping.write_text(json.dumps({"new_episode_index": 67, "old_episode_index": 0, "source_data_uri": "rows.parquet"}))
    rows = [{"episode_index": 67, "frame_index": frame, "timestamp": frame / 10} for frame in (3, 13)]
    pq.write_table(pa.Table.from_pylist(rows), root / "rows.parquet")
    record = fixture_manifest(tmp_path)
    record.pop("schema_contract")
    provenance = {key: "fixture" for key in ("artifact_identity", "generation_identity", "label_identity",
        "source_fingerprint", "checkpoint_sha256", "model_revision")}
    record.update(provenance, schema_version="stage06_flow_manifest_v2", dataset_id="droid")
    with h5py.File(tmp_path / "0.h5", "a") as handle:
        handle.attrs.update(provenance, schema_version="stage06_flow_v2", dataset_id="droid", output_height=224, output_width=224)
        handle["valid_fraction"] = handle["valid_mask"][:].reshape(2, -1).mean(axis=1).astype("float32")
        handle["source_timestamp_s"] = [.3, 1.3]
        handle["target_timestamp_s"] = [1.3, 1.3]
        handle["actual_delta_s"] = [1., 0.]
    original_hash = sha256_file(tmp_path / "0.h5")
    record["sha256"] = original_hash
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps(record))
    remap = tmp_path / "remap.json"
    remap.write_text(json.dumps({"version": 1, "dataset_root": str(root),
        "source_manifest_sha256": sha256_file(manifest), "target_mapping_sha256": sha256_file(mapping),
        "matches": [{"flow_episode": 0, "full_episode": 67, "source_episode": 0}]}))
    contract = flow_contract(dict(dataset_path=str(root), optical_flow_manifest=str(manifest),
        flow_episode_map=str(remap), aux_dataset_identity="droid", flow_delta_frames=10,
        camera_keys=["observation.images.image"]))
    reader = OpticalFlowReader(tmp_path, manifest, contract=contract)
    assert not reader._handles
    assert reader.read(67, 3)["flow_supervision_available"]
    assert not reader.read(0, 3)["flow_supervision_available"]
    reader.close()
    assert sha256_file(tmp_path / "0.h5") == original_hash
    rows[0]["timestamp"] += .1
    pq.write_table(pa.Table.from_pylist(rows), root / "rows.parquet")
    with pytest.raises(DatasetIntegrityError, match="timestamp/source/FPS"):
        reader.read(67, 3)
    assert not reader._handles


def test_flow_excluded_frames_are_masked_without_mutating_the_original_artifact(tmp_path):
    from test_optical_flow_aux import fixture_manifest
    from utils.optical_flow_reader import OpticalFlowReader
    import hashlib
    import h5py
    record = fixture_manifest(tmp_path, frames=(3, 13, 23))
    with h5py.File(tmp_path / "0.h5", "a") as stream:
        stream["target_frame_index"][:] = [13, 23, 23]
        stream["actual_delta_frames"][:] = [10, 10, 0]
    record["sha256"] = hashlib.sha256((tmp_path / "0.h5").read_bytes()).hexdigest()
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps(record))
    original = hashlib.sha256((tmp_path / "0.h5").read_bytes()).hexdigest()
    reader = OpticalFlowReader(tmp_path, manifest)
    reader._handle(0)
    reader.contract = {"excluded_frames": {"0": [3]}}
    assert not reader.read(0, 3)["flow_supervision_available"]
    assert reader.read(0, 3)["flow_exclusion_reason"] == 5
    assert reader.read(0, 13)["flow_supervision_available"]
    assert reader.eligible_frames(.5) == {(0, 13)}
    reader.close()
    assert hashlib.sha256((tmp_path / "0.h5").read_bytes()).hexdigest() == original


def test_flow_exclusion_preserves_source_alignment_and_unchanged_interval_tolerance(tmp_path):
    import h5py
    import pyarrow as pa
    import pyarrow.parquet as pq
    from test_optical_flow_aux import fixture_manifest
    from utils.aux_data_contract import flow_contract
    from utils.optical_flow_reader import OpticalFlowReader
    from utils.training_tokenization import DatasetIntegrityError
    (tmp_path / "meta").mkdir()
    (tmp_path / "meta/info.json").write_text('{"fps": 10}')
    (tmp_path / "meta/stage05_episode_mapping.jsonl").write_text(json.dumps({
        "new_episode_index": 0, "old_episode_index": 0, "source_data_uri": "rows.parquet"}))
    rows = [{"episode_index": 0, "frame_index": frame, "timestamp": time}
            for frame, time in zip((3, 13, 23), (.3, 1.30004, 2.30004))]
    pq.write_table(pa.Table.from_pylist(rows), tmp_path / "rows.parquet")
    record = fixture_manifest(tmp_path, frames=(3, 13, 23))
    record.pop("schema_contract")
    provenance = {key: "fixture" for key in ("artifact_identity", "generation_identity", "label_identity",
        "source_fingerprint", "checkpoint_sha256", "model_revision")}
    with h5py.File(tmp_path / "0.h5", "a") as stream:
        stream.attrs.update(provenance, schema_version="stage06_flow_v2", dataset_id="rh20t", output_height=224, output_width=224)
        stream["valid_fraction"] = stream["valid_mask"][:].reshape(3, -1).mean(axis=1).astype("float32")
        stream["target_frame_index"][:] = [13, 23, 23]
        stream["actual_delta_frames"][:] = [10, 10, 0]
        stream["source_timestamp_s"] = [.3, 1.30004, 2.30004]
        stream["target_timestamp_s"] = [1.30004, 2.30004, 2.30004]
        stream["actual_delta_s"] = np.array([1.00004, 1., 0.], dtype=np.float32)
    record.update(provenance, schema_version="stage06_flow_manifest_v2", dataset_id="rh20t", sha256=sha256_file(tmp_path / "0.h5"))
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps(record))
    entry = dict(dataset_path=str(tmp_path), optical_flow_manifest=str(manifest), aux_dataset_identity="rh20t",
                 camera_keys=["observation.images.image"], flow_delta_frames=10)
    reader = OpticalFlowReader(tmp_path, manifest, contract=flow_contract(entry))
    with pytest.raises(DatasetIntegrityError, match="timestamp/source/FPS"):
        reader.read(0, 13)
    entry["flow_excluded_frames"] = {"0": [3]}
    reader = OpticalFlowReader(tmp_path, manifest, contract=flow_contract(entry))
    assert not reader.read(0, 3)["flow_supervision_available"]
    assert reader.read(0, 13)["flow_supervision_available"]
    reader.close()
    rows[0]["timestamp"] += .1
    pq.write_table(pa.Table.from_pylist(rows), tmp_path / "rows.parquet")
    with pytest.raises(DatasetIntegrityError, match="timestamp/source/FPS"):
        reader.read(0, 13)


def test_flow_resolution_commands_use_revision_and_keep_checkpoint_destinations():
    from utils.three_stage_preflight import preparation_directory
    config = configuration()
    directory = preparation_directory(config)
    assert directory.name == "flow_exclusion_108"
    run, command = training_command(config, "stage1_ar", 50)
    assert str(run).endswith("/validation/stage1_ar/to50")
    for option, name in (("preparation_audit_cache", "audit_snapshot.json"),
                         ("aux_dataset_config", "cached_aux_dataset_routes.json"),
                         ("three_stage_preparation_config", "requested_config.json")):
        assert command[command.index("--" + option) + 1] == str(directory / name)


def test_resume_evidence_checks_exact_state_and_next_window_without_optimizer_update(tmp_path):
    from types import SimpleNamespace
    from utils.training_checkpoint import capture_rng_state
    from utils.validation_resume import save_next_window_evidence, verify_next_window_evidence
    class DiagnosticModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(2))
        def forward(self, batch, progress, **kwargs):
            return {"loss": (self.weight * batch["input"]).sum() + torch.rand(())}
    model = DiagnosticModel().train()
    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1)
    model.optimizer = SimpleNamespace(optimizer=optimizer, single_partition_of_fp32_groups=list(model.parameters()))
    model.training_data_cursor = {"epoch": 0, "batch_idx": 100}
    model.training_sampler_contract = {"seed": 42}
    model.validation_seen_tracker = SimpleNamespace(**{key: [3] for key in
        ("entries", "seen", "unique", "duplicates", "ar_eligible", "fm_eligible", "aux_counts", "bitsets")})
    accelerator = SimpleNamespace(unwrap_model=lambda value: value, process_index=0)
    options = SimpleNamespace(vlm_name_or_path=str(tmp_path), vlm_loss_weight=1., action_expert_loss_weight=5.)
    torch.save({"rng": capture_rng_state()}, tmp_path / "training_runtime_rank0.pt")
    window = [(100, {"input": torch.tensor([3., 4.]), "sample_global_index": torch.tensor([781])})]
    save_next_window_evidence(model, scheduler, accelerator, window, options, tmp_path)
    verified = verify_next_window_evidence(model, scheduler, accelerator, window, options)
    assert verified["state_exact"] and verified["optimizer_updates"] == 0 and not optimizer.state
    assert scheduler.last_epoch == 0 and model.training_data_cursor["batch_idx"] == 100
    window[0][1]["sample_global_index"] += 1
    with pytest.raises(RuntimeError, match="next production"):
        verify_next_window_evidence(model, scheduler, accelerator, window, options)
    model.validation_seen_tracker.seen = [4]
    with pytest.raises(RuntimeError, match="exact resume state"):
        verify_next_window_evidence(model, scheduler, accelerator, window, options)


def test_exact_32_16_partition_has_independent_head_gradient_paths():
    from model.structured_slot_head import build_slot_head
    from model.optical_flow_aux_head import build_optical_flow_head
    from utils.slot_config import SlotConfig, resolve_query_layout
    from utils.optical_flow_config import OpticalFlowConfig
    layout = resolve_query_layout(32, 16, slot_enabled=True, query_enabled=True)
    assert layout["num_slot_queries"] == 16
    slot = build_slot_head(32, 16, SlotConfig(slot_aux_type="structured_slots_v1", slot_loss_weight=1.))
    flow = build_optical_flow_head(32, OpticalFlowConfig(optical_flow_aux_type="dense_regression_v1",
        num_flow_queries=16, optical_flow_loss_weight=1., flow_head_hidden_dim=16, flow_head_num_layers=1))
    queries = torch.randn(1, 32, 32, requires_grad=True)
    sum(value.sum() for value in slot(queries).values()).backward()
    assert queries.grad[:, :16].abs().sum() > 0 and queries.grad[:, 16:].count_nonzero() == 0
    queries.grad = None
    flow(queries).sum().backward()
    assert queries.grad[:, 16:].abs().sum() > 0 and queries.grad[:, :16].count_nonzero() == 0


def test_unannotated_slot_is_distinct_from_corrupt_existing_source(tmp_path):
    from collections import OrderedDict
    import pyarrow as pa
    import pyarrow.parquet as pq
    from utils.slot_supervision import SlotSupervisionReader
    reader = SlotSupervisionReader.__new__(SlotSupervisionReader)
    reader.audit_cache = None
    reader.root = reader.source_root = tmp_path
    reader.mapping = {0: {"old_episode_index": 12, "source_data_uri": "rows.parquet"}}
    reader.lazy_sources = True
    reader.unannotated, reader.anchors = {0}, {}
    reader.verified_episodes, reader.cache = OrderedDict(), OrderedDict()
    row = {"episode_index": 0, "frame_index": 0, "slot_data": None}
    pq.write_table(pa.Table.from_pylist([row]), tmp_path / "rows.parquet")
    assert not any(value.any() for key, value in reader.read(0, 0).items() if key.endswith("mask"))
    row["slot_data"] = '{}'
    pq.write_table(pa.Table.from_pylist([row]), tmp_path / "rows.parquet")
    reader.cache.clear()
    with pytest.raises(ValueError, match="unannotated Slot labels changed"):
        reader.read(0, 0)
    (tmp_path / "episode_000012").mkdir()
    with pytest.raises(ValueError, match="previously unannotated Slot source changed"):
        reader.read(0, 0)
    reader.unannotated.clear()
    with pytest.raises(ValueError, match="missing Slot source episode"):
        reader.read(0, 0)
