"""Regression probes for the third independent auxiliary-training review."""

import json
from types import SimpleNamespace
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from test_optical_flow_aux import fixture_manifest
from utils.aux_data_contract import digest_file
from utils.optical_flow_reader import OpticalFlowReader
from utils.slot_labels import normalize_slot_labels
from utils.slot_config import SlotConfig
from utils.slot_supervision import SlotSupervisedDataset


def real_processor():
    from transformers import AutoProcessor
    return AutoProcessor.from_pretrained("/opt/data/private/lq/models/Qwen3-VL-2B-Instruct", local_files_only=True)


def stage05_flow_fixture(tmp_path):
    from test_stage05_mixed_pretraining import _toy_molmo
    from utils.stage05_sidecar import build_stage05_sidecar
    from utils.aux_data_contract import digest_file
    import pyarrow as pa
    import pyarrow.parquet as pq
    root = tmp_path / "source"
    _toy_molmo(root)
    path = root / "data/chunk-000/file-000.parquet"
    rows = pq.read_table(path).to_pylist()
    for i, row in enumerate(rows):
        row["timestamp"] = i / 10
    pq.write_table(pa.Table.from_pylist(rows), path)
    (root / "meta/stage05_episode_mapping.jsonl").write_text(json.dumps({"new_episode_index": 0,
        "old_episode_index": 19, "source_data_uri": "data/chunk-000/file-000.parquet"}))
    sidecar = tmp_path / "sidecar"
    build_stage05_sidecar(root=root, output=sidecar, dataset_id="toy", kind="molmo", embedded_images=True, build_joint=False)
    flow = tmp_path / "flow"
    flow.mkdir()
    record = fixture_manifest(flow, frames=(0, 1, 2))
    record.pop("schema_contract")
    record.update(source_episode_index=19, camera_key="first_view", dataset_id="toy", schema_version="stage06_flow_manifest_v2")
    with h5py.File(flow / "0.h5", "a") as h:
        h.attrs.update(source_episode_index=19, nominal_delta_frames=2, camera_key="first_view", dataset_id="toy",
                       schema_version="stage06_flow_v2", output_height=224, output_width=224)
        h["valid_fraction"] = h["valid_mask"][:].reshape(3, -1).mean(axis=1).astype(np.float32)
        for key in ("artifact_identity", "generation_identity", "label_identity", "source_fingerprint", "checkpoint_sha256", "model_revision"):
            record[key] = "fixture-v1"
            h.attrs[key] = record[key]
        h["source_timestamp_s"] = np.array([0., .1, .2])
        h["target_timestamp_s"] = np.array([.2, .2, .2])
        h["actual_delta_s"] = np.array([.2, .1, 0.])
    record["sha256"] = digest_file(flow / "0.h5")
    (flow / "manifest.jsonl").write_text(json.dumps(record))
    return dict(dataset_path=str(root), dataset_entry="toy", stats_key="toy", ar_sidecar_path=str(sidecar),
        dataset_adapter="stage05_mixed_pretraining", dataset_type="vla", stage05_kind="molmo", embedded_images=True,
        camera_keys=["first_view", "wrist_image"], sample_ratio=1., use_quantile=True, optical_flow_data_root=str(flow),
        optical_flow_manifest="manifest.jsonl", flow_delta_frames=2, aux_dataset_identity="toy", flow_enabled=True, slot_enabled=True)


def test_production_stage05_flow_reaches_collator_loss_and_query(tmp_path):
    from test_dataset_adapters_v3 import FakeProcessor
    from test_optical_flow_aux import config
    from utils.stage05_dataset import Stage05MixedPretrainingDataset
    from utils.load_training_dataset import custom_collate_fn
    from utils.optical_flow_loss import optical_flow_loss
    from model.optical_flow_aux_head import build_optical_flow_head
    ds = Stage05MixedPretrainingDataset(entry=stage05_flow_fixture(tmp_path), processor=real_processor(), loss_type="aux", max_length=1024)
    batch = custom_collate_fn([ds[0], ds[1], ds[2]])
    assert batch["flow_nominal_delta_frames"].tolist() == [2, 2, 2]
    query = torch.randn(3, 2, 32, requires_grad=True)
    head = build_optical_flow_head(32, config())
    outputs = optical_flow_loss(head(query), batch, config())
    assert outputs["optical_flow_loss_count"] == 1
    outputs["optical_flow_loss"].backward()
    assert query.grad[0].abs().sum() > 0 and query.grad[1:].abs().sum() == 0
    assert any(p.grad.abs().sum() > 0 for p in head.parameters())
    ds.flow_reader.close()


def test_flow_off_ignores_invalid_registry_paths(tmp_path, monkeypatch):
    from test_dataset_adapters_v3 import FakeProcessor
    from utils.stage05_dataset import Stage05MixedPretrainingDataset
    entry = stage05_flow_fixture(tmp_path)
    entry.update(flow_enabled=False, optical_flow_manifest="/does/not/exist")
    def forbidden(*args, **kwargs):
        pytest.fail("Flow-off constructed a reader")
    monkeypatch.setattr("utils.optical_flow_reader.OpticalFlowReader", forbidden)
    for objective in ("vlm", "aux"):
        ds = Stage05MixedPretrainingDataset(entry=entry, processor=real_processor(), loss_type=objective, max_length=1024)
        assert not any(key.startswith("flow_") for key in ds[0])


@pytest.mark.parametrize("mutation", ["dataset", "fps", "nominal", "timestamp", "mapping", "manifest", "source_mapping"])
def test_flow_provenance_and_resume_reject_changes(tmp_path, mutation):
    from test_dataset_adapters_v3 import FakeProcessor
    from utils.aux_data_contract import digest_file
    from utils.stage05_dataset import Stage05MixedPretrainingDataset
    from utils.dataset_manifest import build_resolved_dataset_manifest, write_resolved_dataset_manifest, validate_resume_manifest
    entry = stage05_flow_fixture(tmp_path)
    processor = real_processor()
    original = Stage05MixedPretrainingDataset(entry=entry, processor=processor, loss_type="aux", max_length=1024)
    before = build_resolved_dataset_manifest([original.spec], "aux")
    write_resolved_dataset_manifest(tmp_path / "checkpoint", before)
    validate_resume_manifest(before, tmp_path / "checkpoint")
    manifest = Path(entry["optical_flow_data_root"]) / "manifest.jsonl"
    record = json.loads(manifest.read_text())
    if mutation == "source_mapping":
        mapping_path = Path(entry["dataset_path"]) / "meta/stage05_episode_mapping.jsonl"
        mapping = json.loads(mapping_path.read_text())
        mapping["old_episode_index"] = 20
        mapping_path.write_text(json.dumps(mapping))
    elif mutation in {"mapping", "manifest"}:
        record["source_episode_index"] = 20 if mutation == "mapping" else 19
        record["label_identity"] = "changed"
        manifest.write_text(json.dumps(record))
    else:
        path = Path(entry["optical_flow_data_root"]) / "0.h5"
        with h5py.File(path, "a") as h:
            if mutation == "timestamp":
                h["source_timestamp_s"][0] = 777
            else:
                key = {"dataset": "dataset_id", "fps": "fps", "nominal": "nominal_delta_frames"}[mutation]
                h.attrs[key] = "wrong" if mutation == "dataset" else 777
        record["sha256"] = digest_file(path)
        manifest.write_text(json.dumps(record))
    if mutation == "source_mapping":
        from dataclasses import replace
        from utils.aux_data_contract import flow_contract, public_flow_contract
        with pytest.raises(ValueError, match="sidecar source is stale"):
            Stage05MixedPretrainingDataset(entry=entry, processor=processor, loss_type="aux", max_length=1024)
        changed_contract = flow_contract(entry)
        changed_spec = replace(original.spec, auxiliary_contract={"flow": public_flow_contract(changed_contract)})
        with pytest.raises(ValueError, match="manifest mismatch"):
            validate_resume_manifest(build_resolved_dataset_manifest([changed_spec], "aux"), tmp_path / "checkpoint")
        with pytest.raises(ValueError, match="mapping"):
            OpticalFlowReader(entry["optical_flow_data_root"], manifest, delta_frames=2, contract=changed_contract)
    elif mutation == "mapping":
        with pytest.raises(ValueError, match="mapping"):
            Stage05MixedPretrainingDataset(entry=entry, processor=processor, loss_type="aux", max_length=1024)
    else:
        changed = Stage05MixedPretrainingDataset(entry=entry, processor=processor, loss_type="aux", max_length=1024)
        with pytest.raises(ValueError, match="manifest mismatch"):
            validate_resume_manifest(build_resolved_dataset_manifest([changed.spec], "aux"), tmp_path / "checkpoint")
        with pytest.raises(ValueError):
            changed[0]
        assert not changed.flow_reader._handles and not changed.flow_reader.rows


@pytest.mark.parametrize("frames,frame", [((3, 13), 3), ((1, 0, 10), 0)])
def test_cold_sparse_and_unordered_flow(tmp_path, frames, frame):
    entry = fixture_manifest(tmp_path, frames=frames)
    path = tmp_path / "manifest.jsonl"
    path.write_text(json.dumps(entry))
    reader = OpticalFlowReader(tmp_path, path)
    for _ in range(2):
        assert reader.read(0, frame)["flow_supervision_available"]
    assert not reader.read(0, 99)["flow_supervision_available"]
    reader.close()


def test_failed_lazy_validation_can_retry_without_cache_poison(tmp_path):
    entry = fixture_manifest(tmp_path, frames=(3, 13))
    path = tmp_path / "manifest.jsonl"
    path.write_text(json.dumps(entry))
    reader = OpticalFlowReader(tmp_path, path)
    with h5py.File(tmp_path / "0.h5", "a") as handle:
        handle["frame_index"][1] = 3
    entry["sha256"] = digest_file(tmp_path / "0.h5")
    reader.episodes[0][1]["sha256"] = entry["sha256"]
    with pytest.raises(ValueError, match="duplicate"):
        reader.read(0, 3)
    assert not reader.rows and not reader._handles
    with h5py.File(tmp_path / "0.h5", "a") as handle:
        handle["frame_index"][1] = 13
    entry["sha256"] = digest_file(tmp_path / "0.h5")
    reader.episodes[0][1]["sha256"] = entry["sha256"]
    assert reader.read(0, 3)["flow_supervision_available"]
    reader.close()


def test_shared_classes_and_dataset_specific_q5_scales():
    import copy
    from test_slot_integration import fixture_stats
    from model.structured_slot_head import build_slot_head
    from utils.slot_routing import combined_slot_stats
    from utils.slot_loss import slot_loss
    from utils.slot_labels import empty_slot_labels
    a, b = fixture_stats(), fixture_stats()
    b["q5_mean"] = [[.1, .2, .3], [0., 0., 0.], [-.1, -.2, -.3]]
    b["q5_std"] = [[.01] * 3] * 3
    stats = combined_slot_stats({"b": b, "a": a})
    assert stats["dataset_order"] == ["a", "b"]
    assert stats["classes"]["Q2"]["counts"] == [20] * 7
    config = SlotConfig(slot_aux_type="structured_slots_v1", slot_loss_weight=1., slot_q5_consistency_weight=0.)
    predictions = build_slot_head(32, 7, config)(torch.randn(2, 15, 32))
    predictions["Q5"] = torch.zeros(2, 3, 3)
    labels = {key: torch.stack([value, value]) for key, value in empty_slot_labels().items()}
    labels.update(slot_Q5=torch.tensor([a["q5_mean"], b["q5_mean"]]),
                  slot_Q5_mask=torch.ones(2, 3, dtype=torch.bool), slot_dataset_index=torch.tensor([0, 1]))
    result = slot_loss(predictions, labels, config, stats)
    assert result["slot_Q5_count"] == 2 and result["slot_Q5_loss_sum"] == 0
    labels["slot_dataset_index"] = torch.tensor([1, 0])
    assert slot_loss(predictions, labels, config, stats)["slot_Q5_loss_sum"] > 0


@pytest.mark.parametrize("raw", [
    {"query_5": {"target_displacement_m": [float("nan"), 0, 0], "valid_mask": [1, 0, 0]}},
    {"query_5": {"target_displacement_m": [float("inf"), 0, 0], "valid_mask": [1, 0, 0]}},
    {"query_3": {"target_bbox_t": [.8, 0, .2, 1], "valid_mask": [1, 0]}},
    {"query_3": {"target_bbox_t": [0, 0, 1], "valid_mask": [1, 0]}},
    {"query_3": {"target_bbox_t": [0, 0, 1, 1]}},
    {"query_9": {"obstacles": [{"bbox": [1, 0, 0, 1]}], "valid_mask": [1, 0, 0]}},
])
def test_corrupt_positive_slot_is_error(raw):
    with pytest.raises(ValueError, match=r"dataset=probe episode=7 frame=9.*query_"):
        normalize_slot_labels(raw, is_anchor=True, identity="dataset=probe episode=7 frame=9")


@pytest.mark.parametrize("raw", [
    {"query_6": {"affordance_contact_points": 4, "valid_mask_t": [1, 0]}},
    {"query_6": {"affordance_contact_points": [[0, 0]], "valid_mask_t": [1, 1]}},
    {"query_9": {"obstacles": [4], "valid_mask": [1, 0, 0]}},
    {"query_2": {"phase_t_id": 4, "valid_mask": 1}},
    {"query_3": False},
])
def test_malformed_slot_containers_are_not_missing(raw):
    with pytest.raises(ValueError, match="dataset=probe.*query_"):
        normalize_slot_labels(raw, is_anchor=True, identity="dataset=probe episode=7 frame=9")


@pytest.mark.parametrize("field,value", [("fps", None), ("fps", 777.), ("nominal_delta_frames", None)])
def test_required_flow_time_metadata(tmp_path, field, value):
    entry = fixture_manifest(tmp_path)
    path = tmp_path / "manifest.jsonl"
    path.write_text(json.dumps(entry))
    with h5py.File(tmp_path / "0.h5", "a") as handle:
        if value is None:
            del handle.attrs[field]
        else:
            handle.attrs[field] = value
    entry["sha256"] = digest_file(tmp_path / "0.h5")
    path.write_text(json.dumps(entry))
    with pytest.raises(ValueError, match=field):
        reader = OpticalFlowReader(tmp_path, path)
        reader.read(0, 3)


def test_wrong_slot_route_is_error():
    class Dataset:
        spec = SimpleNamespace(dataset_path="/household", dataset_entry="household")
        requirements, loss_type = None, "vlm_and_action"
        def __len__(self):
            return 1
    reader = SimpleNamespace(root=Path("/tabletop"), anchors={})
    with pytest.raises(ValueError, match="Slot.*route"):
        SlotSupervisedDataset(Dataset(), reader, SlotConfig())


def test_wrapper_preserves_source_accounting():
    class Dataset:
        spec = SimpleNamespace(dataset_path="/tabletop", dataset_entry="tabletop")
        manifest = {"counts": {"source_frames": 2}}
        requirements, loss_type = None, "vlm_and_action"
        steps, subset_indices = [(0, 0), (0, 1)], [0, 1]
        def __len__(self):
            return 2
    reader = SimpleNamespace(root=Path("/tabletop"), anchors={})
    wrapped = SlotSupervisedDataset(Dataset(), reader, SlotConfig())
    assert wrapped.manifest == Dataset.manifest
    assert wrapped.unfiltered_length == 2


def test_sparse_slot_filter_uses_audited_global_index():
    class Dataset:
        spec = SimpleNamespace(dataset_path="/tabletop", dataset_entry="tabletop")
        requirements, loss_type = None, "aux"
        indices = np.array([0, 1])
        def __len__(self):
            return 2
        def _episode_for_global(self, index):
            return {"episode_index": 0}, index
    reader = SimpleNamespace(root=Path("/tabletop"), anchors={(0, 1): {"global_index": 0, "valid": True}})
    wrapped = SlotSupervisedDataset(Dataset(), reader, SlotConfig())
    assert wrapped.indices == [0]


def test_joint_constructor_does_not_walk_all_frames():
    class Dataset:
        spec = SimpleNamespace(dataset_path="/tabletop", dataset_entry="tabletop")
        requirements, loss_type = None, "vlm_and_action"
        def __len__(self):
            return 10_000_000
        def _episode_for_global(self, _):
            pytest.fail("Joint construction walked source frames")
        def sampling_group_ranges(self):
            return [(0, 5_000_000), (5_000_000, 10_000_000)]
    base = Dataset()
    reader = SimpleNamespace(root=Path("/tabletop"), anchors={})
    wrapped = SlotSupervisedDataset(base, reader, SlotConfig())
    assert isinstance(wrapped.indices, range)
    assert wrapped.sampling_group_ranges() == base.sampling_group_ranges()


def test_legacy_ar_samples_and_sampler_equivalent(tmp_path):
    import subprocess
    import types
    from test_dataset_adapters_v3 import FakeProcessor
    from utils.stage05_dataset import Stage05MixedPretrainingDataset
    from utils.load_training_dataset import EpochGroupedSampler
    historical = types.ModuleType("historical_stage05")
    source = subprocess.check_output(["git", "show", "00d611b:utils/stage05_dataset.py"], text=True)
    exec(compile(source, "historical_stage05", "exec"), historical.__dict__)
    entry = stage05_flow_fixture(tmp_path)
    entry["flow_enabled"] = False
    arguments = dict(entry=entry, processor=real_processor(), loss_type="vlm", max_length=1024)
    old = historical.Stage05MixedPretrainingDataset(**arguments)
    new = Stage05MixedPretrainingDataset(**arguments)
    np.testing.assert_array_equal(old.indices, new.indices)
    assert old.manifest == new.manifest
    for index in range(len(old)):
        left, right = old[index], new[index]
        assert left.keys() == right.keys()
        for key in left:
            if isinstance(left[key], torch.Tensor):
                torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)
            else:
                assert left[key] == right[key]
    for seed in (0, 42):
        a = EpochGroupedSampler(torch.utils.data.ConcatDataset([old]), seed=seed)
        b = EpochGroupedSampler(torch.utils.data.ConcatDataset([new]), seed=seed)
        for epoch in (0, 1, 7):
            a.set_epoch(epoch)
            b.set_epoch(epoch)
            assert list(a) == list(b)
