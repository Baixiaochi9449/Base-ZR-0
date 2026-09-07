import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from test_dataset_adapters_v3 import v3_root, FakeProcessor, _entry
from test_structured_slots import config
from utils.slot_labels import normalize_slot_labels
from utils.slot_supervision import SlotSupervisedDataset, SlotSupervisionReader, audit_slots


def test_stage2_never_reads_ar_or_state_columns(v3_root, monkeypatch):
    from utils.dataset_adapters import LeRobotV3FutureDifferenceDataset
    reads = []
    original = pq.read_table
    def read(path, *args, **kwargs):
        if "/data/" in str(path):
            reads.extend(kwargs.get("columns", []))
        return original(path, *args, **kwargs)
    monkeypatch.setattr(pq, "read_table", read)
    dataset = LeRobotV3FutureDifferenceDataset(entry=_entry(v3_root), processor=FakeProcessor(), loss_type="aux", max_length=1024)
    sample = dataset[0]
    assert not set(reads) & {"train_data", "slot_data", "state", "actions"}
    assert "labels" not in sample or not sample["labels"].ne(-100).any()
    assert "action" not in sample and "slot_data" not in sample


def test_stage2_anchor_filter_and_stage3_sampling(v3_root):
    from utils.dataset_adapters import LeRobotV3FutureDifferenceDataset
    labels = {"query_1": {"progress_t": .1, "progress_tK": .9}}
    reader = SimpleNamespace(root=v3_root, anchors={(0, 0): {"valid": True}},
        read=lambda ep, fr: normalize_slot_labels(labels, is_anchor=(ep, fr) == (0, 0)))
    for stage in ("aux", "vlm_and_action"):
        base = LeRobotV3FutureDifferenceDataset(entry=_entry(v3_root), processor=FakeProcessor(), loss_type=stage, max_length=1024)
        wrapped = SlotSupervisedDataset(base, reader, config())
        assert len(wrapped) == (1 if stage == "aux" else len(base))
        assert wrapped[0]["slot_Q1_mask"].all()
        if stage != "aux":
            assert not wrapped[1]["slot_Q1_mask"].any()
            assert wrapped[1]["action_mask"].any()


def test_any_aux_sampling_uses_valid_flow_frames(monkeypatch):
    class Dataset:
        spec = SimpleNamespace(dataset_path="/slots")
        requirements = None
        loss_type = "aux"
        steps = [(0, 0), (0, 1), (0, 2)]
        subset_indices = [0, 1, 2]
        indices = np.array([0, 1, 2])
        flow_reader = SimpleNamespace(eligible_frames=lambda threshold: pytest.fail("startup scanned masks"))
        def __len__(self):
            return 3
    reader = SimpleNamespace(root=Path("/slots"), anchors={})
    monkeypatch.setattr("utils.aux_sampling.load_flow_candidates", lambda dataset: np.array([1]))
    assert len(SlotSupervisedDataset(Dataset(), reader, config())) == 0
    flow = SimpleNamespace(enabled=True, flow_cell_min_valid_fraction=.5)
    wrapped = SlotSupervisedDataset(Dataset(), reader, config(stage2_aux_sampling="any_aux_valid"), flow)
    assert wrapped.indices == [1]


def test_source_anchor_verification_rejects_carried_label_forgery(tmp_path, monkeypatch):
    root, source, out = tmp_path / "data", tmp_path / "annotations", tmp_path / "audit"
    for directory in (root / "meta/episodes", root / "data", source / "episode_000000/shard_0/rank_0", out):
        directory.mkdir(parents=True)
    raw = {"query_1": {"progress_t": 0., "progress_tK": 1.},
           "query_2": {"phase_t_id": "approach", "phase_tK_id": "approach", "valid_mask": 1},
           "query_3": {"target_bbox_t": [0, 0, 1, 1], "target_bbox_tK": [0, 0, 1, 1], "valid_mask": [1, 1]},
           "query_4": {"gripper_bbox_t": [0, 0, 1, 1], "gripper_bbox_tK": [0, 0, 1, 1], "valid_mask": [1, 1]},
           "query_5": {"target_displacement_m": [0, 0, 0], "gripper_displacement_m": [1, 0, 0], "relative_displacement_m": [1, 0, 0], "valid_mask": [1, 1, 1]},
           "query_6": {"affordance_contact_points": [[.1, .2], [.3, .4]], "valid_mask_t": [1, 1]},
           "query_7": {"gripper_transition_id": "maintain_open", "valid_mask": 1},
           "query_8": {"contact_transition_id": "remain_separated", "valid_mask": 1},
           "query_9": {"obstacles": [{"bbox": [0, 0, 1, 1], "risk_score": .5}, {}, {}], "valid_mask": [1, 0, 0]}}
    info = {"total_episodes": 1, "total_frames": 3, "fps": 10, "splits": {"train": "0:1"}, "data_path": "data/rows.parquet"}
    (root / "meta/info.json").write_text(json.dumps(info))
    (root / "meta/stage05_merge.json").write_text(json.dumps({"stage05_dir": str(source)}))
    (root / "meta/stage05_episode_mapping.jsonl").write_text(json.dumps({"new_episode_index": 0, "old_episode_index": 0, "source_data_uri": "data/rows.parquet"}))
    pq.write_table(pa.Table.from_pylist([{"data/chunk_index": 0, "data/file_index": 0}]), root / "meta/episodes/rows.parquet")
    rows = [{"episode_index": 0, "frame_index": i, "index": i, "slot_data": json.dumps(raw)} for i in range(3)]
    pq.write_table(pa.Table.from_pylist(rows), root / "data/rows.parquet")
    record = {"base_data": {"semantic_anchor_frame": 0, "next_keyframe_frame": 2, "schema_version": "future_difference_training_v5",
                           "camera_key": "first_view", "fps": 10, "axis_convention": {"unit": "meter"}, "coordinate_frame": "per_keyframe_pinhole_camera"}, "slot_data": raw}
    (source / "episode_000000/shard_0/rank_0/training_samples.jsonl").write_text(json.dumps(record))
    report, stats, index = audit_slots(root, config())
    assert report["anchors"] == 1 and report["carried_labels_masked"] == 2
    (out / "slot_supervision_stats.json").write_text(json.dumps(stats))
    (out / "slot_anchor_index.json").write_text(json.dumps(index))
    reader = SlotSupervisionReader(out)
    assert reader.read(0, 0)["slot_Q1_mask"].all()
    assert not reader.read(0, 1)["slot_Q1_mask"].any()
    forged = [0, 1, dict(index["anchors"][0][2], global_index=1)]
    index["anchors"].append(forged)
    (out / "slot_anchor_index.json").write_text(json.dumps(index))
    with pytest.raises(ValueError, match="non-anchor"):
        SlotSupervisionReader(out)

    from utils.aux_data_contract import digest_file
    from utils.optical_flow_checkpoint import json_hash
    from utils.slot_supervision import slot_index_implementation_identity
    index["anchors"].pop()
    stats["anchor_contract"] = {"anchors_sha256": json_hash(index["anchors"]),
        "mapping_sha256": digest_file(root / "meta/stage05_episode_mapping.jsonl"), "annotation_root": str(source)}
    stats["index_implementation_identity"] = slot_index_implementation_identity()
    index.update(version=2, stats_sha256=json_hash(stats))
    (out / "slot_supervision_stats.json").write_text(json.dumps(stats))
    (out / "slot_anchor_index.json").write_text(json.dumps(index))
    with monkeypatch.context() as guard:
        guard.setattr("utils.slot_supervision._active_training_samples", lambda *args: pytest.fail("startup scanned annotations"))
        lazy = SlotSupervisionReader(out)
    assert not lazy.verified_episodes
    assert not lazy.read(0, 1)["slot_Q1_mask"].any()
    assert list(lazy.verified_episodes) == [0]
    path = source / "episode_000000/shard_0/rank_0/training_samples.jsonl"
    original = path.read_text()
    path.write_text(original + "\n")
    fresh = SlotSupervisionReader(out)
    with pytest.raises(ValueError, match="source changed"):
        fresh.read(0, 0)
    assert not fresh.verified_episodes and not fresh.cache
    path.write_text(original)
    assert fresh.read(0, 0)["slot_Q1_mask"].all()
    index["anchors"].append(forged)
    (out / "slot_anchor_index.json").write_text(json.dumps(index))
    with pytest.raises(ValueError, match="anchor/mapping"):
        SlotSupervisionReader(out)


def test_flow_eligibility_matches_training_mask_pool(tmp_path):
    from test_optical_flow_aux import fixture_manifest
    from utils.optical_flow_reader import OpticalFlowReader
    entry = fixture_manifest(tmp_path)
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps(entry) + "\n")
    reader = OpticalFlowReader(tmp_path, manifest)
    assert reader.eligible_frames(.5) == {(0, 3)}
    reader.close()


def test_stage05_uses_audited_frame_not_row_offset():
    class Dataset:
        spec = SimpleNamespace(dataset_path="/slots")
        requirements, loss_type, indices = None, "aux", [100, 101]
        def __len__(self):
            return 2
        def _episode_for_global(self, index):
            return {"episode_index": 0}, index - 100
        def __getitem__(self, index):
            return {"episode_id": torch.tensor(0), "frame_id": torch.tensor(10 + index)}
    calls = []
    reader = SimpleNamespace(root=Path("/slots"), anchors={(0, 10): {"valid": True, "global_index": 100}},
                             read=lambda ep, frame: calls.append((ep, frame)) or {})
    wrapped = SlotSupervisedDataset(Dataset(), reader, config())
    assert wrapped.indices == [0]
    wrapped[0]
    assert calls == [(0, 10)]
