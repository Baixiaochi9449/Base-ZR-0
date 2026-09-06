import io
import json
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from PIL import Image

from model.reasoning_vla_model import ZR0Model
from utils.stage05_canonical import (
    canonical_droid_chunk,
    canonical_molmo_chunk,
    canonical_rh20t_chunk,
    denormalize,
    normalize,
    quaternion_wxyz_to_rpy,
    rotvec_to_wrapped_rpy_delta,
)
from utils.stage05_dataset import Stage05MixedPretrainingDataset, build_stage05_message
from utils.stage05_sidecar import (
    GENERATOR_DEPENDENCY_PATHS,
    INCOMPLETE_MARKER_NAME,
    SIDECAR_FORMAT_VERSION,
    TEMP_DIRECTORY_PREFIX,
    build_stage05_sidecar,
    canonical_json_hash,
    generator_identity,
    load_stage05_sidecar,
)
from utils.stage05_roundtrip import RoundTripAccumulator
from utils.dataset_seen_tracker import DatasetSeenTracker
from utils.dataset_spec import resolve_dataset_spec, resolve_objective_requirements
from utils.load_training_dataset import EpochGroupedSampler, custom_collate_fn


TARGET = json.dumps(
    {
        "Task_temporal": "phase",
        "Spatial_motion": "motion",
        "Contact_interaction": "contact",
        "Object_constraints": "constraint",
    }
)


def _png(color):
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), color).save(buffer, format="PNG")
    return {"bytes": buffer.getvalue(), "path": None}


def _toy_molmo(root: Path):
    (root / "meta/episodes/chunk-000").mkdir(parents=True)
    (root / "data/chunk-000").mkdir(parents=True)
    info = {
        "codebase_version": "v3-test",
        "fps": 10,
        "features": {
            "first_view": {"dtype": "image", "shape": [8, 8, 3]},
            "second_view": {"dtype": "image", "shape": [8, 8, 3]},
            "wrist_image": {"dtype": "image", "shape": [8, 8, 3]},
            "state": {"dtype": "float32", "shape": [7]},
            "actions": {"dtype": "float32", "shape": [7]},
            "train_data": {"dtype": "string", "shape": [1]},
        },
    }
    (root / "meta/info.json").write_text(json.dumps(info))
    pq.write_table(
        pa.Table.from_pylist([{"task_index": 0, "task": "pick the object"}]),
        root / "meta/tasks.parquet",
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "episode_index": 0,
                    "tasks": ["pick the object"],
                    "length": 3,
                    "dataset_from_index": 0,
                    "dataset_to_index": 3,
                    "data/chunk_index": 0,
                    "data/file_index": 0,
                }
            ]
        ),
        root / "meta/episodes/chunk-000/file-000.parquet",
    )
    rows = []
    for index in range(3):
        action = [0.01 + index * 0.001, 0.002, 0.003, 0.01, 0.02, 0.03, index % 2]
        if index == 2:
            action[0] = float("nan")
        rows.append(
            {
                "index": index,
                "episode_index": 0,
                "frame_index": index,
                "task_index": 0,
                "train_data": TARGET,
                "slot_data": "retained but unused",
                "first_view": _png("red"),
                "second_view": _png("green"),
                "wrist_image": _png("blue"),
                "state": [0.4 + index * 0.01, 0.1, 0.2, 0.1, 0.2, 0.3, 0.0],
                "actions": action,
            }
        )
    pq.write_table(pa.Table.from_pylist(rows), root / "data/chunk-000/file-000.parquet")


def _video_metadata(episode, start, stop, task):
    row = {
        "episode_index": episode,
        "tasks": [task] if task else [],
        "length": stop - start,
        "dataset_from_index": start,
        "dataset_to_index": stop,
        "data/chunk_index": 0,
        "data/file_index": 0,
    }
    for key in ("observation.images.exterior_1_left", "observation.images.wrist_left"):
        row.update(
            {
                f"videos/{key}/chunk_index": 0,
                f"videos/{key}/file_index": 0,
                f"videos/{key}/from_timestamp": 0.0,
                f"videos/{key}/to_timestamp": float(stop - start) / 10,
            }
        )
    return row


def _write_toy_video_source(root: Path, episodes, rows, *, fps=10, extra_meta=None):
    (root / "meta/episodes/chunk-000").mkdir(parents=True)
    (root / "data/chunk-000").mkdir(parents=True)
    info = {"codebase_version": "v3-test", "fps": fps, "features": {}}
    (root / "meta/info.json").write_text(json.dumps(info))
    if extra_meta:
        for name, value in extra_meta.items():
            (root / "meta" / name).write_text(json.dumps(value))
    pq.write_table(
        pa.Table.from_pylist([{"task_index": 0, "task": "trusted task"}]),
        root / "meta/tasks.parquet",
    )
    pq.write_table(
        pa.Table.from_pylist(episodes), root / "meta/episodes/chunk-000/file-000.parquet"
    )
    pq.write_table(pa.Table.from_pylist(rows), root / "data/chunk-000/file-000.parquet")
    for key in ("observation.images.exterior_1_left", "observation.images.wrist_left"):
        path = root / "videos" / key / "chunk-000/file-000.mp4"
        path.parent.mkdir(parents=True)
        path.touch()


def test_two_view_message_order_and_variable_wrist():
    main = Image.new("RGB", (8, 8))
    wrist = Image.new("RGB", (8, 8))
    two = build_stage05_message("task", [("main camera", main), ("wrist camera", wrist)])
    one = build_stage05_message("task", [("main camera", main)])
    assert [item["text"] for item in two[0]["content"] if item["type"] == "text"][:2] == [
        "main camera", "wrist camera"
    ]
    assert sum(item["type"] == "image" for item in two[0]["content"]) == 2
    assert sum(item["type"] == "image" for item in one[0]["content"]) == 1
    with pytest.raises(ValueError, match="main camera"):
        build_stage05_message("task", [("wrist camera", wrist)])


def test_stage05_metadata_contract_is_collated_without_tensor_conversion():
    sample = {
        "input_ids": torch.ones(4, dtype=torch.long),
        "attention_mask": torch.ones(4, dtype=torch.long),
        "labels": torch.ones(4, dtype=torch.long),
        "task": "pick",
        "train_data": TARGET,
        "slot_data": "retained",
        "stats_key": "stage05_tabletop",
        "episode_id": torch.tensor(7),
        "frame_id": torch.tensor(11),
    }
    batch = custom_collate_fn([sample])
    assert batch["task"] == ["pick"]
    assert batch["train_data"] == [TARGET]
    assert batch["slot_data"] == ["retained"]
    assert batch["stats_key"] == ["stage05_tabletop"]
    assert batch["episode_id"].tolist() == [7]
    assert batch["frame_id"].tolist() == [11]


def test_stage05_projection_never_requests_second_external_view(monkeypatch, tmp_path):
    embedded = object.__new__(Stage05MixedPretrainingDataset)
    embedded.embedded_images = True
    embedded.kind = "molmo"
    embedded.loss_type = "vlm"
    requested = []

    def decode(value, identity):
        requested.append(value)
        return Image.new("RGB", (8, 8))

    embedded._decode_embedded = decode
    columns = embedded._columns()
    assert "second_view" not in columns
    images = embedded._images(
        0, {"first_view": "first", "wrist_image": None, "second_view": "forbidden"}, "toy"
    )
    assert [label for label, _ in images] == ["main camera"]
    assert requested == ["first"]

    video = object.__new__(Stage05MixedPretrainingDataset)
    video.embedded_images = False
    video.kind = "rh20t"
    video.loss_type = "vlm"
    decoded = []
    video._video_path = lambda episode, key: (tmp_path / key, 0.0)
    video._decode_video = lambda episode, key, timestamp: (
        decoded.append(key) or Image.new("RGB", (8, 8))
    )
    columns = video._columns()
    assert "observation.images.exterior_2_left" not in columns
    images = video._images(
        0,
        {
            "timestamp": 0.0,
            "observation.camera_sync.wrist_left.timestamp_offset_ms": 101,
        },
        "toy",
    )
    assert [label for label, _ in images] == ["main camera"]
    assert decoded == ["observation.images.exterior_1_left"]


def test_molmo_and_droid_native_h32_tail_masks_and_semantics():
    states = np.array(
        [[0.1, 0.2, 0.3, 0, 0, 0, 0], [0.2, 0.2, 0.3, 0, 0, 0, 1]], dtype=float
    )
    actions = np.array(
        [[0.1, 0, 0, 0, 0, 0, 0], [0.2, 0, 0, 0, 0, 0, 1]], dtype=float
    )
    molmo = canonical_molmo_chunk(states, actions, base=1, horizon=32)
    assert molmo.temporal_mask.tolist() == [True] + [False] * 31
    assert molmo.action[0, 6] == 0

    targets = states[:, :6].copy()
    targets[:, 0] += 0.01
    droid = canonical_droid_chunk(
        states[:, :6], [0.2, 0.8], targets, [0.0, 1.0], base=0, horizon=32
    )
    assert droid.temporal_mask.tolist() == [True, True] + [False] * 30
    np.testing.assert_allclose(droid.action[:2, 0], 0.01)
    np.testing.assert_array_equal(droid.action[:2, 6], [1, 0])


def test_rh20t_quaternion_conversion_and_invalid_action_mask():
    identity = [1.0, 0.0, 0.0, 0.0]
    states = np.asarray([[0.1, 0.2, 0.3, *identity, 0.8]] * 2)
    actions = np.asarray([[0, 0, 0, 0, 0, 0.1, 1], [0, 0, 0, 0, 0, 0, 0]])
    chunk = canonical_rh20t_chunk(states, actions, [True, False], base=0, horizon=32)
    assert chunk.temporal_mask.tolist() == [True] + [False] * 31
    np.testing.assert_allclose(quaternion_wxyz_to_rpy(identity), [0, 0, 0], atol=1e-6)
    np.testing.assert_allclose(
        rotvec_to_wrapped_rpy_delta([0, 0, 0], [0, 0, 0.1]), [0, 0, 0.1], atol=1e-6
    )


def test_normalization_roundtrip_for_each_stats_key():
    values = np.asarray([[0.2, -0.1, 0.5, -0.3, 0.4, 0.8, 1.0]])
    for stats_key in ("stage05_droid", "stage05_household", "stage05_tabletop", "stage05_rh20t"):
        q01 = np.arange(7, dtype=float) - 2
        q99 = q01 + np.arange(7, dtype=float) + 2
        restored = denormalize(normalize(values, q01, q99), q01, q99)
        np.testing.assert_allclose(restored, values, atol=1e-6, err_msg=stats_key)


def test_ar_and_joint_sidecars_are_independent_and_never_project_second_view(tmp_path, monkeypatch):
    root = tmp_path / "source"
    _toy_molmo(root)
    projected = []
    original = pq.read_table

    def audited(path, *args, **kwargs):
        if "/data/" in str(path):
            projected.extend(kwargs.get("columns") or [])
        return original(path, *args, **kwargs)

    monkeypatch.setattr("utils.stage05_sidecar.pq.read_table", audited)
    ar_dir = tmp_path / "ar"
    ar = build_stage05_sidecar(
        root=root, output=ar_dir, dataset_id="toy", kind="molmo",
        embedded_images=True, build_joint=False,
    )
    assert ar["counts"]["ar_eligible_frames"] == 3
    assert "state" not in projected and "actions" not in projected
    assert "second_view" not in projected
    ar_entry = {
        "dataset_path": str(root),
        "dataset_type": "vla",
        "dataset_adapter": "stage05_mixed_pretraining",
        "target_text_field": "train_data",
        "camera_keys": ["first_view", "wrist_image"],
        "ar_sidecar_path": str(ar_dir),
        "sample_ratio": 1.0,
    }
    requirements = resolve_objective_requirements(
        "vlm",
        adapter="stage05_mixed_pretraining",
        target_text_field="train_data",
        dataset_type="vla",
        dataset_entry="toy",
    )
    spec = resolve_dataset_spec(
        "toy", ar_entry, action_horizon=32, window_size=1, requirements=requirements
    )
    assert spec.stats_key is None and spec.normalization == "none"

    projected.clear()
    joint_dir = tmp_path / "joint"
    joint = build_stage05_sidecar(
        root=root, output=joint_dir, dataset_id="toy", kind="molmo",
        embedded_images=True, build_joint=True,
    )
    # Frame 2 has text but its entire one-step tail is invalid: AR yes, Joint no.
    assert joint["counts"]["ar_eligible_frames"] == 3
    assert joint["counts"]["joint_action_eligible_frames"] == 2
    assert np.load(joint_dir / "joint_indices.npy").tolist() == [0, 1]
    assert "second_view" not in projected

    loaded = load_stage05_sidecar(joint_dir)
    assert loaded["canonical_schema"]["action_pose"].startswith("native-next-step")
    rebuilt = build_stage05_sidecar(
        root=root,
        output=tmp_path / "joint-rebuilt",
        dataset_id="toy",
        kind="molmo",
        embedded_images=True,
        build_joint=True,
    )
    assert rebuilt["content_hash"] == joint["content_hash"]
    assert rebuilt["files"] == joint["files"]
    with (joint_dir / "joint_indices.npy").open("ab") as output:
        output.write(b"corrupt")
    with pytest.raises(ValueError, match="corrupt"):
        load_stage05_sidecar(joint_dir, verify_source=False)


def test_droid_missing_task_is_excluded_without_pseudo_task(tmp_path):
    root = tmp_path / "droid"
    episodes = [
        _video_metadata(0, 0, 2, "trusted task"),
        _video_metadata(1, 2, 4, ""),
    ]
    rows = []
    for index in range(4):
        episode = index // 2
        state = [0.4 + 0.01 * index, 0.0, 0.3, 0.0, 0.0, 0.0]
        target = [state[0] + 0.01, *state[1:]]
        gripper = [0.0]
        rows.append(
            {
                "index": index,
                "episode_index": episode,
                "frame_index": index % 2,
                "task_index": 0 if episode == 0 else 1,
                "train_data": TARGET,
                "slot_data": None,
                "observation.state.cartesian_position": state,
                "observation.state.gripper_position": gripper,
                "action.cartesian_position": target,
                "action.cartesian_velocity": [0.01, 0, 0, 0, 0, 0],
                "action.original": [*target, 0.0],
                "action.gripper_position": gripper,
                "action.gripper_velocity": [0.0],
                "timestamp": (index % 2) / 15,
                "language_instruction": "" if episode else "trusted task",
                "language_instruction_2": "",
                "language_instruction_3": "",
            }
        )
    _write_toy_video_source(root, episodes, rows, fps=15)
    forbidden = (
        root
        / "videos/observation.images.exterior_2_left/chunk-000/file-000.mp4"
    )
    forbidden.parent.mkdir(parents=True)
    forbidden.touch()
    manifest = build_stage05_sidecar(
        root=root,
        output=tmp_path / "droid-sidecar",
        dataset_id="toy-droid",
        kind="droid",
        embedded_images=False,
        build_joint=True,
    )
    assert manifest["counts"]["ar_eligible_frames"] == 2
    assert manifest["counts"]["joint_action_eligible_frames"] == 2
    assert not any(
        "exterior_2_left" in record["path"]
        for record in manifest["source_inventory"]["payload"]
    )
    episode_rows = pq.read_table(tmp_path / "droid-sidecar/episodes.parquet").to_pylist()
    assert episode_rows[1]["exclusion_reason"] == "missing_trusted_task"


def test_rh20t_episode_gates_and_zero_fm_chunks(tmp_path):
    root = tmp_path / "rh20t"
    specs = [
        (0, 3, True, True),
        (1, 1, False, True),
        (2, 1, False, False),
        (403, 1, True, True),
    ]
    episodes = []
    rows = []
    start = 0
    for episode, length, successful, rating_valid in specs:
        episodes.append(_video_metadata(episode, start, start + length, "trusted task"))
        for frame in range(length):
            rows.append(
                {
                    "index": start + frame,
                    "episode_index": episode,
                    "frame_index": frame,
                    "task_index": 0,
                    "train_data": TARGET,
                    "slot_data": None,
                    "observation.state": [0.4, 0.0, 0.3, 1.0, 0.0, 0.0, 0.0, 0.8],
                    "action": [0.001, 0, 0, 0, 0, 0, 1.0],
                    "action.valid": episode == 0 and frame == 0,
                    "is_episode_successful": successful,
                    "is_episode_successful_valid": rating_valid,
                    "observation.camera_sync.wrist_left.timestamp_offset_ms": 101 if frame == 0 else 0,
                    "timestamp": frame / 10,
                }
            )
        start += length
    _write_toy_video_source(root, episodes, rows)
    manifest = build_stage05_sidecar(
        root=root,
        output=tmp_path / "rh20t-sidecar",
        dataset_id="toy-rh20t",
        kind="rh20t",
        embedded_images=False,
        build_joint=True,
    )
    assert manifest["counts"]["ar_eligible_frames"] == 3
    assert manifest["counts"]["joint_action_eligible_frames"] == 1
    assert manifest["filter_counts"]["zero_fm_count_chunk"] == 2
    assert np.load(tmp_path / "rh20t-sidecar/joint_indices.npy").tolist() == [0]
    episode_rows = {
        row["episode_index"]: row
        for row in pq.read_table(tmp_path / "rh20t-sidecar/episodes.parquet").to_pylist()
    }
    assert episode_rows[1]["exclusion_reason"] == "failed_episode"
    assert episode_rows[2]["exclusion_reason"] == "unrated_episode"
    assert episode_rows[403]["exclusion_reason"] == "numeric_anomaly_episode"


def test_strict_joint_requires_fm_but_allows_empty_ar():
    batch = {
        "input_ids": torch.ones((1, 3), dtype=torch.long),
        "labels": torch.full((1, 3), -100, dtype=torch.long),
        "observation.state": torch.zeros((1, 1, 7)),
        "state_mask": torch.ones((1, 1, 7), dtype=torch.bool),
        "action": torch.zeros((1, 32, 7)),
        "action_mask": torch.ones((1, 32, 7), dtype=torch.bool),
        "action_supervision_available": torch.tensor([True]),
        "strict_joint_fm": torch.tensor([True]),
    }
    ZR0Model._validate_training_inputs(batch, "vlm_and_action")
    batch["action_mask"].zero_()
    with pytest.raises(ValueError, match="FM_count"):
        ZR0Model._validate_training_inputs(batch, "vlm_and_action")


def test_exact_seen_unique_duplicate_manifest_and_resume(tmp_path):
    class Accelerator:
        is_main_process = True

        @staticmethod
        def gather(value):
            return value

    class Dataset:
        manifest = {"counts": {"source_frames": 5}}
        spec = type("Spec", (), {"dataset_entry": "stage05_toy"})()

    concat = type("Concat", (), {"datasets": [Dataset()]})()
    tracker = DatasetSeenTracker(concat, Accelerator())
    tracker.update(
        [
            {
                "dataset_id": torch.tensor([0, 0, 0]),
                "sample_global_index": torch.tensor([1, 2, 1]),
                "ar_eligible": torch.tensor([1, 0, 1]),
                "fm_eligible": torch.tensor([1, 1, 1]),
            }
        ]
    )
    manifest = tracker.manifest(epoch=0, global_step=1)["datasets"][0]
    assert (manifest["seen"], manifest["unique"], manifest["duplicate"]) == (3, 2, 1)
    assert manifest["ar_eligible_seen"] == 2 and manifest["fm_eligible_seen"] == 3
    tracker.save(tmp_path, epoch=0, global_step=1)
    resumed = DatasetSeenTracker(concat, Accelerator(), resume_directory=tmp_path)
    assert resumed.manifest(epoch=0, global_step=1)["datasets"][0]["unique"] == 2


def test_stage05_sampler_full_epoch_ratio_is_natural_eligible_frame_ratio():
    class Dataset(torch.utils.data.Dataset):
        def __init__(self, lengths):
            self.lengths = lengths
            self.natural_mix_block_size = 128

        def __len__(self):
            return sum(self.lengths)

        def __getitem__(self, index):
            return index

        def sampling_group_ranges(self):
            result = []
            start = 0
            for length in self.lengths:
                result.append((start, start + length))
                start += length
            return result

    concat = torch.utils.data.ConcatDataset([Dataset([2, 3]), Dataset([4, 1, 2])])
    sampled = list(EpochGroupedSampler(concat, seed=42))
    assert sorted(sampled) == list(range(12))
    first_seen = sum(index < 5 for index in sampled)
    second_seen = sum(index >= 5 for index in sampled)
    assert (first_seen / len(sampled), second_seen / len(sampled)) == (5 / 12, 7 / 12)

    sizes = [673, 56, 22, 249]
    datasets = [Dataset([size]) for size in sizes]
    concat = torch.utils.data.ConcatDataset(datasets)
    prefix = list(EpochGroupedSampler(concat, seed=42))[:128]
    boundaries = np.cumsum([0, *sizes])
    counts = [
        sum(boundaries[index] <= value < boundaries[index + 1] for value in prefix)
        for index in range(len(sizes))
    ]
    expected = np.asarray(sizes) / sum(sizes) * 128
    assert np.max(np.abs(np.asarray(counts) - expected)) < 1


def test_sidecar_generator_identity_covers_target_canonicalization_and_is_stable(tmp_path):
    repository = tmp_path / "repository"
    for relative in GENERATOR_DEPENDENCY_PATHS:
        destination = repository / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(Path(__file__).resolve().parents[1] / relative, destination)
    first = generator_identity(repository)
    assert first == generator_identity(repository)
    target_dependency = repository / "utils/dataset_adapters.py"
    target_dependency.write_text(
        target_dependency.read_text(encoding="utf-8") + "\n# identity mutation\n",
        encoding="utf-8",
    )
    assert generator_identity(repository) != first
    assert {item["relative_path"] for item in first["dependencies"]} == set(
        GENERATOR_DEPENDENCY_PATHS
    )


def test_ar_generator_identity_remains_independent_of_joint_canonical_code(tmp_path):
    repository = tmp_path / "repository"
    for relative in GENERATOR_DEPENDENCY_PATHS:
        destination = repository / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(Path(__file__).resolve().parents[1] / relative, destination)
    ar_before = generator_identity(repository, build_joint=False)
    joint_before = generator_identity(repository, build_joint=True)
    canonical = repository / "utils/stage05_canonical.py"
    canonical.write_text(
        canonical.read_text(encoding="utf-8") + "\n# joint-only mutation\n",
        encoding="utf-8",
    )
    assert generator_identity(repository, build_joint=False) == ar_before
    assert generator_identity(repository, build_joint=True) != joint_before


def test_sidecar_format_version_mismatch_fails_before_other_validation(tmp_path):
    root = tmp_path / "source"
    _toy_molmo(root)
    output = tmp_path / "sidecar"
    build_stage05_sidecar(
        root=root, output=output, dataset_id="toy", kind="molmo",
        embedded_images=True, build_joint=True,
    )
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["sidecar_format_version"] = SIDECAR_FORMAT_VERSION - 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="format version mismatch"):
        load_stage05_sidecar(output, verify_source=False)


def test_sidecar_generation_parameter_hash_mismatch_fails_fast(tmp_path):
    root = tmp_path / "source"
    _toy_molmo(root)
    output = tmp_path / "sidecar"
    build_stage05_sidecar(
        root=root, output=output, dataset_id="toy", kind="molmo",
        embedded_images=True, build_joint=True,
    )
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["generation"]["video_backend"] = "changed"
    manifest["content_hash"] = canonical_json_hash(
        {key: value for key, value in manifest.items() if key != "content_hash"}
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="generation parameters are stale"):
        load_stage05_sidecar(output, verify_source=False)


@pytest.mark.parametrize("stage", ["after_first_npy", "before_manifest"])
def test_sidecar_partial_write_is_never_published_and_rebuilds(tmp_path, stage):
    root = tmp_path / "source"
    _toy_molmo(root)
    output = tmp_path / "sidecar"

    def fail(current, _temporary):
        if current == stage:
            raise RuntimeError("injected interruption")

    with pytest.raises(RuntimeError, match="injected"):
        build_stage05_sidecar(
            root=root, output=output, dataset_id="toy", kind="molmo",
            embedded_images=True, build_joint=True, _failure_injector=fail,
        )
    assert not output.exists()
    build_stage05_sidecar(
        root=root, output=output, dataset_id="toy", kind="molmo",
        embedded_images=True, build_joint=True,
    )
    assert load_stage05_sidecar(output)["counts"]["joint_action_eligible_frames"] == 2


def test_sidecar_validation_failure_does_not_publish(tmp_path):
    root = tmp_path / "source"
    _toy_molmo(root)
    output = tmp_path / "sidecar"

    def corrupt(stage, temporary):
        if stage == "before_validation":
            with (temporary / "ar_indices.npy").open("ab") as destination:
                destination.write(b"corrupt")

    with pytest.raises(ValueError, match="corrupt"):
        build_stage05_sidecar(
            root=root, output=output, dataset_id="toy", kind="molmo",
            embedded_images=True, build_joint=True, _failure_injector=corrupt,
        )
    assert not output.exists()


def test_sidecar_existing_output_is_not_overwritten_or_confused_with_similar_directory(tmp_path):
    root = tmp_path / "source"
    _toy_molmo(root)
    output = tmp_path / "sidecar"
    first = build_stage05_sidecar(
        root=root, output=output, dataset_id="toy", kind="molmo",
        embedded_images=True, build_joint=True,
    )
    manifest_bytes = (output / "manifest.json").read_bytes()
    similar = tmp_path / f"{TEMP_DIRECTORY_PREFIX}{output.name}-not-owned"
    similar.mkdir()
    (similar / "user-file").write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError, match="overwrite"):
        build_stage05_sidecar(
            root=root, output=output, dataset_id="toy", kind="molmo",
            embedded_images=True, build_joint=True,
        )
    assert (output / "manifest.json").read_bytes() == manifest_bytes
    assert first["content_hash"] == load_stage05_sidecar(output)["content_hash"]
    assert (similar / "user-file").read_text(encoding="utf-8") == "keep"


def test_owned_orphan_does_not_block_rebuild_and_is_not_mistaken_for_output(tmp_path):
    root = tmp_path / "source"
    _toy_molmo(root)
    output = tmp_path / "sidecar"
    orphan = tmp_path / f"{TEMP_DIRECTORY_PREFIX}{output.name}-old-interruption"
    orphan.mkdir()
    (orphan / INCOMPLETE_MARKER_NAME).write_text("unpublished\n", encoding="utf-8")
    (orphan / "partial.npy").write_bytes(b"partial")

    build_stage05_sidecar(
        root=root, output=output, dataset_id="toy", kind="molmo",
        embedded_images=True, build_joint=True,
    )

    assert load_stage05_sidecar(output)["counts"]["joint_action_eligible_frames"] == 2
    assert (orphan / "partial.npy").read_bytes() == b"partial"


def test_concurrent_sidecar_publish_has_exactly_one_winner(tmp_path):
    root = tmp_path / "source"
    _toy_molmo(root)
    output = tmp_path / "sidecar"
    barrier = threading.Barrier(2)

    def build():
        def synchronize(stage, _temporary):
            if stage == "before_publish":
                barrier.wait(timeout=10)

        try:
            build_stage05_sidecar(
                root=root,
                output=output,
                dataset_id="toy",
                kind="molmo",
                embedded_images=True,
                build_joint=True,
                _failure_injector=synchronize,
            )
            return "published"
        except OSError:
            return "lost_race"

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _index: build(), range(2)))
    assert sorted(results) == ["lost_race", "published"]
    assert load_stage05_sidecar(output)["counts"]["joint_action_eligible_frames"] == 2


def test_real_production_canonical_chunk_uses_clipping_aware_roundtrip():
    dataset = object.__new__(Stage05MixedPretrainingDataset)
    dataset.kind = "molmo"
    dataset.action_horizon = 32
    dataset._canonical_cache = __import__("collections").OrderedDict()
    rows = [
        {
            "state": [0.4, 0.1, 0.2, 0.1, 0.2, 0.3, 0.0],
            "actions": [0.01 + index * 0.001, 0, 0, 0, 0, 0, index % 2],
        }
        for index in range(4)
    ]
    chunk = dataset._canonical_chunk(7, rows, 0)
    stats = {
        "q01": np.asarray([-0.1] * 6 + [0.0], dtype=np.float32),
        "q99": np.asarray([0.1] * 6 + [1.0], dtype=np.float32),
    }
    identities = [
        {"dataset": "toy", "episode": 7, "frame": index, "time": index / 10}
        for index in range(32)
    ]
    accumulator = RoundTripAccumulator()
    valid = chunk.temporal_mask[:, None] & chunk.dimension_mask[None, :]
    accumulator.update(chunk.action, valid, stats, identities)
    assert all(item["valid_element_count"] == 4 for item in accumulator.report())


def test_clipping_audit_detects_saturation_and_nonzero_roundtrip_error():
    values = np.zeros((32, 7), dtype=np.float32)
    values[0, 2] = 100.0
    stats = {
        "q01": np.zeros(7, dtype=np.float32),
        "q99": np.ones(7, dtype=np.float32),
    }
    accumulator = RoundTripAccumulator()
    accumulator.update(
        values,
        np.ones_like(values, dtype=bool),
        stats,
        [
            {"dataset": "toy", "episode": 1, "frame": index, "time": index / 10}
            for index in range(32)
        ],
    )
    dimension = accumulator.report()[2]
    assert dimension["clipped_count"] == 1
    assert dimension["denormalization_error_max"] > 0
    assert dimension["worst_sample"]["frame"] == 0
    source = (Path(__file__).resolve().parents[1] / "scripts/audit_stage05_adapter_samples.py").read_text()
    assert "0.375" not in source
