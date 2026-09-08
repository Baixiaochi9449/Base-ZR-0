import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts.prepare_robotwin_eval_metadata import CAMERA_KEYS, prepare_metadata


def make_source(tmp_path):
    root = tmp_path / "source"
    (root / "meta/episodes/chunk-000").mkdir(parents=True)
    (root / "data/chunk-000").mkdir(parents=True)
    features = {key: {"dtype": "float32", "shape": [14]}
                for key in ("observation.state", "action")}
    features.update({key: {"dtype": "video", "shape": [480, 640, 3]}
                     for key in CAMERA_KEYS})
    info = {"codebase_version": "v3.0", "robot_type": "aloha", "total_episodes": 2,
            "total_frames": 101, "total_tasks": 1, "chunks_size": 1000,
            "fps": 30, "splits": {"train": "0:2"}, "features": features}
    (root / "meta/info.json").write_text(json.dumps(info))
    (root / "meta/stats.json").write_text('{"original": "preserved"}\n')
    pq.write_table(pa.Table.from_pylist([{"task_index": 0, "task": "Lift bottle"}]),
                   root / "meta/tasks.parquet")
    episodes = [{"episode_index": i, "tasks": ["Lift bottle"], "length": end - start,
                 "dataset_from_index": start, "dataset_to_index": end}
                for i, (start, end) in enumerate(((0, 100), (100, 101)))]
    pq.write_table(pa.Table.from_pylist(episodes), root / "meta/episodes/chunk-000/file-000.parquet")
    values = np.append(np.arange(100, dtype=np.float32), np.float32(10000))
    for i, (start, end) in enumerate(((0, 100), (100, 101))):
        state = np.repeat(values[start:end, None], 14, axis=1)
        table = pa.table({
            "observation.state": pa.array(state.tolist(), type=pa.list_(pa.float32())),
            "action": pa.array((2 * state + 1).tolist(), type=pa.list_(pa.float32())),
            "index": np.arange(start, end), "episode_index": [i] * (end - start),
            "frame_index": np.arange(end - start),
        })
        pq.write_table(table, root / f"data/chunk-000/file-{i:03d}.parquet")
    return root


def test_full_frame_quantiles_and_existing_loader(tmp_path):
    from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata

    source = make_source(tmp_path)
    output = tmp_path / "export"
    original = (source / "meta/stats.json").read_bytes()
    provenance = prepare_metadata(source, output, allow_unverified_stats=True)
    meta = LeRobotDatasetMetadata(repo_id="local-fixture", root=output)
    assert meta.total_episodes == 2 and len(meta.tasks) == 1
    assert tuple(meta.camera_keys) == CAMERA_KEYS
    assert meta.info["metadata_only"] and not provenance["checkpoint_match_verified"]
    assert provenance["frames_used"] == 101
    np.testing.assert_array_equal(meta.stats["observation.state"]["q01"], np.ones(14))
    np.testing.assert_array_equal(meta.stats["observation.state"]["q99"], np.full(14, 99))
    np.testing.assert_array_equal(meta.stats["action"]["q99"], np.full(14, 199))
    assert sum(row["length"] for row in meta.episodes.values()) == 101
    assert (source / "meta/stats.json").read_bytes() == original
    assert not (output / "data").exists() and not (output / "videos").exists()
    assert len(provenance["exported_files"]) == 4
    with pytest.raises(FileExistsError):
        prepare_metadata(source, output, allow_unverified_stats=True)


def test_requires_explicit_diagnostic_choice(tmp_path):
    with pytest.raises(ValueError, match="allow-unverified-stats"):
        prepare_metadata(tmp_path / "unused", tmp_path / "export")


def test_refuses_source_directory_output(tmp_path):
    source = make_source(tmp_path)
    with pytest.raises(ValueError, match="outside"):
        prepare_metadata(source, source / "export", allow_unverified_stats=True)


@pytest.mark.parametrize("mutation", ["nonfinite", "dimension", "index", "frame_count"])
def test_rejects_invalid_data_without_publishing(tmp_path, mutation):
    source = make_source(tmp_path)
    if mutation == "frame_count":
        path = source / "meta/info.json"
        info = json.loads(path.read_text())
        info["total_frames"] += 1
        path.write_text(json.dumps(info))
    else:
        path = source / "data/chunk-000/file-001.parquet"
        table = pq.read_table(path)
        if mutation == "index":
            key, value = "index", pa.array([98], type=pa.int64())
        else:
            key = "action"
            values = [float("nan")] * 14 if mutation == "nonfinite" else [0.0] * 13
            value = pa.array([values], type=pa.list_(pa.float32()))
        table = table.set_column(table.schema.get_field_index(key), key, value)
        pq.write_table(table, path)
    output = tmp_path / "export"
    with pytest.raises(ValueError):
        prepare_metadata(source, output, allow_unverified_stats=True)
    assert not output.exists()
