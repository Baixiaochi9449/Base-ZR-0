"""Export diagnostic v2 metadata with exact, locally recomputed quantiles."""

import argparse
import copy
import hashlib
import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CAMERA_KEYS = (
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)
VECTOR_KEYS = ("observation.state", "action")
STATUS = "local_recomputed_checkpoint_unverified"


def file_identity(path, root):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path.relative_to(root)), "bytes": path.stat().st_size,
            "sha256": digest.hexdigest()}


def read_source_metadata(source):
    info = json.loads((source / "meta/info.json").read_text())
    if info.get("codebase_version") != "v3.0":
        raise ValueError("source must be a LeRobot v3.0 dataset")
    for key in VECTOR_KEYS:
        if info["features"][key]["shape"] != [14]:
            raise ValueError(f"{key}: expected dimension 14")
    cameras = tuple(key for key, feature in info["features"].items()
                    if feature["dtype"] in ("image", "video"))
    if cameras != CAMERA_KEYS:
        raise ValueError(f"unexpected camera keys/order: {cameras}")
    episode_files = sorted((source / "meta/episodes").rglob("*.parquet"))
    episodes = []
    for path in episode_files:
        episodes.extend(pq.read_table(path, columns=[
            "episode_index", "tasks", "length", "dataset_from_index", "dataset_to_index",
        ]).to_pylist())
    episodes.sort(key=lambda row: row["episode_index"])
    tasks = pq.read_table(source / "meta/tasks.parquet",
                          columns=["task_index", "task"]).to_pylist()
    tasks.sort(key=lambda row: row["task_index"])
    if ([row["episode_index"] for row in episodes] != list(range(info["total_episodes"]))
            or [row["task_index"] for row in tasks] != list(range(info["total_tasks"]))):
        raise ValueError("metadata episode/task counts or indices are inconsistent")
    known_tasks = {row["task"] for row in tasks}
    if len(known_tasks) != len(tasks):
        raise ValueError("duplicate task strings")
    cursor = 0
    for row in episodes:
        if (row["length"] <= 0 or row["dataset_from_index"] != cursor
                or row["dataset_to_index"] != cursor + row["length"]
                or not row["tasks"] or not set(row["tasks"]).issubset(known_tasks)):
            raise ValueError(f"invalid episode metadata: {row['episode_index']}")
        cursor = row["dataset_to_index"]
    if cursor != info["total_frames"] or cursor <= 0:
        raise ValueError("episode lengths do not match total_frames")
    paths = [source / "meta/info.json", source / "meta/stats.json",
             source / "meta/tasks.parquet", *episode_files]
    return info, episodes, tasks, paths


def compute_stats(source, info, episodes, data_files):
    from lerobot.common.datasets.utils import get_stats

    count = info["total_frames"]
    if sum(pq.ParquetFile(path).metadata.num_rows for path in data_files) != count:
        raise ValueError("data row count does not match total_frames")
    vectors = {key: np.empty((count, 14), dtype=np.float32) for key in VECTOR_KEYS}
    starts = np.asarray([row["dataset_from_index"] for row in episodes])
    ends = np.asarray([row["dataset_to_index"] for row in episodes])
    columns = [*VECTOR_KEYS, "index", "episode_index", "frame_index"]
    cursor = 0
    for path in data_files:
        for batch in pq.ParquetFile(path).iter_batches(
                batch_size=65536, columns=columns, use_threads=False):
            stop = cursor + batch.num_rows
            if stop > count:
                raise ValueError("data grew during the export")
            indices = np.arange(cursor, stop)
            episode_ids = np.searchsorted(ends, indices, side="right")
            expected = {"index": indices, "episode_index": episode_ids,
                        "frame_index": indices - starts[episode_ids]}
            for key, values in expected.items():
                if not np.array_equal(batch.column(key).to_numpy(), values):
                    raise ValueError(f"{path}: {key} does not match episode metadata")
            for key in VECTOR_KEYS:
                column = batch.column(key)
                if column.null_count or not np.all(pc.list_value_length(column).to_numpy() == 14):
                    raise ValueError(f"{path}: {key} must contain non-null 14D vectors")
                flat = column.flatten()
                if flat.null_count:
                    raise ValueError(f"{path}: null values in {key}")
                values = flat.to_numpy().reshape(-1, 14)
                if values.dtype != np.float32 or not np.isfinite(values).all():
                    raise ValueError(f"{path}: {key} must be finite float32")
                vectors[key][cursor:stop] = values
            cursor = stop
        print(f"Read {path.name}: {cursor}/{count} frames", flush=True)
    if cursor != count:
        raise ValueError("incomplete frame scan")
    result = {}
    for key in VECTOR_KEYS:
        print(f"Computing full-dataset statistics: {key}", flush=True)
        result[key] = get_stats(vectors.pop(key))
    return result


def prepare_metadata(source, output, *, allow_unverified_stats=False):
    if not allow_unverified_stats:
        raise ValueError("explicit --allow-unverified-stats is required for diagnostic statistics")
    source = Path(source).expanduser().resolve()
    output = Path(output).expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite {output}")
    if output.resolve().is_relative_to(source):
        raise ValueError("output must be outside the original dataset")
    info, episodes, tasks, metadata_files = read_source_metadata(source)
    data_files = sorted((source / "data").rglob("*.parquet"))
    source_files = [*metadata_files, *data_files]
    before = [file_identity(path, source) for path in source_files]
    stats = compute_stats(source, info, episodes, data_files)
    if [file_identity(path, source) for path in source_files] != before:
        raise ValueError("source files changed during statistics computation")

    exported_info = copy.deepcopy(info)
    exported_info.update({
        "codebase_version": "v2.1", "metadata_only": True,
        "source_codebase_version": info["codebase_version"],
        "normalization_status": STATUS, "source_dataset_path": str(source),
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "total_chunks": (info["total_episodes"] + info["chunks_size"] - 1) // info["chunks_size"],
        "total_videos": 0,
    })
    for key in ("data_files_size_in_mb", "video_files_size_in_mb"):
        exported_info.pop(key, None)
    helper = REPO_ROOT / "lerobot/lerobot/common/datasets/utils.py"
    provenance = {
        "schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
        "status": STATUS, "checkpoint_match_verified": False, "metadata_only": True,
        "source_dataset_path": str(source), "source_codebase_version": "v3.0",
        "frames_used": info["total_frames"], "episodes_used": info["total_episodes"],
        "camera_keys": list(CAMERA_KEYS), "fps": info["fps"],
        "statistics_function": "lerobot.common.datasets.utils.get_stats",
        "statistics_input_dtype": "float32", "quantile_method": "numpy.percentile linear",
        "quantiles_percent": [1, 99], "numpy_version": np.__version__,
        "sampling": "all frames, each exactly once, sorted Parquet paths",
        "source_files": before, "helper_source": file_identity(helper, REPO_ROOT),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".robotwin-meta-", dir=output.parent) as tmp:
        export = Path(tmp) / "export"
        meta = export / "meta"
        meta.mkdir(parents=True)
        for name, payload in (("info.json", exported_info), ("stats.json", stats)):
            (meta / name).write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
        for name, rows in (("tasks.jsonl", tasks), ("episodes.jsonl", episodes)):
            with (meta / name).open("w") as stream:
                for row in rows:
                    if name == "episodes.jsonl":
                        row = {key: row[key] for key in ("episode_index", "tasks", "length")}
                    stream.write(json.dumps(row, ensure_ascii=True, allow_nan=False) + "\n")
        provenance["exported_files"] = [file_identity(path, export) for path in sorted(meta.iterdir())]
        (meta / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
        if output.exists():
            raise FileExistsError(f"refusing to overwrite {output}")
        export.rename(output)
    return provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-unverified-stats", action="store_true",
                        help="Accept local statistics only for diagnostic evaluation.")
    args = parser.parse_args()
    result = prepare_metadata(args.source, args.output, allow_unverified_stats=args.allow_unverified_stats)
    print(json.dumps({"output": str(args.output), "status": result["status"],
                      "frames_used": result["frames_used"]}, indent=2))


if __name__ == "__main__":
    main()
