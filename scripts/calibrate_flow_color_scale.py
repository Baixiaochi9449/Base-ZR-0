"""Bounded-memory calibration of the fixed V2 flow color scale."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pyarrow.parquet as pq

from utils.optical_flow_v2 import COLOR_PROTOCOL
from utils.optical_flow_reader import (
    CURRENT_FILE_SCHEMA, CURRENT_MANIFEST_SCHEMA, FlowFileIntegrityVerifier,
    VALID_FRACTION_ATOL,
)


class ReservoirQuantile:
    """Exact uniform reservoir sampling with memory bounded by capacity."""

    def __init__(self, capacity, seed):
        if type(capacity) is not int or capacity < 1:
            raise ValueError("reservoir capacity must be positive")
        self.capacity = capacity
        self.rng = np.random.default_rng(seed)
        self.values = np.empty(capacity, dtype=np.float32)
        self.size = 0
        self.seen = 0

    def update(self, values):
        values = np.asarray(values, dtype=np.float32).reshape(-1)
        if values.size == 0:
            return
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("flow magnitudes must be finite and non-negative")
        offset = min(self.capacity - self.size, values.size)
        if offset:
            self.values[self.size:self.size + offset] = values[:offset]
            self.size += offset
            self.seen += offset
            values = values[offset:]
        if values.size == 0:
            return
        incoming = int(values.size)
        selected = int(self.rng.hypergeometric(incoming, self.seen, self.capacity))
        if selected:
            source = self.rng.choice(incoming, size=selected, replace=False)
            destination = self.rng.choice(self.capacity, size=selected, replace=False)
            self.values[destination] = values[source]
        self.seen += incoming

    def quantile(self, value):
        if self.size == 0:
            raise ValueError("no valid flow values found")
        return float(np.quantile(self.values[:self.size], value))

    @property
    def resident_bytes(self):
        return self.values.nbytes


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_episode_splits(info):
    total = info.get("total_episodes")
    splits = info.get("splits")
    if type(total) is not int or total < 1 or not isinstance(splits, dict) or not splits:
        raise ValueError("dataset split authority has no valid episode split metadata")
    assignments = {}
    normalized = {}
    for split, interval in splits.items():
        if (not isinstance(split, str) or not split or not isinstance(interval, str)
                or interval.count(":") != 1):
            raise ValueError("dataset split authority has an unsupported episode range")
        start_text, stop_text = interval.split(":")
        if not start_text.isdigit() or not stop_text.isdigit():
            raise ValueError("dataset split authority has an unsupported episode range")
        start, stop = int(start_text), int(stop_text)
        if start < 0 or stop <= start or stop > total:
            raise ValueError("dataset split authority episode range is out of bounds")
        for episode in range(start, stop):
            if episode in assignments:
                raise ValueError("dataset split authority episode ranges overlap")
            assignments[episode] = split
        normalized[split] = [start, stop]
    if set(assignments) != set(range(total)):
        raise ValueError("dataset split authority does not cover every episode")
    return assignments, normalized


def _load_dataset_split_authority(dataset_root, rows, manifest_sha256):
    root = Path(dataset_root).resolve()
    info_path = root / "meta/info.json"
    mapping_path = root / "meta/stage05_episode_mapping.jsonl"
    merge_path = root / "meta/stage05_merge.json"
    if not info_path.is_file() or not mapping_path.is_file() or not merge_path.is_file():
        raise ValueError(
            "V2 calibration requires dataset meta/info.json, meta/stage05_episode_mapping.jsonl "
            "and meta/stage05_merge.json, or explicit externally_trusted mode"
        )
    try:
        info = json.loads(info_path.read_text())
        merge = json.loads(merge_path.read_text())
        mapping_rows = [json.loads(line) for line in mapping_path.read_text().splitlines() if line.strip()]
    except Exception as error:
        raise ValueError("dataset split authority metadata is malformed") from error
    assignments, split_ranges = _parse_episode_splits(info)
    manifest_dataset_ids = {row.get("dataset_id") for row in rows}
    if (len(manifest_dataset_ids) != 1
            or merge.get("expected_dataset_id") != next(iter(manifest_dataset_ids))):
        raise ValueError("Flow manifest dataset identity differs from the split authority source")

    # The Flow camera must be a real image/video feature in the same published
    # Stage05 modality contract.  Dataset IDs alone do not establish that a
    # label camera corresponds to an observation camera.
    features = info.get("features")
    modality_path = root / "meta/modality.json"
    if not isinstance(features, dict) or not modality_path.is_file():
        raise ValueError("dataset camera authority requires meta/info.json features and meta/modality.json")
    try:
        modality = json.loads(modality_path.read_text())
    except Exception as error:
        raise ValueError("dataset camera authority metadata is malformed") from error
    video_features = modality.get("video")
    if not isinstance(video_features, dict):
        raise ValueError("dataset camera authority has no video feature contract")
    camera_contract = {}
    for camera in sorted({row.get("camera_key") for row in rows}):
        feature = features.get(camera)
        if (not isinstance(feature, dict) or feature.get("dtype") not in {"image", "video"}
                or camera not in video_features or not isinstance(video_features[camera], dict)):
            raise ValueError(
                f"Flow camera {camera!r} is absent from the dataset image feature/modality contract"
            )
        shape = feature.get("shape")
        if (not isinstance(shape, list) or len(shape) != 3
                or any(type(value) is not int or value <= 0 for value in shape)):
            raise ValueError(f"dataset camera feature {camera!r} has no valid image shape")
        camera_contract[camera] = {
            "feature_dtype": feature["dtype"],
            "feature_shape": list(shape),
            "modality": video_features[camera],
        }

    # Read the compact episode metadata sidecar once.  It is the authoritative
    # mapping from a data chunk/file URI to the source episode and row count;
    # reading only these columns avoids scanning image/state columns.
    episode_metadata_paths = sorted((root / "meta/episodes").glob("**/*.parquet"))
    if not episode_metadata_paths:
        raise ValueError("dataset source identity requires meta/episodes parquet metadata")
    source_metadata = {}
    try:
        for metadata_path in episode_metadata_paths:
            table = pq.read_table(metadata_path, columns=[
                "episode_index", "data/chunk_index", "data/file_index", "length",
            ])
            for item in table.to_pylist():
                episode = item.get("episode_index")
                chunk = item.get("data/chunk_index")
                file_index = item.get("data/file_index")
                length = item.get("length")
                if (type(episode) is not int or episode < 0 or type(chunk) is not int or chunk < 0
                        or type(file_index) is not int or file_index < 0
                        or type(length) is not int or length < 1):
                    raise ValueError("dataset episode metadata has invalid source identity fields")
                key = (chunk, file_index)
                source_metadata.setdefault(key, []).append({"episode": episode, "length": length})
    except ValueError:
        raise
    except Exception as error:
        raise ValueError("dataset episode metadata is unreadable") from error

    def validate_source_uri(mapping_row):
        uri = mapping_row.get("source_data_uri")
        if not isinstance(uri, str) or not uri or "\x00" in uri:
            raise ValueError("stage05 episode mapping has no valid source_data_uri")
        candidate = Path(uri)
        resolved = (candidate if candidate.is_absolute() else root / candidate).resolve()
        try:
            relative = resolved.relative_to(root)
        except ValueError as error:
            raise ValueError(
                f"source_data_uri escapes dataset root: {uri!r} -> {resolved}"
            ) from error
        if not resolved.is_file() or resolved.suffix.lower() != ".parquet":
            raise ValueError(f"source_data_uri is not an in-root parquet file: {uri!r}")
        parts = relative.parts
        if len(parts) != 3 or parts[0] != "data" or not parts[1].startswith("chunk-"):
            raise ValueError(f"source_data_uri does not use the dataset data layout: {uri!r}")
        try:
            chunk = int(parts[1].removeprefix("chunk-"))
            file_index = int(Path(parts[2]).stem.removeprefix("file-"))
        except ValueError as error:
            raise ValueError(f"source_data_uri has no parseable chunk/file identity: {uri!r}") from error
        expected = source_metadata.get((chunk, file_index))
        if not expected:
            raise ValueError(f"source_data_uri is absent from dataset episode metadata: {uri!r}")
        return resolved, expected, (chunk, file_index)

    old_to_new = {}
    seen_new = set()
    source_file_checks = {}
    source_identities = {}
    for row in mapping_rows:
        old, new = row.get("old_episode_index"), row.get("new_episode_index")
        if (type(old) is not int or old < 0 or type(new) is not int or new < 0
                or old in old_to_new or new in seen_new):
            raise ValueError("dataset episode mapping is malformed or not one-to-one")
        path, source_meta, source_key = validate_source_uri(row)
        # `old_episode_index` is the source/original identity.  The Parquet URI
        # belongs to the merged Stage05 dataset, so its actual episode_index is
        # the mapped `new_episode_index`; this distinction is material for
        # sources whose episode IDs were remapped during Stage05.
        source_record = next((item for item in source_meta if item["episode"] == new), None)
        if source_record is None:
            raise ValueError(
                f"source_data_uri episode identity conflicts with "
                f"mapping target episode {new} (source episode {old}): {path}"
            )
        if source_key not in source_file_checks:
            try:
                before_stat = path.stat()
                parquet = pq.ParquetFile(path)
                episode_column = parquet.schema_arrow.field("episode_index")
                if episode_column is None:
                    raise ValueError("source parquet has no episode_index column")
                expected_episodes = {item["episode"] for item in source_meta}
                expected_rows = sum(item["length"] for item in source_meta)
                if parquet.metadata.num_rows != expected_rows:
                    raise ValueError("source parquet row count conflicts with episode metadata")
                minimum, maximum, observed_rows = None, None, 0
                for group_index in range(parquet.metadata.num_row_groups):
                    group = parquet.metadata.row_group(group_index)
                    column = next((group.column(index) for index in range(group.num_columns)
                                   if group.column(index).path_in_schema == "episode_index"), None)
                    if (column is None or column.statistics is None
                            or not column.statistics.has_min_max
                            or column.statistics.num_values != group.num_rows):
                        raise ValueError("source parquet episode_index statistics are unavailable")
                    minimum = column.statistics.min if minimum is None else min(minimum, column.statistics.min)
                    maximum = column.statistics.max if maximum is None else max(maximum, column.statistics.max)
                    observed_rows += column.statistics.num_values
                if (observed_rows != expected_rows or minimum != min(expected_episodes)
                        or maximum != max(expected_episodes)):
                    raise ValueError("source parquet episode_index statistics conflict with episode metadata")
                # Confirm the per-episode identity from the actual source
                # column.  Batches stay bounded and only the integer identity
                # column is read; image/state/action payloads are untouched.
                observed_counts = Counter()
                for batch in parquet.iter_batches(columns=["episode_index"], batch_size=65536):
                    values = batch.column(0).to_numpy(zero_copy_only=False)
                    if not np.issubdtype(values.dtype, np.integer):
                        raise ValueError("source parquet episode_index is not integral")
                    observed_counts.update(int(value) for value in values)
                expected_counts = Counter({item["episode"]: item["length"] for item in source_meta})
                if observed_counts != expected_counts:
                    raise ValueError("source parquet episode_index values conflict with episode metadata")
                after_stat = path.stat()
                if (before_stat.st_dev, before_stat.st_ino, before_stat.st_size,
                        before_stat.st_mtime_ns, before_stat.st_ctime_ns) != (
                            after_stat.st_dev, after_stat.st_ino, after_stat.st_size,
                            after_stat.st_mtime_ns, after_stat.st_ctime_ns):
                    raise ValueError("source parquet changed during identity verification")
                source_file_checks[source_key] = {
                    "episode_count": len(source_meta), "row_count": expected_rows,
                    "episode_min": minimum, "episode_max": maximum,
                }
            except ValueError:
                raise
            except Exception as error:
                raise ValueError(f"source_data_uri cannot verify parquet identity: {path}") from error
        old_to_new[old] = new
        seen_new.add(new)
        source_identities[str(new)] = {
            "source_episode_index": old,
            "target_episode_index": new,
            "source_data_uri": str(path.relative_to(root)),
            "source_data_uri_resolved": str(path),
            "source_data_size_bytes": path.stat().st_size,
            "source_data_mtime_ns": path.stat().st_mtime_ns,
            "source_data_inode": path.stat().st_ino,
            "source_episode_length": source_record["length"],
            "source_file_episode_count": len(source_meta),
            "source_file_row_count": source_file_checks[source_key]["row_count"],
        }
    if seen_new != set(range(info["total_episodes"])):
        raise ValueError("dataset episode mapping does not cover authoritative episode metadata")
    flow_to_dataset = {}
    for row in rows:
        flow_episode = row.get("merged_episode_index")
        source_episode = row.get("source_episode_index")
        if (type(flow_episode) is not int or flow_episode < 0 or type(source_episode) is not int
                or source_episode < 0 or flow_episode in flow_to_dataset
                or source_episode not in old_to_new):
            raise ValueError("Flow manifest cannot be mapped to the authoritative dataset split")
        dataset_episode = old_to_new[source_episode]
        if dataset_episode not in assignments:
            raise ValueError("mapped Flow episode is absent from the authoritative dataset splits")
        flow_to_dataset[flow_episode] = dataset_episode
    train_flow_episodes = {
        flow_episode for flow_episode, dataset_episode in flow_to_dataset.items()
        if assignments[dataset_episode] == "train"
    }
    return train_flow_episodes, {
        "mode": "dataset_metadata",
        "training_membership": "independently_verified",
        "dataset_root": str(root),
        "info_path": "meta/info.json",
        "info_sha256": _sha256(info_path),
        "episode_mapping_path": "meta/stage05_episode_mapping.jsonl",
        "episode_mapping_sha256": _sha256(mapping_path),
        "stage05_merge_path": "meta/stage05_merge.json",
        "stage05_merge_sha256": _sha256(merge_path),
        "source_dataset": merge.get("source_dataset"),
        "source_identity_sha256": merge.get("source_identity_sha256"),
        "flow_manifest_sha256": manifest_sha256,
        "split_ranges": split_ranges,
        "flow_to_dataset_episode_mapping": "manifest.source_episode_index_to_stage05.old_episode_index",
        "camera_contract": camera_contract,
        "modality_sha256": _sha256(modality_path),
        "source_identity": source_identities,
        "source_identity_contract": (
            "meta/episodes parquet episode_index/length + bounded actual source "
            "episode_index counts + in-root source_data_uri"
        ),
    }


def _load_training_frames(path, *, manifest_sha256, dataset_ids, cameras,
                          authoritative_train_episodes=None):
    if path is None:
        raise ValueError("V2 calibration requires an explicit manifest-bound training frame index")
    source = Path(path)
    try:
        payload = json.loads(source.read_text())
    except Exception as error:
        raise ValueError("training frame index must be a JSON split contract") from error
    required = {"version", "split", "selection", "flow_manifest_sha256",
                "dataset_ids", "cameras", "frames"}
    if (not isinstance(payload, dict) or set(payload) != required
            or payload.get("version") != 1 or payload.get("split") != "train"
            or payload.get("selection") != "explicit_episode_frame_allowlist"
            or payload.get("flow_manifest_sha256") != manifest_sha256
            or payload.get("dataset_ids") != sorted(dataset_ids)
            or payload.get("cameras") != sorted(cameras)
            or not isinstance(payload.get("frames"), list)):
        raise ValueError("training frame index conflicts with the Flow source/split contract")
    frames = set()
    for row in payload["frames"]:
        if not isinstance(row, dict) or set(row) != {"episode", "frame"}:
            raise ValueError("training index rows require exactly episode/frame")
        episode, frame = row.get("episode"), row.get("frame")
        if type(episode) is not int or episode < 0 or type(frame) is not int or frame < 0:
            raise ValueError("training index rows require non-negative episode/frame integers")
        if (episode, frame) in frames:
            raise ValueError("training index contains duplicate episode/frame")
        if (authoritative_train_episodes is not None
                and episode not in authoritative_train_episodes):
            raise ValueError(
                f"training index includes Flow episode {episode} outside the authoritative train split"
            )
        frames.add((episode, frame))
    if not frames:
        raise ValueError("training index is empty")
    return frames, _sha256(source), {key: payload[key] for key in required - {"frames"}}


def calibrate(root, manifest, *, expected_actual_delta, label_source=1,
              min_valid_fraction=0.95, reservoir_capacity=1_000_000,
              quantile=0.99, seed=0, split="train", training_index=None,
              dataset_root=None, externally_trusted=False):
    if split != "train":
        raise ValueError("V2 color scale calibration is restricted to the train split")
    if type(expected_actual_delta) is not int or expected_actual_delta < 1:
        raise ValueError("expected_actual_delta must be a positive integer")
    if type(label_source) is not int or label_source < 1:
        raise ValueError("label_source must be a positive integer")
    if not 0 < min_valid_fraction <= 1 or not 0 < quantile < 1:
        raise ValueError("valid fraction and quantile must be in (0,1]")
    root, manifest = Path(root), Path(manifest)
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError("flow manifest is empty")
    manifest_sha256 = _sha256(manifest)
    manifest_dataset_ids = {row.get("dataset_id") for row in rows}
    manifest_cameras = {row.get("camera_key") for row in rows}
    if (None in manifest_dataset_ids or None in manifest_cameras
            or any(not isinstance(value, str) or not value
                   for value in manifest_dataset_ids | manifest_cameras)):
        raise ValueError("flow manifest lacks dataset/camera source identity")
    if dataset_root is not None and externally_trusted:
        raise ValueError("choose dataset split authority or externally_trusted mode, not both")
    if dataset_root is not None:
        authoritative_train_episodes, split_authority = _load_dataset_split_authority(
            dataset_root, rows, manifest_sha256)
    elif externally_trusted:
        authoritative_train_episodes = None
        split_authority = {
            "mode": "externally_trusted_allowlist",
            "training_membership": "not_independently_verified",
            "flow_manifest_sha256": manifest_sha256,
        }
    else:
        raise ValueError(
            "V2 calibration requires --dataset-root split metadata; use externally_trusted only "
            "for a reviewed external allowlist"
        )
    training_frames, training_index_sha256, training_contract = _load_training_frames(
        training_index, manifest_sha256=manifest_sha256,
        dataset_ids=manifest_dataset_ids, cameras=manifest_cameras,
        authoritative_train_episodes=authoritative_train_episodes)
    matched_training_frames = set()
    observed_manifest_frames = set()
    reservoir = ReservoirQuantile(reservoir_capacity, seed)
    counts = {"manifest_frames": 0, "train_selected_frames": 0, "full_delta_frames": 0,
              "label_source_frames": 0, "valid_fraction_frames": 0}
    generation_ids = set()
    label_ids = set()
    dataset_ids = set()
    cameras = set()
    integrity = FlowFileIntegrityVerifier()
    for row in rows:
        episode = int(row["merged_episode_index"])
        path = root / row["hdf5_path"]
        if row.get("schema_version") != CURRENT_MANIFEST_SCHEMA:
            raise ValueError("V2 calibration requires the current Stage06 Flow manifest schema")
        required_identity = {name: row.get(name) for name in
                             ("dataset_id", "camera_key", "generation_identity", "label_identity", "sha256")}
        if any(not isinstance(value, str) or not value for value in required_identity.values()):
            raise ValueError("calibration label identity/content differs from manifest")
        integrity.verify(
            path, required_identity["sha256"],
            source=(f"calibration dataset={required_identity['dataset_id']}, "
                    f"camera={required_identity['camera_key']}, flow_episode={episode}"),
        )
        generation_ids.add(required_identity["generation_identity"])
        label_ids.add(required_identity["label_identity"])
        dataset_ids.add(required_identity["dataset_id"])
        cameras.add(required_identity["camera_key"])
        with h5py.File(path, "r") as handle:
            if handle.attrs.get("schema_version") != CURRENT_FILE_SCHEMA or "valid_fraction" not in handle:
                raise ValueError("V2 calibration requires the current Stage06 Flow file schema")
            for name in ("dataset_id", "camera_key", "generation_identity", "label_identity"):
                if handle.attrs.get(name) != required_identity[name]:
                    raise ValueError(f"calibration manifest/HDF5 identity mismatch: {name}")
            if int(handle.attrs["nominal_delta_frames"]) != expected_actual_delta:
                raise ValueError("manifest file nominal delta conflicts with calibration configuration")
            frames = np.asarray(handle["frame_index"][:], dtype=np.int64)
            deltas = np.asarray(handle["actual_delta_frames"][:])
            sources = np.asarray(handle["label_source"][:])
            stored_fractions = handle["valid_fraction"]
            if not (len(frames) == len(deltas) == len(sources) == int(row["frame_count"])):
                raise ValueError("flow manifest/HDF5 frame count mismatch")
            if (stored_fractions.shape != (len(frames),) or stored_fractions.dtype.kind != "f"):
                raise ValueError("invalid calibration valid_fraction schema")
            if len(np.unique(frames)) != len(frames):
                raise ValueError("flow label file contains duplicate frame indices")
            file_keys = {(episode, int(frame)) for frame in frames}
            if observed_manifest_frames & file_keys:
                raise ValueError("flow manifest contains duplicate episode/frame identities")
            observed_manifest_frames.update(file_keys)
            counts["manifest_frames"] += len(frames)
            for start in range(0, len(frames), 16):
                stop = min(start + 16, len(frames))
                keys = [(episode, int(frame)) for frame in frames[start:stop]]

                # Validate the complete block before applying train/delta/source
                # filters.  Excluded or tail rows are still part of the signed
                # Stage06 payload and cannot hide malformed values.
                block_size = stop - start
                raw_masks = np.asarray(handle["valid_mask"][start:stop])
                if raw_masks.shape != (block_size, 1, 224, 224):
                    raise ValueError("invalid calibration valid_mask shape")
                if raw_masks.dtype.kind not in "biuf":
                    raise ValueError("invalid calibration valid_mask dtype")
                if raw_masks.dtype.kind == "f" and not np.isfinite(raw_masks).all():
                    raise ValueError("calibration valid_mask contains non-finite values")
                if not np.isin(raw_masks, (0, 1)).all():
                    raise ValueError("calibration valid_mask must contain only 0/1")
                masks = raw_masks.astype(bool, copy=False)
                declared_fractions = np.asarray(stored_fractions[start:stop], dtype=np.float64)
                fractions = masks.reshape(block_size, -1).mean(axis=1)
                if (not np.isfinite(declared_fractions).all()
                        or ((declared_fractions < 0) | (declared_fractions > 1)).any()
                        or not np.allclose(declared_fractions, fractions, rtol=0,
                                           atol=VALID_FRACTION_ATOL)):
                    raise ValueError("calibration valid_fraction does not match valid_mask")
                flows = np.asarray(handle["flow"][start:stop])
                if (flows.shape != (block_size, 2, 224, 224)
                        or flows.dtype.kind != "f" or not np.isfinite(flows).all()):
                    raise ValueError("invalid calibration flow values")

                split_ok = np.asarray([key in training_frames for key in keys], dtype=bool)
                matched_training_frames.update(key for key, selected in zip(keys, split_ok) if selected)
                counts["train_selected_frames"] += int(split_ok.sum())
                delta_ok = split_ok & (deltas[start:stop] == expected_actual_delta)
                counts["full_delta_frames"] += int(delta_ok.sum())
                source_ok = delta_ok & (sources[start:stop] == label_source)
                counts["label_source_frames"] += int(source_ok.sum())
                if not source_ok.any():
                    continue
                quality_ok = source_ok & (fractions >= min_valid_fraction)
                counts["valid_fraction_frames"] += int(quality_ok.sum())
                if not quality_ok.any():
                    continue
                for local in np.flatnonzero(quality_ok):
                    magnitude = np.sqrt(np.square(flows[local, 0]) + np.square(flows[local, 1]))
                    reservoir.update(magnitude[masks[local, 0]])
        integrity.verify(
            path, required_identity["sha256"],
            source=(f"calibration dataset={required_identity['dataset_id']}, "
                    f"camera={required_identity['camera_key']}, flow_episode={episode}"),
        )
    missing_training_frames = training_frames - matched_training_frames
    if missing_training_frames:
        example = sorted(missing_training_frames)[:3]
        raise ValueError(f"training frame index names frames outside the Flow manifest: {example}")
    scale = reservoir.quantile(quantile)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("qualified training flow has no positive calibration scale")
    sample = reservoir.values[:reservoir.size]
    return {
        "version": 4,
        "split": "train",
        "training_selection": (
            "authoritative_dataset_split_subset" if dataset_root is not None
            else "externally_trusted_episode_frame_allowlist"
        ),
        "split_authority": split_authority,
        "training_index_sha256": training_index_sha256,
        "training_split_contract": training_contract,
        "training_index_entry_count": len(training_frames),
        "included_scope": {
            "episode_count": len({episode for episode, _ in matched_training_frames}),
            "episode_min": min(episode for episode, _ in matched_training_frames),
            "episode_max": max(episode for episode, _ in matched_training_frames),
            "frame_min": min(frame for _, frame in matched_training_frames),
            "frame_max": max(frame for _, frame in matched_training_frames),
        },
        "manifest_sha256": manifest_sha256,
        "generation_identities": sorted(value for value in generation_ids if isinstance(value, str)),
        "label_identities": sorted(label_ids),
        "dataset_ids": sorted(dataset_ids),
        "cameras": sorted(cameras),
        "camera_contract": split_authority.get("camera_contract"),
        "mapping_source_identity": split_authority.get("source_identity"),
        "color_protocol": COLOR_PROTOCOL,
        "flow_color_scale": scale,
        "units": "normalized_source_image_extent",
        "quantile": quantile,
        "quantile_kind": "estimated_from_uniform_reservoir",
        "sampling_algorithm": "chunked_hypergeometric_uniform_reservoir_v1",
        "reservoir_capacity": reservoir.capacity,
        "sampled_values": reservoir.size,
        "observed_valid_values": reservoir.seen,
        "resident_bytes": reservoir.resident_bytes,
        "seed": seed,
        "expected_actual_delta": expected_actual_delta,
        "label_source": label_source,
        "tail_short_delta": "excluded",
        "min_valid_fraction": min_valid_fraction,
        "estimated_truncation_rate": float((sample > scale).mean()),
        "frame_filter_counts": counts,
        "qualified_frame_coverage": counts["valid_fraction_frames"] / max(counts["train_selected_frames"], 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", required=True, choices=("train",))
    parser.add_argument("--expected-actual-delta", type=int, required=True)
    parser.add_argument("--label-source", type=int, default=1)
    parser.add_argument("--min-valid-fraction", type=float, default=0.95)
    parser.add_argument("--training-index", required=True,
                        help="JSON train split contract bound to this Flow manifest and its episode/frame allowlist")
    parser.add_argument("--dataset-root", default=None,
                        help="dataset root containing authoritative info, Stage05 mapping and merge identity")
    parser.add_argument("--externally-trusted-training-index", action="store_true",
                        help="accept a reviewed allowlist when no independently verifiable split metadata exists")
    parser.add_argument("--reservoir-capacity", type=int, default=1_000_000)
    parser.add_argument("--quantile", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    result = calibrate(args.root, args.manifest,
        expected_actual_delta=args.expected_actual_delta, label_source=args.label_source,
        min_valid_fraction=args.min_valid_fraction, reservoir_capacity=args.reservoir_capacity,
        quantile=args.quantile, seed=args.seed, split=args.split, training_index=args.training_index,
        dataset_root=args.dataset_root, externally_trusted=args.externally_trusted_training_index)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite locked calibration: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
