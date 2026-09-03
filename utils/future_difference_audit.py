"""Read-only audits for future-difference token lengths and action intervals."""

from __future__ import annotations

import bisect
import hashlib
import json
import math
import pickle
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import pyarrow.parquet as pq
from PIL import Image


TEXT_AUDIT_COLUMNS = ("episode_index", "frame_index", "task_index", "train_data")
INTERVAL_CLASSIFICATIONS = ("exact", "full", "partial", "none")
INTERVAL_DATASET_IDENTITY_PATHS = (
    "meta/info.json",
    "meta/steps_data_index.pkl",
    "meta/stage05_episode_mapping.jsonl",
    "meta/stage05_merge.json",
)


class FutureDifferenceTokenMeasurer:
    """Measure exact adapter token boundaries using metadata-sized dummy images."""

    def __init__(self, processor, camera_shapes: Mapping[str, Iterable[int]]):
        if len(camera_shapes) != 3:
            raise ValueError("camera_shapes must describe exactly three current cameras")
        self.processor = processor
        self.images = []
        for camera_key, raw_shape in camera_shapes.items():
            shape = tuple(int(value) for value in raw_shape)
            if len(shape) != 3 or shape[2] != 3 or min(shape) < 1:
                raise ValueError(f"camera {camera_key!r} must have a positive [H,W,3] shape")
            self.images.append((camera_key, Image.new("RGB", (shape[1], shape[0]))))
        self._layout_by_task: dict[str, tuple[int, int]] = {}
        self._target_length_cache: dict[str, int] = {}

    def _target_length(self, target: str) -> int:
        if target not in self._target_length_cache:
            from utils.dataset_adapters import encode_future_difference_target_tokens

            length = len(
                encode_future_difference_target_tokens(self.processor.tokenizer, target)
            )
            if length < 1:
                raise ValueError("empty target token sequence")
            self._target_length_cache[target] = length
        return self._target_length_cache[target]

    def _layout(self, task: str, target: str) -> tuple[int, int, int]:
        target_length = self._target_length(target)
        if task not in self._layout_by_task:
            from utils.dataset_adapters import (
                build_future_difference_message,
                measure_future_difference_token_boundary,
            )

            messages = build_future_difference_message(task, self.images, target)
            _, boundary = measure_future_difference_token_boundary(
                messages, self.processor, target
            )
            measured_target_length = int(boundary["target_ids"].numel())
            if measured_target_length != target_length:
                raise ValueError("processor target token boundary is inconsistent")
            context_length = int(boundary["assistant_start"])
            termination_length = int(boundary["termination_ids"].numel())
            if termination_length < 1:
                raise ValueError("processor returned an invalid target boundary")
            self._layout_by_task[task] = (context_length, termination_length)
        context_length, termination_length = self._layout_by_task[task]
        return context_length, target_length, termination_length

    def __call__(
        self, task: str, target: str, max_length: int, sample_id: str
    ) -> dict[str, Any]:
        if max_length < 1:
            raise ValueError("max_length must be positive")
        try:
            context_length, target_length, termination_length = self._layout(task, target)
        except ValueError as error:
            raise ValueError(f"{sample_id}: {error}") from error
        target_region_length = target_length + termination_length
        projected = context_length + target_region_length
        kept_region = max(0, min(target_region_length, max_length - context_length))
        kept_target = min(target_length, kept_region)
        kept_termination = max(0, kept_region - target_length)
        target_truncated = kept_region < target_region_length
        return {
            "context_tokens": context_length,
            "original_target_tokens": target_length,
            "kept_target_tokens": kept_target,
            "chat_termination_tokens": termination_length,
            "kept_chat_termination_tokens": kept_termination,
            "original_target_region_tokens": target_region_length,
            "kept_target_region_tokens": kept_region,
            "supervised_tokens": kept_region,
            "projected_sequence_tokens": projected,
            "input_truncated": context_length > max_length,
            "target_truncated": target_truncated,
        }


def _linear_percentile(sorted_values: list[int], percentile: float) -> float | int:
    if not sorted_values:
        raise ValueError("cannot compute a percentile of an empty collection")
    position = (len(sorted_values) - 1) * percentile / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def _distribution(values: Iterable[int]) -> dict[str, float | int | None]:
    ordered = sorted(int(value) for value in values)
    if not ordered:
        return {
            "min": None,
            "max": None,
            "mean": None,
            "p50": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "p99.9": None,
        }
    return {
        "min": ordered[0],
        "max": ordered[-1],
        "mean": sum(ordered) / len(ordered),
        "p50": _linear_percentile(ordered, 50.0),
        "p90": _linear_percentile(ordered, 90.0),
        "p95": _linear_percentile(ordered, 95.0),
        "p99": _linear_percentile(ordered, 99.0),
        "p99.9": _linear_percentile(ordered, 99.9),
    }


def summarize_token_lengths(
    records: Iterable[Mapping[str, Any]], max_length: int
) -> dict[str, Any]:
    if max_length < 1:
        raise ValueError("max_length must be positive")
    context_lengths: list[int] = []
    target_lengths: list[int] = []
    kept_target_lengths: list[int] = []
    termination_lengths: list[int] = []
    target_region_lengths: list[int] = []
    supervised_lengths: list[int] = []
    projected_lengths: list[int] = []
    input_truncated_count = 0
    input_only_count = 0
    target_only_count = 0
    input_and_target_count = 0
    target_truncated_count = 0
    sequence_overflow_count = 0

    for record in records:
        context = int(record["context_tokens"])
        target = int(record["original_target_tokens"])
        kept_target = int(record["kept_target_tokens"])
        termination = int(record.get("chat_termination_tokens", 0))
        target_region = int(
            record.get("original_target_region_tokens", target + termination)
        )
        kept_region = int(record.get("kept_target_region_tokens", kept_target))
        supervised = int(record["supervised_tokens"])
        projected = int(record["projected_sequence_tokens"])
        if min(
            context,
            target,
            kept_target,
            termination,
            target_region,
            kept_region,
            supervised,
            projected,
        ) < 0:
            raise ValueError("token counts must be non-negative")
        if (
            kept_target > target
            or target_region != target + termination
            or kept_region > target_region
            or supervised > kept_region
        ):
            raise ValueError("kept/supervised target counts are inconsistent")
        input_truncated = bool(record["input_truncated"])
        target_truncated = bool(record["target_truncated"])
        if target_truncated != (kept_region < target_region):
            raise ValueError("target_truncated must agree with kept target-region tokens")

        context_lengths.append(context)
        target_lengths.append(target)
        kept_target_lengths.append(kept_target)
        termination_lengths.append(termination)
        target_region_lengths.append(target_region)
        supervised_lengths.append(supervised)
        projected_lengths.append(projected)
        input_truncated_count += int(input_truncated)
        input_only_count += int(input_truncated and not target_truncated)
        target_only_count += int(target_truncated and not input_truncated)
        input_and_target_count += int(input_truncated and target_truncated)
        target_truncated_count += int(target_truncated)
        sequence_overflow_count += int(projected > max_length)

    sample_count = len(target_lengths)
    denominator = sample_count or 1
    return {
        "sample_count": sample_count,
        "max_length": max_length,
        "sequence_overflow_count": sequence_overflow_count,
        "sequence_overflow_ratio": sequence_overflow_count / denominator,
        "input_truncated_count": input_truncated_count,
        "input_truncated_ratio": input_truncated_count / denominator,
        "input_only_truncated_count": input_only_count,
        "input_only_truncated_ratio": input_only_count / denominator,
        "target_only_truncated_count": target_only_count,
        "target_only_truncated_ratio": target_only_count / denominator,
        "input_and_target_truncated_count": input_and_target_count,
        "input_and_target_truncated_ratio": input_and_target_count / denominator,
        "target_truncated_count": target_truncated_count,
        "target_truncated_ratio": target_truncated_count / denominator,
        "context_tokens": _distribution(context_lengths),
        "original_target_tokens": _distribution(target_lengths),
        "kept_target_tokens": _distribution(kept_target_lengths),
        "chat_termination_tokens": _distribution(termination_lengths),
        "target_region_tokens": _distribution(target_region_lengths),
        "supervised_tokens": _distribution(supervised_lengths),
        "projected_sequence_tokens": _distribution(projected_lengths),
    }


def _read_parquet_columns(path: Path, columns: Iterable[str]):
    return pq.read_table(path, columns=list(columns))


def _load_tasks(dataset_root: Path) -> dict[int, str]:
    table = _read_parquet_columns(dataset_root / "meta" / "tasks.parquet", ("task_index", "task"))
    tasks = {}
    for row in table.to_pylist():
        task_index = int(row["task_index"])
        if task_index in tasks:
            raise ValueError(f"duplicate task_index {task_index}")
        tasks[task_index] = row["task"]
    return tasks


def _data_paths(dataset_root: Path, info: Mapping[str, Any]) -> list[Path]:
    metadata_paths = sorted((dataset_root / "meta" / "episodes").rglob("*.parquet"))
    if not metadata_paths:
        raise FileNotFoundError(f"no episode metadata under {dataset_root / 'meta' / 'episodes'}")
    locations: set[tuple[int, int]] = set()
    for metadata_path in metadata_paths:
        table = _read_parquet_columns(
            metadata_path, ("data/chunk_index", "data/file_index")
        )
        for row in table.to_pylist():
            locations.add((int(row["data/chunk_index"]), int(row["data/file_index"])))
    template = info.get("data_path")
    if not isinstance(template, str) or not template:
        raise ValueError("meta/info.json must define data_path")
    return [
        dataset_root / template.format(chunk_index=chunk, file_index=file_index)
        for chunk, file_index in sorted(locations)
    ]


def audit_future_difference_lengths(
    dataset_root: str | Path,
    *,
    max_length: int,
    measure_sample: Callable[[str, str, int, str], Mapping[str, Any]],
    canonicalize_target: Callable[[Any, str], str] | None = None,
    progress_sample: Callable[[int], None] | None = None,
) -> dict[str, Any]:
    """Audit every published row without reading image, state, or action columns."""
    root = Path(dataset_root).resolve()
    if max_length < 1:
        raise ValueError("max_length must be positive")
    info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    tasks = _load_tasks(root)
    if canonicalize_target is None:
        from utils.dataset_adapters import canonicalize_future_difference_target

        canonicalize_target = canonicalize_future_difference_target

    sample_index = 0

    def records():
        nonlocal sample_index
        for data_path in _data_paths(root, info):
            table = _read_parquet_columns(data_path, TEXT_AUDIT_COLUMNS)
            for row in table.to_pylist():
                episode = int(row["episode_index"])
                frame = int(row["frame_index"])
                current_index = sample_index
                sample_index += 1
                sample_id = f"episode={episode} frame={frame} sample={current_index}"
                task_index = int(row["task_index"])
                if task_index not in tasks:
                    raise ValueError(f"{sample_id}: task_index {task_index} is missing")
                target = canonicalize_target(row["train_data"], sample_id)
                measurement = measure_sample(tasks[task_index], target, max_length, sample_id)
                if progress_sample is not None:
                    progress_sample(sample_index)
                yield measurement

    result = summarize_token_lengths(records(), max_length=max_length)
    expected_frames = info.get("total_frames")
    if expected_frames is not None and result["sample_count"] != int(expected_frames):
        raise ValueError(
            f"audited {result['sample_count']} rows, expected meta/info.json total_frames={expected_frames}"
        )
    result.update(
        {
            "dataset_root": str(root),
            "columns_read": list(TEXT_AUDIT_COLUMNS),
            "scope": "all_published_frames_without_training_eligible_filter",
        }
    )
    return result


@dataclass(frozen=True)
class IntervalOverlap:
    classification: str
    annotation_length: int
    chunk_length: int
    intersection_length: int


def classify_interval_overlap(
    chunk_start: int, chunk_end: int, annotation_start: int, annotation_end: int
) -> IntervalOverlap:
    values = (chunk_start, chunk_end, annotation_start, annotation_end)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise TypeError("interval endpoints must be integers")
    if chunk_start > chunk_end:
        raise ValueError("chunk_start must not exceed chunk_end")
    if annotation_start > annotation_end:
        raise ValueError("annotation_start must not exceed annotation_end")
    intersection = max(
        0, min(chunk_end, annotation_end) - max(chunk_start, annotation_start) + 1
    )
    chunk_length = chunk_end - chunk_start + 1
    annotation_length = annotation_end - annotation_start + 1
    if chunk_start == annotation_start and chunk_end == annotation_end:
        classification = "exact"
    elif intersection == chunk_length:
        classification = "full"
    elif intersection:
        classification = "partial"
    else:
        classification = "none"
    return IntervalOverlap(classification, annotation_length, chunk_length, intersection)


def _load_steps(dataset_root: Path) -> list[tuple[int, int]]:
    path = dataset_root / "meta" / "steps_data_index.pkl"
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict) or not isinstance(payload.get("steps"), list):
        raise ValueError(f"{path}: expected a dict containing a steps list")
    steps = []
    for sample_index, step in enumerate(payload["steps"]):
        if not isinstance(step, (tuple, list)) or len(step) != 2:
            raise ValueError(f"{path}: invalid step at sample {sample_index}")
        steps.append((int(step[0]), int(step[1])))
    return steps


def _load_episode_mapping(dataset_root: Path) -> dict[int, dict[str, Any]]:
    path = dataset_root / "meta" / "stage05_episode_mapping.jsonl"
    mapping = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            new_episode = int(row["new_episode_index"])
            if new_episode in mapping:
                raise ValueError(f"{path}:{line_number}: duplicate new_episode_index {new_episode}")
            mapping[new_episode] = row
    return mapping


def _resolve_annotation_root(
    dataset_root: Path, annotation_root: str | Path | None
) -> tuple[Path | None, str]:
    if annotation_root is not None:
        candidate = Path(annotation_root).expanduser().resolve()
        return (candidate if candidate.is_dir() else None, "command_line")
    manifest_path = dataset_root / "meta" / "stage05_merge.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        configured = manifest.get("stage05_dir")
        if isinstance(configured, str) and configured:
            candidate = Path(configured).expanduser()
            if not candidate.is_absolute():
                candidate = dataset_root / candidate
            candidate = candidate.resolve()
            if candidate.is_dir():
                return candidate, "meta/stage05_merge.json:stage05_dir"
    return None, "unavailable"


def _active_training_samples(annotation_root: Path, old_episode: int) -> Path | None:
    episode_root = annotation_root / f"episode_{old_episode:06d}"
    status_paths = sorted(episode_root.glob("shard_*/rank_*/status.json"))
    for status_path in status_paths:
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
            artifact_path = Path(status["artifacts"]["training_samples"]["path"])
        except (KeyError, TypeError, json.JSONDecodeError):
            continue
        candidates = [
            artifact_path,
            status_path.parent / artifact_path.name,
            annotation_root.parents[1] / artifact_path
            if len(annotation_root.parents) > 1
            else artifact_path,
        ]
        for candidate in candidates:
            if candidate.is_file():
                return candidate.resolve()
    candidates = sorted(episode_root.glob("shard_*/rank_*/training_samples*.jsonl"))
    return candidates[0].resolve() if len(candidates) == 1 else None


@dataclass(frozen=True)
class _AnnotationInterval:
    anchor: int
    start: int | None
    end: int | None
    sample_id: str | None
    error: str | None = None


def _load_annotation_intervals(path: Path) -> list[_AnnotationInterval]:
    intervals = []
    seen_anchors = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                base = record["base_data"]
                anchor = int(base["semantic_anchor_frame"])
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(f"{path}:{line_number}: malformed annotation record") from error
            if anchor in seen_anchors:
                raise ValueError(f"{path}:{line_number}: duplicate semantic anchor {anchor}")
            seen_anchors.add(anchor)
            semantic_interval = base.get("language_action_interval")
            try:
                start = int(semantic_interval["start_frame_inclusive"])
                end = int(semantic_interval["end_frame_inclusive"])
                error = None if start <= end else "annotation_interval_invalid"
            except (KeyError, TypeError, ValueError):
                start = end = None
                error = "annotation_interval_missing"
            intervals.append(
                _AnnotationInterval(anchor, start, end, base.get("sample_id"), error)
            )
    intervals.sort(key=lambda item: item.anchor)
    return intervals


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _interval_metadata_identity(
    dataset_root: Path,
    annotation_root: Path | None,
) -> tuple[list[dict[str, Any]], str]:
    files: list[tuple[str, Path]] = []
    for relative in INTERVAL_DATASET_IDENTITY_PATHS:
        path = dataset_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"interval audit identity file is missing: {path}")
        files.append((f"dataset/{relative}", path))
    if annotation_root is not None:
        for pattern in ("episode_*/shard_*/rank_*/status.json", "episode_*/shard_*/rank_*/training_samples*.jsonl"):
            for path in sorted(annotation_root.glob(pattern)):
                files.append(
                    (f"annotation/{path.relative_to(annotation_root).as_posix()}", path)
                )
    records = [
        {
            "relative_path": relative,
            "size_bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for relative, path in sorted(files)
    ]
    identity = hashlib.sha256(
        json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return records, identity


def audit_future_difference_intervals(
    dataset_root: str | Path,
    *,
    action_horizon: int,
    annotation_root: str | Path | None = None,
) -> dict[str, Any]:
    """Compare nominal action chunks with materialized annotation intervals."""
    if action_horizon < 1:
        raise ValueError("action_horizon must be positive")
    root = Path(dataset_root).resolve()
    steps = _load_steps(root)
    mapping = _load_episode_mapping(root)
    resolved_annotation_root, annotation_root_source = _resolve_annotation_root(
        root, annotation_root
    )
    try:
        info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    except Exception as error:
        raise ValueError(f"failed to read interval audit info.json: {error}") from error
    codebase_version = info.get("codebase_version")
    if not isinstance(codebase_version, str) or not codebase_version:
        raise ValueError("interval audit info.json requires codebase_version")
    identity_files, audit_identity = _interval_metadata_identity(
        root, resolved_annotation_root
    )

    classifications = Counter({name: 0 for name in INTERVAL_CLASSIFICATIONS})
    unavailable_reasons: Counter[str] = Counter()
    unavailable_examples = []
    annotation_lengths = []
    intersection_lengths = []
    episode_cache: dict[int, list[_AnnotationInterval] | None] = {}

    for sample_index, (new_episode, frame) in enumerate(steps):
        reason = None
        mapping_row = mapping.get(new_episode)
        interval = None
        if mapping_row is None:
            reason = "episode_mapping_missing"
        elif resolved_annotation_root is None:
            reason = "annotation_root_missing"
        else:
            old_episode = int(mapping_row["old_episode_index"])
            if old_episode not in episode_cache:
                path = _active_training_samples(resolved_annotation_root, old_episode)
                episode_cache[old_episode] = (
                    _load_annotation_intervals(path) if path is not None else None
                )
            episode_intervals = episode_cache[old_episode]
            if episode_intervals is None:
                reason = "annotation_episode_missing"
            elif not episode_intervals:
                reason = "annotation_episode_empty"
            else:
                anchors = [candidate.anchor for candidate in episode_intervals]
                position = bisect.bisect_right(anchors, frame) - 1
                if position < 0:
                    reason = "no_preceding_annotation"
                else:
                    interval = episode_intervals[position]
                    reason = interval.error

        if reason is not None:
            unavailable_reasons[reason] += 1
            if len(unavailable_examples) < 20:
                unavailable_examples.append(
                    {
                        "sample_id": f"episode={new_episode} frame={frame} sample={sample_index}",
                        "reason": reason,
                    }
                )
            continue

        assert interval is not None and interval.start is not None and interval.end is not None
        overlap = classify_interval_overlap(
            frame, frame + action_horizon - 1, interval.start, interval.end
        )
        classifications[overlap.classification] += 1
        annotation_lengths.append(overlap.annotation_length)
        intersection_lengths.append(overlap.intersection_length)

    available_count = sum(classifications.values())
    unavailable_count = sum(unavailable_reasons.values())
    mismatch_count = available_count - classifications["exact"]
    return {
        "dataset_root": str(root),
        "annotation_root": (
            str(resolved_annotation_root) if resolved_annotation_root is not None else None
        ),
        "annotation_root_source": annotation_root_source,
        "codebase_version": codebase_version,
        "metadata_identity_files": identity_files,
        "audit_identity_sha256": audit_identity,
        "scope": "all_published_frames_without_training_eligible_filter",
        "sample_count": len(steps),
        "action_horizon": action_horizon,
        "chunk_interval": "[frame, frame + action_horizon - 1] (closed, nominal before tail padding)",
        "classification_definitions": {
            "exact": "chunk and annotation have identical closed endpoints",
            "full": "the full chunk is covered by the annotation interval, but endpoints differ",
            "partial": "closed intervals overlap, but the full chunk is not covered",
            "none": "closed intervals do not overlap",
        },
        "available_count": available_count,
        "unavailable_count": unavailable_count,
        "classification_counts": dict(classifications),
        "mismatch_count": mismatch_count,
        "mismatch_ratio": mismatch_count / available_count if available_count else None,
        "unavailable_reasons": dict(sorted(unavailable_reasons.items())),
        "unavailable_examples": unavailable_examples,
        "annotation_interval_length": _distribution(annotation_lengths),
        "intersection_length": _distribution(intersection_lengths),
        "eligibility_computed": False,
    }
