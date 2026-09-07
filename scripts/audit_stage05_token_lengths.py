#!/usr/bin/env python3
"""Audit exact two-view Stage05 sequence lengths without decoding every image."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import sys
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.dataset_adapters import canonicalize_future_difference_target
from utils.stage05_dataset import (
    STAGE05_CAMERA_LABELS,
    build_stage05_message,
    resolve_stage05_vision_contract,
)
from utils.stage05_sidecar import (
    SIDECAR_FORMAT_VERSION,
    canonical_json_hash,
    generator_identity,
    sha256_file,
)
from utils.training_tokenization import (
    _find_last_subsequence,
    assistant_termination_token_ids,
    encode_target_text_tokens,
    measure_assistant_response_boundary,
)


TOKEN_AUDIT_FORMAT_VERSION = 2
STAGE05_TRUSTED_SPEC_PATH = ROOT / "configs/stage05_four_dataset_experiment.json"
DATASET_KEYS_TO_ENTRIES = OrderedDict(
    (
        ("droid", "stage05_droid_mixed"),
        ("household", "stage05_household_mixed"),
        ("tabletop", "stage05_tabletop_mixed"),
        ("rh20t", "stage05_rh20t_mixed"),
    )
)
TOKENIZATION_IMPLEMENTATION_PATHS = (
    "dataset2feature.yaml",
    "utils/dataset_adapters.py",
    "utils/stage05_dataset.py",
    "utils/stage05_sidecar.py",
    "utils/training_tokenization.py",
    "scripts/audit_stage05_token_lengths.py",
)
PROCESSOR_IDENTITY_FILENAMES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "chat_template.json",
    "chat_template.jinja",
    "preprocessor_config.json",
    "processor_config.json",
    "video_preprocessor_config.json",
    "config.json",
    "configuration.json",
    "vocab.json",
    "merges.txt",
)
REQUIRED_PROCESSOR_FILES = {
    "tokenizer.json",
    "tokenizer_config.json",
    "preprocessor_config.json",
    "config.json",
}


def _audit_error(message: str) -> ValueError:
    return ValueError(
        f"Stage05 token audit is stale: {message}; rerun the complete token audit"
    )


def _content_hash(value: dict[str, Any]) -> str:
    return canonical_json_hash(
        {key: item for key, item in value.items() if key != "content_hash"}
    )


def _file_records(paths: list[Path], *, base: Path | None = None) -> list[dict[str, Any]]:
    records = []
    for path in sorted(paths, key=lambda item: str(item)):
        resolved = path.resolve()
        recorded_path = (
            path.absolute().relative_to(base.resolve()).as_posix()
            if base is not None
            else str(resolved)
        )
        records.append(
            {
                "path": recorded_path,
                "sha256": sha256_file(resolved),
                "size": resolved.stat().st_size,
            }
        )
    return records


def _repository_implementation_identity(repository_root: Path = ROOT) -> dict[str, Any]:
    paths = [repository_root / relative for relative in TOKENIZATION_IMPLEMENTATION_PATHS]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise _audit_error(f"missing implementation dependencies: {missing}")
    records = _file_records(paths, base=repository_root)
    return {"dependencies": records, "sha256": canonical_json_hash(records)}


def _validate_ar_compatibility_identity(report_identity: dict[str, Any], repository_root: Path) -> bool:
    """Accept only the complete, reviewed pair of historical/current identities."""
    from utils.stage05_compatibility import verified_legacy_identity
    return verified_legacy_identity(report_identity, _repository_implementation_identity(repository_root), "ar_tokenization")


def _processor_files(processor_path: Path) -> list[dict[str, Any]]:
    paths = [
        processor_path / name
        for name in PROCESSOR_IDENTITY_FILENAMES
        if (processor_path / name).is_file()
    ]
    present = {path.name for path in paths}
    missing = sorted(REQUIRED_PROCESSOR_FILES - present)
    if missing:
        raise _audit_error(f"processor identity files are missing: {missing}")
    if not ({"chat_template.json", "chat_template.jinja"} & present):
        raise _audit_error("processor has no standalone chat template file")
    return _file_records(paths, base=processor_path)


def _package_versions() -> dict[str, str]:
    result = {}
    for distribution in ("transformers", "qwen-vl-utils", "tokenizers"):
        try:
            result[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError as error:
            raise _audit_error(f"required package is missing: {distribution}") from error
    return result


def _runtime_source_files(processor) -> list[dict[str, Any]]:
    import inspect
    import transformers.processing_utils
    from utils import dataset_adapters

    objects = (
        processor.__class__,
        processor.tokenizer.__class__,
        processor.image_processor.__class__,
        processor.__call__,
        transformers.processing_utils.ProcessorMixin.apply_chat_template,
        dataset_adapters._qwen_process_vision_info,
    )
    paths = []
    for value in objects:
        if value is None:
            continue
        source = inspect.getsourcefile(value)
        if source is not None and Path(source).is_file():
            paths.append(Path(source))
    unique = list(dict.fromkeys(path.resolve() for path in paths))
    return _file_records(unique)


def _processor_runtime_identity(processor, processor_path: Path) -> dict[str, Any]:
    image_processor = processor.image_processor
    tokenizer = processor.tokenizer
    visual_parameters = {
        key: image_processor.to_dict().get(key)
        for key in (
            "do_resize",
            "size",
            "resample",
            "patch_size",
            "temporal_patch_size",
            "merge_size",
            "do_rescale",
            "rescale_factor",
            "do_normalize",
            "image_mean",
            "image_std",
        )
    }
    return {
        "audited_processor_path": str(processor_path.resolve()),
        "processor_class": f"{type(processor).__module__}.{type(processor).__name__}",
        "tokenizer_class": f"{type(tokenizer).__module__}.{type(tokenizer).__name__}",
        "image_processor_class": (
            f"{type(image_processor).__module__}.{type(image_processor).__name__}"
        ),
        "package_versions": _package_versions(),
        "model_files": _processor_files(processor_path),
        "runtime_source_files": _runtime_source_files(processor),
        "tokenizer_parameters": {
            "model_max_length": int(tokenizer.model_max_length),
            "padding_side": tokenizer.padding_side,
            "truncation_side": tokenizer.truncation_side,
            "special_tokens_map": tokenizer.special_tokens_map,
        },
        "visual_token_parameters": visual_parameters,
    }


def _read_registry(repository_root: Path = ROOT) -> dict[str, Any]:
    value = yaml.safe_load((repository_root / "dataset2feature.yaml").read_text())
    if not isinstance(value, dict):
        raise _audit_error("dataset registry is not a mapping")
    return value


def _read_sidecar_identity(
    *,
    key: str,
    entry_name: str,
    entry: dict[str, Any],
    sidecar_path: Path,
    repository_root: Path = ROOT,
) -> dict[str, Any]:
    manifest_path = sidecar_path / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as error:
        raise _audit_error(f"cannot read {key} sidecar manifest {manifest_path}") from error
    if manifest.get("sidecar_format_version") != SIDECAR_FORMAT_VERSION:
        raise _audit_error(f"{key} sidecar format version changed")
    if manifest.get("content_hash") != _content_hash(manifest):
        raise _audit_error(f"{key} sidecar manifest content hash is invalid")
    if manifest.get("generator_identity") != generator_identity(
        repository_root=repository_root, build_joint=False
    ):
        from utils.stage05_compatibility import verified_legacy_identity
        if not verified_legacy_identity(manifest.get("generator_identity"), generator_identity(repository_root=repository_root, build_joint=False), "ar_generator"):
            raise _audit_error(f"{key} AR sidecar generator identity changed")
    eligible_path = sidecar_path / "ar_indices.npy"
    eligible_hash = sha256_file(eligible_path)
    if manifest.get("files", {}).get("ar_indices.npy") != eligible_hash:
        raise _audit_error(f"{key} eligible index hash differs from its sidecar manifest")
    eligible = np.load(eligible_path, mmap_mode="r", allow_pickle=False)
    eligible_count = int(manifest.get("counts", {}).get("ar_eligible_frames", -1))
    if eligible.dtype != np.uint32 or eligible.ndim != 1 or len(eligible) != eligible_count:
        raise _audit_error(f"{key} eligible index shape/count is invalid")
    stats_key = str(entry.get("stats_key") or "")
    if manifest.get("dataset_id") != stats_key:
        raise _audit_error(f"{key} sidecar stats key differs from the registry")
    return {
        "dataset_key": key,
        "dataset_entry": entry_name,
        "stats_key": stats_key,
        "dataset_path": str(Path(entry["dataset_path"]).resolve()),
        "sidecar_path": str(sidecar_path.resolve()),
        "sidecar_format_version": manifest["sidecar_format_version"],
        "manifest_content_hash": manifest["content_hash"],
        "generator_identity": manifest["generator_identity"],
        "eligible_index_sha256": eligible_hash,
        "eligible_count": eligible_count,
    }


def _data_identities(
    sidecar_root: Path | None = None, *, repository_root: Path = ROOT
) -> list[dict[str, Any]]:
    registry = _read_registry(repository_root)
    identities = []
    for key, entry_name in DATASET_KEYS_TO_ENTRIES.items():
        entry = registry.get(entry_name)
        if not isinstance(entry, dict):
            raise _audit_error(f"registry entry is missing: {entry_name}")
        configured = Path(entry["ar_sidecar_path"]).resolve()
        selected = (sidecar_root / "ar" / key).resolve() if sidecar_root else configured
        if selected != configured:
            raise _audit_error(
                f"{entry_name} audit sidecar differs from the current registry"
            )
        identities.append(
            _read_sidecar_identity(
                key=key,
                entry_name=entry_name,
                entry=entry,
                sidecar_path=selected,
                repository_root=repository_root,
            )
        )
    return identities


def _compare_identity(expected: Any, actual: Any, description: str) -> None:
    if expected != actual:
        raise _audit_error(f"{description} identity changed")


def _validate_trusted_spec(
    report_path: Path, report: dict[str, Any], trusted_spec_path: Path
) -> None:
    try:
        spec = json.loads(trusted_spec_path.read_text(encoding="utf-8"))
    except Exception as error:
        raise _audit_error(f"cannot read trusted Stage05 experiment spec {trusted_spec_path}") from error
    if not isinstance(spec, dict) or spec.get("schema_version") != 1:
        raise _audit_error("trusted Stage05 experiment spec schema version is unsupported")
    token_spec = spec.get("token_audit")
    required = {
        "format_version",
        "required_max_length",
        "report_content_hash",
        "report_file_sha256",
        "implementation_identity_sha256",
    }
    if not isinstance(token_spec, dict) or not required.issubset(token_spec):
        raise _audit_error("trusted Stage05 experiment spec is incomplete")
    if token_spec["format_version"] != TOKEN_AUDIT_FORMAT_VERSION:
        raise _audit_error("trusted token audit format version is unsupported")
    if sha256_file(report_path) != token_spec["report_file_sha256"]:
        raise _audit_error("token audit report file hash differs from trusted spec")
    if report.get("content_hash") != token_spec["report_content_hash"]:
        raise _audit_error("token audit report content hash differs from trusted spec")
    if report.get("required_max_length") != token_spec["required_max_length"]:
        raise _audit_error("token audit required_max_length differs from trusted spec")
    implementation = report.get("implementation_identity")
    if not isinstance(implementation, dict) or implementation.get("sha256") != token_spec[
        "implementation_identity_sha256"
    ]:
        raise _audit_error("token audit implementation identity differs from trusted spec")


def validate_token_audit(
    report_path: Path | str,
    *,
    processor_path: Path | str,
    max_length: int,
    repository_root: Path = ROOT,
    trusted_spec_path: Path | str | None = None,
) -> int:
    report_path = Path(report_path).resolve()
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except Exception as error:
        raise _audit_error(f"cannot read report {report_path}") from error
    if report.get("content_hash") != _content_hash(report):
        raise _audit_error("report content hash mismatch")
    if report.get("token_audit_format_version") != TOKEN_AUDIT_FORMAT_VERSION:
        raise _audit_error(
            f"format version must be {TOKEN_AUDIT_FORMAT_VERSION}"
        )
    if trusted_spec_path is not None:
        _validate_trusted_spec(report_path, report, Path(trusted_spec_path).resolve())
    required_sections = {
        "data_identity",
        "processor_identity",
        "implementation_identity",
        "vision_contract",
        "required_max_length",
    }
    missing = sorted(required_sections - set(report))
    if missing:
        raise _audit_error(f"report identity fields are missing: {missing}")

    current_data = _data_identities(repository_root=repository_root)
    _compare_identity(report["data_identity"], current_data, "Stage05 AR sidecar")

    processor_identity = report["processor_identity"]
    if not isinstance(processor_identity, dict):
        raise _audit_error("processor identity is not an object")
    required_processor_fields = {
        "audited_processor_path",
        "processor_class",
        "tokenizer_class",
        "image_processor_class",
        "package_versions",
        "model_files",
        "runtime_source_files",
        "tokenizer_parameters",
        "visual_token_parameters",
    }
    missing_processor_fields = sorted(
        required_processor_fields - set(processor_identity)
    )
    if missing_processor_fields:
        raise _audit_error(
            f"processor identity fields are missing: {missing_processor_fields}"
        )
    for class_field in (
        "processor_class",
        "tokenizer_class",
        "image_processor_class",
    ):
        if not isinstance(processor_identity[class_field], str) or not processor_identity[
            class_field
        ]:
            raise _audit_error(f"processor identity {class_field} is invalid")
    if not processor_identity["runtime_source_files"]:
        raise _audit_error("processor runtime source file identity is empty")
    current_processor_path = Path(processor_path).resolve()
    _compare_identity(
        processor_identity.get("model_files"),
        _processor_files(current_processor_path),
        "processor/tokenizer files",
    )
    _compare_identity(
        processor_identity.get("package_versions"),
        _package_versions(),
        "processor package versions",
    )
    current_runtime_files = []
    for record in processor_identity.get("runtime_source_files", []):
        if not isinstance(record, dict) or not isinstance(record.get("path"), str):
            raise _audit_error("runtime source file identity is incomplete")
        path = Path(record["path"])
        if not path.is_file():
            raise _audit_error(f"runtime source file is missing: {path}")
        current_runtime_files.append(
            {"path": str(path.resolve()), "sha256": sha256_file(path), "size": path.stat().st_size}
        )
    _compare_identity(
        processor_identity.get("runtime_source_files"),
        current_runtime_files,
        "processor runtime source files",
    )
    try:
        _compare_identity(report["implementation_identity"],
                          _repository_implementation_identity(repository_root),
                          "production tokenization implementation")
    except ValueError:
        # Explicit legacy AR contract: only the auxiliary registration files
        # may differ; tokenization source remains byte-for-byte verified.
        if not _validate_ar_compatibility_identity(report["implementation_identity"], repository_root):
            raise

    vision = report["vision_contract"]
    if (
        not isinstance(vision, dict)
        or vision.get("camera_order") != list(STAGE05_CAMERA_LABELS)
        or vision.get("valid_image_counts") != [1, 2]
        or vision.get("resize", {}).get("height") != 224
        or vision.get("resize", {}).get("width") != 224
        or set(vision.get("processor_measured_visual_tokens", {})) != {"1", "2"}
    ):
        raise _audit_error("main+wrist visual contract is not the current Stage05 contract")
    required = report["required_max_length"]
    if isinstance(required, bool) or not isinstance(required, int) or required < 1:
        raise _audit_error("required_max_length is invalid")
    if isinstance(max_length, bool) or not isinstance(max_length, int) or max_length < required:
        raise ValueError(
            f"ZR0_MAX_LENGTH={max_length} is below the audited Stage05 minimum {required}"
        )
    return required


def _distribution(histogram: Counter[int]) -> dict[str, float | int | None]:
    count = sum(histogram.values())
    if not count:
        return {key: None for key in ("min", "max", "mean", "p50", "p90", "p95", "p99", "p99.9")}
    ordered = sorted(histogram.items())

    def percentile(percent: float) -> float:
        position = (count - 1) * percent / 100.0

        def value_at(rank: int) -> int:
            cumulative = 0
            for value, frequency in ordered:
                cumulative += frequency
                if rank < cumulative:
                    return value
            raise AssertionError("histogram rank is out of bounds")

        lower, upper = math.floor(position), math.ceil(position)
        low, high = value_at(lower), value_at(upper)
        return float(low + (high - low) * (position - lower))

    return {
        "min": ordered[0][0],
        "max": ordered[-1][0],
        "mean": sum(value * frequency for value, frequency in ordered) / count,
        "p50": percentile(50),
        "p90": percentile(90),
        "p95": percentile(95),
        "p99": percentile(99),
        "p99.9": percentile(99.9),
    }


class ExactLengthMeasurer:
    def __init__(self, processor):
        self.processor = processor
        self.blank = Image.new("RGB", (224, 224))
        self.vision = resolve_stage05_vision_contract(processor)
        self.visual_tokens = {
            int(count): int(values["total_visual_tokens"])
            for count, values in self.vision["processor_measured_visual_tokens"].items()
        }
        self.termination_tokens = len(assistant_termination_token_ids(processor))
        self.layouts: dict[tuple[str, int], int] = {}
        self.target_lengths: dict[str, int] = {}
        self._validate_fast_projection()

    def prime_targets(self, raw_targets, *, batch_size: int = 4096) -> None:
        unknown = list(
            dict.fromkeys(
                target
                for target in raw_targets
                if isinstance(target, str)
                and target.strip()
                and target not in self.target_lengths
            )
        )
        for start in range(0, len(unknown), batch_size):
            raw_batch = unknown[start : start + batch_size]
            canonical = [
                canonicalize_future_difference_target(target, "token audit")
                for target in raw_batch
            ]
            encoded = self.processor.tokenizer(
                canonical,
                add_special_tokens=False,
                padding=False,
                truncation=False,
                return_length=True,
            )
            lengths = encoded.get("length")
            if lengths is None:
                lengths = [len(ids) for ids in encoded["input_ids"]]
            for target, length in zip(raw_batch, lengths):
                if int(length) < 1:
                    raise ValueError("valid Stage05 target tokenized to zero tokens")
                self.target_lengths[target] = int(length)

    def _rendered_context(self, task: str, image_count: int) -> int:
        marker = "__ZR0_STAGE05_LENGTH_BOUNDARY__"
        images = [
            (label, self.blank) for label in STAGE05_CAMERA_LABELS[:image_count]
        ]
        rendered = self.processor.apply_chat_template(
            build_stage05_message(task, images, marker),
            tokenize=False,
            add_generation_prompt=False,
        )
        rendered_ids = np.asarray(
            encode_target_text_tokens(self.processor.tokenizer, rendered), dtype=np.int64
        )
        marker_ids = np.asarray(
            encode_target_text_tokens(self.processor.tokenizer, marker), dtype=np.int64
        )
        start = _find_last_subsequence(
            torch.from_numpy(rendered_ids),
            torch.from_numpy(marker_ids),
        )
        if start < 0:
            raise ValueError("target boundary marker is missing from rendered Stage05 chat")
        # Text rendering contains one image-pad token per view. The real processor
        # expands each placeholder to its measured visual-token count.
        return int(start + self.visual_tokens[image_count] - image_count)

    def _validate_fast_projection(self) -> None:
        marker = "__ZR0_STAGE05_LENGTH_BOUNDARY__"
        for image_count in (1, 2):
            images = [
                (label, self.blank) for label in STAGE05_CAMERA_LABELS[:image_count]
            ]
            _, boundary = measure_assistant_response_boundary(
                build_stage05_message("projection validation", images, marker),
                self.processor,
                image_inputs=[self.blank] * image_count,
                video_inputs=None,
                target=marker,
            )
            projected = self._rendered_context("projection validation", image_count)
            if projected != int(boundary["assistant_start"]):
                raise ValueError(
                    f"fast token projection differs for {image_count} views: "
                    f"{projected} != {int(boundary['assistant_start'])}"
                )

    def measure(self, task: str, raw_target: str, image_count: int) -> tuple[int, int, int]:
        layout_key = (task, image_count)
        if layout_key not in self.layouts:
            self.layouts[layout_key] = self._rendered_context(task, image_count)
        if raw_target not in self.target_lengths:
            self.prime_targets([raw_target])
        context = self.layouts[layout_key]
        target = self.target_lengths[raw_target]
        return context, target, context + target + self.termination_tokens


def _source_episodes(root: Path) -> dict[int, dict]:
    result = {}
    for path in sorted((root / "meta" / "episodes").glob("**/*.parquet")):
        columns = ["episode_index", "data/chunk_index", "data/file_index"]
        for row in pq.read_table(path, columns=columns).to_pylist():
            result[int(row["episode_index"])] = row
    return result


def _audit_dataset(name: str, root: Path, sidecar: Path, measurer: ExactLengthMeasurer):
    layouts_before = len(measurer.layouts)
    manifest = json.loads((sidecar / "manifest.json").read_text(encoding="utf-8"))
    eligible = np.load(sidecar / "ar_indices.npy", mmap_mode="r", allow_pickle=False)
    packed_validity = np.load(sidecar / "validity_packed.npy", mmap_mode="r", allow_pickle=False)
    episode_records = pq.read_table(sidecar / "episodes.parquet").to_pylist()
    eligible_ranges = {}
    eligible_cursor = 0
    for record in episode_records:
        count = int(record["ar_eligible_frames"])
        start, stop = int(record["dataset_from_index"]), int(record["dataset_to_index"])
        episode_eligible = eligible[eligible_cursor : eligible_cursor + count]
        if count and not (
            int(episode_eligible[0]) >= start and int(episode_eligible[-1]) < stop
        ):
            raise ValueError(f"{name}: AR indices cross episode {record['episode_index']} boundary")
        eligible_ranges[int(record["episode_index"])] = episode_eligible
        eligible_cursor += count
    if eligible_cursor != len(eligible):
        raise ValueError(f"{name}: episode AR counts do not cover the sidecar index")
    source_episodes = _source_episodes(root)
    by_file: dict[Path, list[dict]] = {}
    for record in episode_records:
        source = source_episodes[int(record["episode_index"])]
        path = root / "data" / f"chunk-{int(source['data/chunk_index']):03d}" / f"file-{int(source['data/file_index']):03d}.parquet"
        by_file.setdefault(path, []).append(record)

    histograms = {key: Counter() for key in ("context", "target", "total")}
    image_counts = Counter()
    tokenized_targets_within_files = 0
    audited = 0
    maximum = None
    ordered_files = sorted(by_file.items(), key=lambda item: str(item[0]))
    for file_number, (path, records) in enumerate(ordered_files, start=1):
        print(f"[{name}] token parquet {file_number}/{len(ordered_files)}", file=sys.stderr, flush=True)
        rows = pq.read_table(path, columns=["episode_index", "index", "train_data"]).to_pylist()
        grouped: dict[int, list[dict]] = {}
        for row in rows:
            grouped.setdefault(int(row["episode_index"]), []).append(row)
        prepared = {}
        for record in records:
            episode = int(record["episode_index"])
            episode_rows = sorted(grouped[episode], key=lambda row: int(row["index"]))
            start = int(record["dataset_from_index"])
            episode_eligible = eligible_ranges[episode]
            prepared[episode] = (episode_rows, start, episode_eligible)
        # Raw Stage05 targets are often frame-specific. Bound memory to one
        # Parquet file instead of retaining millions of strings for the audit.
        measurer.target_lengths.clear()
        measurer.prime_targets(
            episode_rows[int(global_index) - start]["train_data"]
            for episode_rows, start, episode_eligible in prepared.values()
            for global_index in episode_eligible
        )
        tokenized_targets_within_files += len(measurer.target_lengths)
        for record in records:
            episode = int(record["episode_index"])
            episode_rows, start, episode_eligible = prepared[episode]
            for global_index in episode_eligible:
                global_index = int(global_index)
                row = episode_rows[global_index - start]
                packed = int(packed_validity[global_index // 8, 1])
                wrist_valid = bool((packed >> (7 - global_index % 8)) & 1)
                image_count = 2 if wrist_valid else 1
                context, target, total = measurer.measure(
                    str(record["task"]), row["train_data"], image_count
                )
                histograms["context"][context] += 1
                histograms["target"][target] += 1
                histograms["total"][total] += 1
                image_counts[image_count] += 1
                audited += 1
                if maximum is None or total > maximum["tokens"]:
                    maximum = {
                        "tokens": total,
                        "episode_index": episode,
                        "global_index": global_index,
                        "image_count": image_count,
                    }
    if audited != int(manifest["counts"]["ar_eligible_frames"]):
        raise ValueError(f"{name}: audited {audited} != sidecar eligible count")
    return {
        "ar_eligible_frames": audited,
        "image_count_frames": {str(key): value for key, value in sorted(image_counts.items())},
        "context_tokens": _distribution(histograms["context"]),
        "target_content_tokens": _distribution(histograms["target"]),
        "full_sequence_tokens": _distribution(histograms["total"]),
        "maximum_sample": maximum,
        "tokenized_unique_targets_within_parquet_files": tokenized_targets_within_files,
        "new_unique_task_image_layouts": len(measurer.layouts) - layouts_before,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sidecar-root", type=Path)
    parser.add_argument("--processor-path", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--max-length", type=int)
    parser.add_argument(
        "--trusted-spec",
        type=Path,
        default=STAGE05_TRUSTED_SPEC_PATH,
        help=(
            "Versioned external trust specification. Omit this option to use the "
            "checked-in Stage05 experiment specification."
        ),
    )
    args = parser.parse_args()
    if args.validate_only:
        if args.report is None or args.max_length is None:
            parser.error("--validate-only requires --report and --max-length")
        required = validate_token_audit(
            args.report,
            processor_path=args.processor_path,
            max_length=args.max_length,
            trusted_spec_path=args.trusted_spec,
        )
        print(
            json.dumps(
                {
                    "status": "valid",
                    "report": str(args.report.resolve()),
                    "processor_path": str(args.processor_path.resolve()),
                    "required_max_length": required,
                },
                sort_keys=True,
            )
        )
        return
    if args.sidecar_root is None or args.output is None:
        parser.error("full audit requires --sidecar-root and --output")
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(args.processor_path)
    measurer = ExactLengthMeasurer(processor)
    data_identity = _data_identities(args.sidecar_root)
    registry = _read_registry()
    report = {
        "token_audit_format_version": TOKEN_AUDIT_FORMAT_VERSION,
        "method": "exact chat-template/tokenizer lengths with processor-validated measured visual expansion",
        "vision_contract": measurer.vision,
        "assistant_termination_tokens": measurer.termination_tokens,
        "data_identity": data_identity,
        "processor_identity": _processor_runtime_identity(
            processor, args.processor_path
        ),
        "implementation_identity": _repository_implementation_identity(),
        "datasets": OrderedDict(),
    }
    identities_by_key = {item["dataset_key"]: item for item in data_identity}
    for name, entry_name in DATASET_KEYS_TO_ENTRIES.items():
        entry = registry[entry_name]
        report["datasets"][name] = _audit_dataset(
            name,
            Path(entry["dataset_path"]),
            Path(identities_by_key[name]["sidecar_path"]),
            measurer,
        )
    report["minimum_zero_truncation_max_length"] = max(
        value["full_sequence_tokens"]["max"] for value in report["datasets"].values()
    )
    report["required_max_length"] = report["minimum_zero_truncation_max_length"]
    report["content_hash"] = _content_hash(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "max_length": report["minimum_zero_truncation_max_length"]}))


if __name__ == "__main__":
    main()
