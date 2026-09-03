"""Shared dataset schema and normalization resolution for training and policy use."""

from __future__ import annotations

import json
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from utils.normalization import min_max_denorm


SUPPORTED_DATASET_ADAPTERS = ("lerobot_v2", "lerobot_v3_future_difference")
SUPPORTED_LOSS_TYPES = ("action", "vlm", "vlm_and_action")


@dataclass(frozen=True)
class ObjectiveRequirements:
    requires_target: bool
    requires_action: bool
    requires_state: bool
    requires_stats: bool
    requires_fast_tokenizer: bool

    def dependency_names(self) -> list[str]:
        names = ["images", "task"]
        if self.requires_target:
            names.append("target_text")
        if self.requires_state:
            names.append("state")
        if self.requires_action:
            names.append("actions")
        if self.requires_stats:
            names.append("stats")
        if self.requires_fast_tokenizer:
            names.append("fast_tokenizer")
        return names


def resolve_objective_requirements(
    loss_type: str,
    *,
    adapter: str,
    target_text_field: str | None,
    dataset_type: str = "vla",
    dataset_entry: str | None = None,
) -> ObjectiveRequirements:
    if loss_type not in SUPPORTED_LOSS_TYPES:
        raise ValueError(
            f"loss_type must be one of {list(SUPPORTED_LOSS_TYPES)}, got {loss_type!r}"
        )
    if dataset_type == "vlm":
        if loss_type == "action":
            entry = dataset_entry or "<unknown>"
            raise ValueError(
                f"dataset entry {entry!r} has dataset_type='vlm', which is "
                f"incompatible with loss_type='action'"
            )
        return ObjectiveRequirements(
            requires_target=True,
            requires_action=False,
            requires_state=False,
            requires_stats=False,
            requires_fast_tokenizer=False,
        )
    if dataset_type != "vla":
        raise ValueError(
            f"dataset entry {dataset_entry or '<unknown>'!r} has unsupported "
            f"dataset_type={dataset_type!r}"
        )

    needs_target = loss_type in {"vlm", "vlm_and_action"}
    needs_action = loss_type in {"action", "vlm_and_action"}
    needs_fast = (
        adapter == "lerobot_v2" and needs_target and target_text_field is None
    )
    if loss_type == "vlm" and needs_fast:
        raise ValueError(
            "lerobot_v2 action-independent AR-only requires an explicit "
            "target_text_field; the legacy target depends on FAST action tokens"
        )
    return ObjectiveRequirements(
        requires_target=needs_target,
        requires_action=needs_action,
        requires_state=needs_action,
        requires_stats=needs_action,
        requires_fast_tokenizer=needs_fast,
    )


@dataclass(frozen=True)
class ObservationContract:
    version: int
    window_size: int
    history_order: str
    history_stride: str


@dataclass(frozen=True)
class ResolvedDatasetSpec:
    dataset_entry: str
    dataset_path: str
    dataset_type: str
    adapter: str
    target_text_field: str | None
    camera_keys: tuple[str, ...]
    grounding_camera_keys: tuple[str, ...]
    state_key: str
    action_key: str
    state_dim: int
    action_dim: int
    action_horizon: int
    stats_path: str | None
    stats_key: str | None
    normalization: str
    normalization_stats: dict[str, Any] | None
    sample_ratio: float
    training_eligibility_exists: bool
    training_eligibility_used: bool
    data_version: str | None
    task_key: str = "task"
    stats_sha256: str | None = None
    training_eligibility_source: str = "not_available"
    observation_contract: ObservationContract = ObservationContract(
        version=1,
        window_size=1,
        history_order="single_current_frame",
        history_stride="not_applicable",
    )
    vision_input_contract: dict[str, Any] | None = None


def resolve_dataset_adapter_name(entry: dict[str, Any]) -> str:
    adapter = entry.get("dataset_adapter", "lerobot_v2")
    if adapter not in SUPPORTED_DATASET_ADAPTERS:
        raise ValueError(
            f"unknown dataset_adapter {adapter!r}; expected one of "
            f"{list(SUPPORTED_DATASET_ADAPTERS)}"
        )
    return adapter


def validate_v3_quantile_stats(
    raw_stats: Any,
    *,
    dataset_entry: str = "v3 dataset",
) -> dict[str, dict[str, np.ndarray]]:
    try:
        statistics = raw_stats["statistics"]
        source = {
            "observation.state": statistics["state"],
            "action": statistics["actions"],
        }
    except (KeyError, TypeError) as error:
        raise ValueError(
            f"{dataset_entry}: stats must contain statistics.state and statistics.actions"
        ) from error
    validated: dict[str, dict[str, np.ndarray]] = {}
    for output_key, key_stats in source.items():
        validated[output_key] = {}
        for quantile in ("q01", "q99"):
            try:
                values = np.asarray(key_stats[quantile], dtype=np.float32)
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"{dataset_entry}: stats {output_key}.{quantile} is missing or invalid"
                ) from error
            if values.shape != (7,) or not np.isfinite(values).all():
                raise ValueError(
                    f"{dataset_entry}: stats {output_key}.{quantile} must be 7 finite values"
                )
            validated[output_key][quantile] = values
        if not np.all(validated[output_key]["q99"] > validated[output_key]["q01"]):
            raise ValueError(
                f"{dataset_entry}: stats {output_key} requires q01 < q99 in every dimension"
            )
    return validated


def _positive_int(value: Any, field: str, dataset_entry: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{dataset_entry}: {field} must be a positive integer")
    return value


def _feature_dim(
    features: Any,
    key: str,
    field: str,
    dataset_entry: str,
) -> int:
    try:
        shape = features[key]["shape"]
    except (KeyError, TypeError) as error:
        raise ValueError(
            f"{dataset_entry}: {field} {key!r} is missing from meta/info.json features"
        ) from error
    if not isinstance(shape, (list, tuple)) or len(shape) != 1:
        raise ValueError(f"{dataset_entry}: {field} {key!r} must have a one-dimensional shape")
    return _positive_int(shape[0], f"{field} dimension", dataset_entry)


def _optional_feature_dim(features: Any, key: str) -> int:
    try:
        shape = features[key]["shape"]
    except (KeyError, TypeError):
        return 0
    if not isinstance(shape, (list, tuple)) or len(shape) != 1:
        return 0
    value = shape[0]
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return 0
    return value


def _sample_ratio(entry: dict[str, Any], dataset_entry: str) -> float:
    value = entry.get("sample_ratio", 1.0)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0 < value <= 1
    ):
        raise ValueError(f"{dataset_entry}: sample_ratio must be finite and in (0, 1]")
    return float(value)


def _resolve_v3_spec(
    dataset_entry: str,
    entry: dict[str, Any],
    action_horizon: int,
    window_size: int,
    requirements: ObjectiveRequirements,
) -> ResolvedDatasetSpec:
    if window_size != 1:
        raise ValueError(f"{dataset_entry}: v3 datasets require window_size=1")
    root = Path(entry.get("dataset_path", "")).expanduser().resolve()
    info_path = root / "meta/info.json"
    try:
        info = json.loads(info_path.read_text(encoding="utf-8"))
    except Exception as error:
        raise ValueError(f"{dataset_entry}: failed to read metadata field {info_path}: {error}") from error
    features = info.get("features")
    if not isinstance(features, dict):
        raise ValueError(f"{dataset_entry}: meta/info.json features must be an object")

    camera_keys = entry.get("camera_keys")
    if (
        not isinstance(camera_keys, (list, tuple))
        or len(camera_keys) != 3
        or not all(isinstance(key, str) and key for key in camera_keys)
    ):
        raise ValueError(f"{dataset_entry}: camera_keys must contain exactly three field names")
    for key in camera_keys:
        feature = features.get(key)
        if not isinstance(feature, dict) or feature.get("dtype") != "image":
            raise ValueError(f"{dataset_entry}: camera_keys field {key!r} is missing or not an image")

    state_key = entry.get("state_field", "state")
    action_key = entry.get("action_field", "actions")
    state_dim = (
        _feature_dim(features, state_key, "state_field", dataset_entry)
        if requirements.requires_state
        else _optional_feature_dim(features, state_key)
    )
    action_dim = (
        _feature_dim(features, action_key, "action_field", dataset_entry)
        if requirements.requires_action
        else _optional_feature_dim(features, action_key)
    )
    if requirements.requires_state and state_dim != 7:
        raise ValueError(
            f"{dataset_entry}: v3 state dimension must be 7, got {state_dim}"
        )
    if requirements.requires_action and action_dim != 7:
        raise ValueError(
            f"{dataset_entry}: v3 action dimension must be 7, got {action_dim}"
        )
    if requirements.requires_stats and entry.get("use_quantile", True) is not True:
        raise ValueError(f"{dataset_entry}: v3 use_quantile must be true")

    stats_path_value = entry.get("stats_path", "meta/stats_gr00t.json")
    if requirements.requires_stats and (
        not isinstance(stats_path_value, str) or not stats_path_value
    ):
        raise ValueError(f"{dataset_entry}: stats_path must be a non-empty string")
    stats: dict[str, Any] | None = None
    stats_sha256 = None
    if requirements.requires_stats:
        stats_path = Path(stats_path_value)
        resolved_stats_path = stats_path if stats_path.is_absolute() else root / stats_path
        try:
            stats_bytes = resolved_stats_path.read_bytes()
            raw_stats = json.loads(stats_bytes.decode("utf-8"))
        except Exception as error:
            raise ValueError(
                f"{dataset_entry}: failed to read stats_path {resolved_stats_path}: {error}"
            ) from error
        stats = validate_v3_quantile_stats(raw_stats, dataset_entry=dataset_entry)
        stats_sha256 = hashlib.sha256(stats_bytes).hexdigest()

    target_field = entry.get("target_text_field", "train_data")
    if requirements.requires_target and (
        not isinstance(target_field, str) or target_field not in features
    ):
        raise ValueError(
            f"{dataset_entry}: target_text_field {target_field!r} is missing from meta/info.json features"
        )
    eligibility_exists = "training_eligible" in features
    return ResolvedDatasetSpec(
        dataset_entry=dataset_entry,
        dataset_path=str(root),
        dataset_type=str(entry.get("dataset_type", "vla")),
        adapter="lerobot_v3_future_difference",
        target_text_field=target_field,
        camera_keys=tuple(camera_keys),
        grounding_camera_keys=(),
        state_key=state_key,
        action_key=action_key,
        state_dim=state_dim,
        action_dim=action_dim,
        action_horizon=action_horizon,
        stats_path=stats_path_value,
        stats_key="statistics.state/actions.q01/q99",
        normalization="quantile_min_max_q01_q99",
        normalization_stats=stats,
        sample_ratio=_sample_ratio(entry, dataset_entry),
        training_eligibility_exists=eligibility_exists,
        training_eligibility_used=False,
        data_version=info.get("codebase_version"),
        task_key="task",
        stats_sha256=stats_sha256,
        training_eligibility_source=(
            "meta/info.json:features.training_eligible"
            if eligibility_exists
            else "unavailable_in_release"
        ),
        observation_contract=ObservationContract(
            version=1,
            window_size=1,
            history_order="single_current_frame",
            history_stride="not_applicable",
        ),
    )


def _metadata_feature_dim(metadata: Any, key: str, fallback_stats_key: str) -> int:
    features = getattr(metadata, "features", None)
    if isinstance(features, dict) and key in features:
        feature = features[key]
        shape = feature.get("shape") if isinstance(feature, dict) else getattr(feature, "shape", None)
        if isinstance(shape, (list, tuple)) and len(shape) == 1:
            return int(shape[0])
    stats = getattr(metadata, "stats", {}).get(fallback_stats_key, {})
    for statistic in ("q01", "min", "mean"):
        if statistic in stats:
            return int(np.asarray(stats[statistic]).shape[-1])
    raise ValueError(f"metadata does not define dimension for {key}")


def _resolve_v2_spec(
    dataset_entry: str,
    entry: dict[str, Any],
    action_horizon: int,
    window_size: int,
    requirements: ObjectiveRequirements,
    metadata: Any,
) -> ResolvedDatasetSpec:
    if metadata is None:
        raise ValueError(f"{dataset_entry}: lerobot_v2 resolution requires dataset metadata")
    try:
        camera_keys = tuple(metadata.camera_keys)
        grounding_camera_keys = tuple(metadata.grounding_camera_keys or ())
        stats = metadata.stats if requirements.requires_stats else None
        metadata_features = getattr(metadata, "features", {})
        state_dim = (
            _metadata_feature_dim(
                metadata, "observation.state", "observation.state"
            )
            if requirements.requires_state
            else _optional_feature_dim(metadata_features, "observation.state")
        )
        action_dim = (
            _metadata_feature_dim(metadata, "action", "action")
            if requirements.requires_action
            else _optional_feature_dim(metadata_features, "action")
        )
    except Exception as error:
        raise ValueError(f"{dataset_entry}: invalid lerobot_v2 metadata: {error}") from error
    use_quantile = bool(entry.get("use_quantile", True))
    target_field = entry.get("target_text_field")
    if requirements.requires_target and target_field is not None:
        if not isinstance(target_field, str) or target_field not in metadata_features:
            raise ValueError(
                f"{dataset_entry}: target_text_field {target_field!r} is missing "
                "from lerobot_v2 metadata features"
            )
    eligibility_exists = (
        isinstance(metadata_features, dict)
        and "training_eligible" in metadata_features
    )
    stats_sha256 = None
    metadata_root = getattr(metadata, "root", None)
    if requirements.requires_stats and metadata_root is not None:
        stats_file = Path(metadata_root) / "meta/stats.json"
        if stats_file.is_file():
            stats_sha256 = hashlib.sha256(stats_file.read_bytes()).hexdigest()
    return ResolvedDatasetSpec(
        dataset_entry=dataset_entry,
        dataset_path=str(Path(entry["dataset_path"]).expanduser().resolve()),
        dataset_type=str(entry.get("dataset_type", "vla")),
        adapter="lerobot_v2",
        target_text_field=target_field,
        camera_keys=camera_keys,
        grounding_camera_keys=grounding_camera_keys,
        state_key="observation.state",
        action_key="action",
        state_dim=state_dim,
        action_dim=action_dim,
        action_horizon=action_horizon,
        stats_path=None,
        stats_key="dataset_meta.stats",
        normalization="quantile" if use_quantile else "min_max",
        normalization_stats=stats,
        sample_ratio=_sample_ratio(entry, dataset_entry),
        training_eligibility_exists=eligibility_exists,
        training_eligibility_used=False,
        data_version=getattr(metadata, "codebase_version", None),
        task_key="task",
        stats_sha256=stats_sha256,
        training_eligibility_source=(
            "metadata.features.training_eligible"
            if eligibility_exists
            else "unavailable_in_release"
        ),
        observation_contract=ObservationContract(
            version=1,
            window_size=window_size,
            history_order="frame_major_oldest_to_current",
            history_stride="previous_policy_execution_horizon",
        ),
    )


def resolve_dataset_spec(
    dataset_entry: str,
    entry: dict[str, Any],
    *,
    action_horizon: int,
    window_size: int = 1,
    require_action: bool | None = None,
    requirements: ObjectiveRequirements | None = None,
    v2_metadata: Any = None,
) -> ResolvedDatasetSpec:
    if not isinstance(dataset_entry, str) or not dataset_entry:
        raise ValueError("dataset_entry must be a non-empty string")
    action_horizon = _positive_int(action_horizon, "action_horizon", dataset_entry)
    window_size = _positive_int(window_size, "window_size", dataset_entry)
    try:
        adapter = resolve_dataset_adapter_name(entry)
    except ValueError as error:
        raise ValueError(f"{dataset_entry}: {error}") from error
    if requirements is None:
        if require_action is None:
            raise ValueError(
                f"{dataset_entry}: resolve_dataset_spec requires objective requirements"
            )
        requirements = resolve_objective_requirements(
            "vlm_and_action" if require_action else "vlm",
            adapter=adapter,
            target_text_field=entry.get("target_text_field"),
            dataset_type=str(entry.get("dataset_type", "vla")),
            dataset_entry=dataset_entry,
        )
    elif require_action is not None and require_action != requirements.requires_action:
        raise ValueError(
            f"{dataset_entry}: require_action conflicts with objective requirements"
        )
    if adapter == "lerobot_v3_future_difference":
        return _resolve_v3_spec(
            dataset_entry, entry, action_horizon, window_size, requirements
        )
    return _resolve_v2_spec(
        dataset_entry,
        entry,
        action_horizon,
        window_size,
        requirements,
        v2_metadata,
    )


def denormalize_actions(
    normalized_actions: torch.Tensor,
    spec: ResolvedDatasetSpec,
) -> torch.Tensor:
    if spec.normalization_stats is None:
        raise ValueError(f"{spec.dataset_entry}: normalization stats were not resolved")
    if not isinstance(normalized_actions, torch.Tensor) or normalized_actions.ndim < 1:
        raise ValueError(f"{spec.dataset_entry}: normalized actions must be a tensor")
    if normalized_actions.shape[-1] < spec.action_dim:
        raise ValueError(
            f"{spec.dataset_entry}: normalized action width {normalized_actions.shape[-1]} "
            f"is smaller than action dimension {spec.action_dim}"
        )
    values = normalized_actions[..., : spec.action_dim]
    if not torch.isfinite(values).all():
        raise ValueError(f"{spec.dataset_entry}: normalized actions contain NaN or Inf")
    use_quantile = spec.normalization == "quantile_min_max_q01_q99" or spec.normalization == "quantile"
    return min_max_denorm(values, spec.normalization_stats["action"], use_quantile)
