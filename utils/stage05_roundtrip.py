"""Round-trip diagnostics using the production normalization implementation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from utils.normalization import (
    NORMALIZED_VALUE_CLIP,
    min_max_denorm,
    min_max_norm,
    min_max_norm_unclipped,
)


@dataclass
class RoundTripAccumulator:
    dimensions: int = 7
    counts: np.ndarray = field(init=False)
    clipped: np.ndarray = field(init=False)
    pre_min: np.ndarray = field(init=False)
    pre_max: np.ndarray = field(init=False)
    error_sum: np.ndarray = field(init=False)
    error_values: list[list[np.ndarray]] = field(init=False)
    worst: list[dict[str, Any] | None] = field(init=False)

    def __post_init__(self) -> None:
        self.counts = np.zeros(self.dimensions, dtype=np.int64)
        self.clipped = np.zeros(self.dimensions, dtype=np.int64)
        self.pre_min = np.full(self.dimensions, np.inf, dtype=np.float64)
        self.pre_max = np.full(self.dimensions, -np.inf, dtype=np.float64)
        self.error_sum = np.zeros(self.dimensions, dtype=np.float64)
        self.error_values = [[] for _ in range(self.dimensions)]
        self.worst = [None for _ in range(self.dimensions)]

    def update(
        self,
        values,
        valid_mask,
        stats: dict[str, np.ndarray],
        identities: list[dict[str, Any]],
    ) -> None:
        values = torch.as_tensor(values, dtype=torch.float32)
        valid = torch.as_tensor(valid_mask, dtype=torch.bool)
        if values.ndim != 2 or values.shape[1] != self.dimensions:
            raise ValueError("round-trip values must have shape [N,7]")
        if valid.shape != values.shape or len(identities) != values.shape[0]:
            raise ValueError("round-trip mask/identities must align with values")
        pre = min_max_norm_unclipped(values, stats, True)
        normalized = min_max_norm(values, stats, True)
        restored = min_max_denorm(normalized, stats, True)
        errors = (restored - values).abs()
        low, high = NORMALIZED_VALUE_CLIP
        saturated = (pre < low) | (pre > high)

        for dimension in range(self.dimensions):
            selected = valid[:, dimension]
            if not selected.any():
                continue
            selected_pre = pre[selected, dimension].detach().cpu().numpy().astype(np.float64)
            selected_errors = errors[selected, dimension].detach().cpu().numpy().astype(np.float64)
            selected_saturation = saturated[selected, dimension].detach().cpu().numpy()
            selected_rows = (
                torch.nonzero(selected, as_tuple=False).flatten().detach().cpu().tolist()
            )
            self.counts[dimension] += len(selected_errors)
            self.clipped[dimension] += int(selected_saturation.sum())
            self.pre_min[dimension] = min(self.pre_min[dimension], float(selected_pre.min()))
            self.pre_max[dimension] = max(self.pre_max[dimension], float(selected_pre.max()))
            self.error_sum[dimension] += float(selected_errors.sum())
            self.error_values[dimension].append(selected_errors)
            worst_local = int(np.argmax(selected_errors))
            worst_error = float(selected_errors[worst_local])
            if self.worst[dimension] is None or worst_error > self.worst[dimension]["error"]:
                identity = dict(identities[selected_rows[worst_local]])
                identity.update({"dimension": dimension, "error": worst_error})
                self.worst[dimension] = identity

    def report(self) -> list[dict[str, Any]]:
        result = []
        for dimension in range(self.dimensions):
            count = int(self.counts[dimension])
            errors = (
                np.concatenate(self.error_values[dimension])
                if self.error_values[dimension]
                else np.empty(0, dtype=np.float64)
            )
            result.append(
                {
                    "dimension": dimension,
                    "valid_element_count": count,
                    "normalized_before_clip_min": (
                        float(self.pre_min[dimension]) if count else None
                    ),
                    "normalized_before_clip_max": (
                        float(self.pre_max[dimension]) if count else None
                    ),
                    "clipped_count": int(self.clipped[dimension]),
                    "clipped_ratio": float(self.clipped[dimension] / count) if count else None,
                    "denormalization_error_max": float(errors.max()) if count else None,
                    "denormalization_error_mean": (
                        float(self.error_sum[dimension] / count) if count else None
                    ),
                    "denormalization_error_p99": (
                        float(np.quantile(errors, 0.99)) if count else None
                    ),
                    "worst_sample": self.worst[dimension],
                }
            )
        return result


def mathematical_in_range_roundtrip(stats: dict[str, np.ndarray]) -> float:
    q01, q99 = stats["q01"], stats["q99"]
    values = torch.from_numpy(np.stack([q01, (q01 + q99) / 2.0, q99])).float()
    restored = min_max_denorm(min_max_norm(values, stats, True), stats, True)
    return float((restored - values).abs().max())
