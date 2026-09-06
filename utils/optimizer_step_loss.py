"""Optimizer-step loss normalization and detached metric aggregation."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from utils.training_tokenization import (
    TOKENIZATION_METRIC_SCHEMA,
    token_metric_validity_key,
)


AR_LOSS_TYPES = {"vlm", "vlm_and_action"}
FM_LOSS_TYPES = {"action", "vlm_and_action"}


@dataclass(frozen=True)
class GlobalSupervisionCounts:
    ar_tokens: torch.Tensor
    action_elements: torch.Tensor


def _batch_device(batch: dict) -> torch.device:
    for value in batch.values():
        if isinstance(value, torch.Tensor):
            return value.device
    return torch.device("cpu")


def global_supervision_counts(
    batches: list[dict],
    *,
    loss_type: str,
    accelerator,
) -> GlobalSupervisionCounts:
    if not batches:
        raise ValueError("optimizer-step accumulation window must not be empty")
    device = _batch_device(batches[0])
    local = torch.zeros(2, dtype=torch.float64, device=device)
    for batch in batches:
        if loss_type in AR_LOSS_TYPES:
            labels = batch.get("labels")
            if not isinstance(labels, torch.Tensor):
                raise ValueError("labels are required to count optimizer-step AR tokens")
            local[0] += labels[..., 1:].ne(-100).sum().to(torch.float64)
        if loss_type in FM_LOSS_TYPES:
            action_mask = batch.get("action_mask")
            if not isinstance(action_mask, torch.Tensor):
                raise ValueError(
                    "action_mask is required to count optimizer-step action elements"
                )
            local[1] += action_mask.to(dtype=torch.bool).sum().to(torch.float64)
    global_counts = accelerator.reduce(local, reduction="sum").detach()
    if loss_type == "vlm" and global_counts[0].item() <= 0:
        raise ValueError("optimizer-step AR supervision count is zero")
    strict_joint_fm = any(
        isinstance(batch.get("strict_joint_fm"), torch.Tensor)
        and bool(batch["strict_joint_fm"].to(dtype=torch.bool).any())
        for batch in batches
    )
    if (loss_type == "action" or strict_joint_fm) and global_counts[1].item() <= 0:
        raise ValueError("optimizer-step action supervision count is zero")
    return GlobalSupervisionCounts(
        ar_tokens=global_counts[0],
        action_elements=global_counts[1],
    )


def scaled_microbatch_loss(
    outputs: dict,
    *,
    counts: GlobalSupervisionCounts,
    loss_type: str,
    vlm_loss_weight: float,
    action_expert_loss_weight: float,
    gradient_accumulation_steps: int,
    data_parallel_world_size: int,
) -> torch.Tensor:
    scale = float(gradient_accumulation_steps * data_parallel_world_size)
    terms = []
    if loss_type in AR_LOSS_TYPES:
        ar_sum = outputs.get("ar_loss_sum")
        if not isinstance(ar_sum, torch.Tensor):
            raise ValueError("model outputs must contain differentiable ar_loss_sum")
        if counts.ar_tokens.item() > 0:
            terms.append(vlm_loss_weight * ar_sum / counts.ar_tokens.to(ar_sum.dtype))
        else:
            terms.append(ar_sum * 0.0)
    if loss_type in FM_LOSS_TYPES:
        fm_sum = outputs.get("flow_matching_loss_sum")
        if not isinstance(fm_sum, torch.Tensor):
            raise ValueError(
                "model outputs must contain differentiable flow_matching_loss_sum"
            )
        if counts.action_elements.item() > 0:
            terms.append(
                action_expert_loss_weight
                * fm_sum
                / counts.action_elements.to(fm_sum.dtype)
            )
        else:
            terms.append(fm_sum * 0.0)
    if not terms:
        raise ValueError(f"unsupported optimizer-step loss_type {loss_type!r}")
    return sum(terms[1:], terms[0]) * scale


class OptimizerStepMetricAccumulator:
    """Accumulate detached numerators/counts and reduce them once per step."""

    def __init__(
        self,
        *,
        loss_type: str,
        vlm_loss_weight: float,
        action_expert_loss_weight: float,
        collect_diagnostics: bool = False,
    ):
        self.loss_type = loss_type
        self.vlm_loss_weight = float(vlm_loss_weight)
        self.action_expert_loss_weight = float(action_expert_loss_weight)
        self.collect_diagnostics = bool(collect_diagnostics)
        self._loss_sums = {}
        self._loss_counts = {}
        self._token_stats = {}
        self._diagnostic_stats = {}

    @staticmethod
    def _detached_scalar(value, *, dtype=torch.float64):
        if not isinstance(value, torch.Tensor):
            raise ValueError("optimizer-step loss statistics must be tensors")
        return value.detach().to(dtype=dtype).reshape(())

    def update(self, outputs: dict, batch: dict) -> None:
        for branch, sum_key, count_key in (
            ("ar", "ar_loss_sum", "ar_loss_count"),
            ("fm", "flow_matching_loss_sum", "flow_matching_loss_count"),
        ):
            if sum_key not in outputs:
                continue
            value_sum = self._detached_scalar(outputs[sum_key])
            value_count = self._detached_scalar(outputs[count_key])
            self._loss_sums[branch] = self._loss_sums.get(
                branch, torch.zeros_like(value_sum)
            ) + value_sum
            self._loss_counts[branch] = self._loss_counts.get(
                branch, torch.zeros_like(value_count)
            ) + value_count

        for key in TOKENIZATION_METRIC_SCHEMA:
            value = batch.get(key)
            validity = batch.get(token_metric_validity_key(key))
            if not isinstance(value, torch.Tensor) or not isinstance(validity, torch.Tensor):
                continue
            values = value.detach().reshape(-1).to(torch.float64)
            valid = validity.detach().reshape(-1).to(device=values.device, dtype=torch.bool)
            if values.numel() != valid.numel():
                raise ValueError(f"token metric {key!r} and validity mask shapes differ")
            selected = values[valid]
            if key not in self._token_stats:
                self._token_stats[key] = {
                    "sum": torch.zeros((), dtype=torch.float64, device=values.device),
                    "count": torch.zeros((), dtype=torch.float64, device=values.device),
                    "min": torch.full((), float("inf"), dtype=torch.float64, device=values.device),
                    "max": torch.full((), float("-inf"), dtype=torch.float64, device=values.device),
                }
            stats = self._token_stats[key]
            if selected.numel():
                stats["sum"] += selected.sum()
                stats["count"] += selected.numel()
                stats["min"] = torch.minimum(stats["min"], selected.min())
                stats["max"] = torch.maximum(stats["max"], selected.max())

        if not self.collect_diagnostics:
            return

        retry_counts = batch.get("data_read_retry_count")
        if isinstance(retry_counts, torch.Tensor):
            retries = retry_counts.detach().reshape(-1).to(torch.float64)
            self._add_diagnostic("data_read_retries", retries.sum())
            self._add_diagnostic(
                "data_read_samples",
                torch.tensor(
                    retries.numel(), dtype=torch.float64, device=retries.device
                ),
            )
            self._add_diagnostic("data_read_retried_samples", retries.gt(0).sum())

        action_mask = batch.get("action_mask")
        if isinstance(action_mask, torch.Tensor):
            valid = action_mask.detach().to(dtype=torch.bool)
            self._add_diagnostic("action_valid_elements", valid.sum())
            self._add_diagnostic(
                "action_total_elements",
                torch.tensor(
                    valid.numel(), dtype=torch.float64, device=valid.device
                ),
            )
            valid_timesteps = valid.any(dim=-1)
            self._add_diagnostic("action_valid_timesteps", valid_timesteps.sum())
            self._add_diagnostic(
                "action_total_timesteps",
                torch.tensor(
                    valid_timesteps.numel(),
                    dtype=torch.float64,
                    device=valid_timesteps.device,
                ),
            )

    def _add_diagnostic(self, key: str, value) -> None:
        if not isinstance(value, torch.Tensor):
            value = torch.as_tensor(value, dtype=torch.float64)
        value = value.detach().to(dtype=torch.float64).reshape(())
        self._diagnostic_stats[key] = self._diagnostic_stats.get(
            key, torch.zeros_like(value)
        ) + value

    @staticmethod
    def _reduce(accelerator, value: torch.Tensor, reduction: str) -> torch.Tensor:
        if reduction in {"sum", "mean"}:
            return accelerator.reduce(value, reduction=reduction).detach()
        gathered = accelerator.gather(value.reshape(1)).detach()
        if reduction == "min":
            return gathered.min()
        if reduction == "max":
            return gathered.max()
        raise ValueError(f"unsupported metric reduction {reduction!r}")

    def finalize(self, accelerator) -> dict[str, torch.Tensor]:
        metrics = {}
        branch_means = {}
        device = None
        for branch in ("ar", "fm"):
            if branch not in self._loss_sums:
                continue
            global_sum = self._reduce(accelerator, self._loss_sums[branch], "sum")
            global_count = self._reduce(accelerator, self._loss_counts[branch], "sum")
            device = global_sum.device
            mean = global_sum / global_count.clamp_min(1.0)
            mean = mean * (global_count > 0).to(mean.dtype)
            branch_means[branch] = mean
            if branch == "ar":
                metrics.update(
                    ar_loss=mean,
                    vlm_loss=mean,
                    ar_loss_count=global_count,
                )
            else:
                metrics.update(
                    flow_matching_loss=mean,
                    action_expert_loss=mean,
                    flow_matching_loss_count=global_count,
                )

        if "ar" in branch_means:
            weighted_ar = branch_means["ar"] * self.vlm_loss_weight
            metrics["weighted_ar_loss"] = weighted_ar
            metrics["weighted_vlm_loss"] = weighted_ar
        if "fm" in branch_means:
            weighted_fm = branch_means["fm"] * self.action_expert_loss_weight
            metrics["weighted_flow_matching_loss"] = weighted_fm
            metrics["weighted_action_expert_loss"] = weighted_fm
        weighted_terms = [
            metrics[key]
            for key in ("weighted_ar_loss", "weighted_flow_matching_loss")
            if key in metrics
        ]
        if weighted_terms:
            total = sum(weighted_terms[1:], weighted_terms[0])
            metrics["loss"] = total
            metrics["total_loss"] = total
        reference_device = device if device is not None else torch.device("cpu")
        metrics["vlm_loss_weight"] = torch.tensor(
            self.vlm_loss_weight, device=reference_device, dtype=torch.float64
        )
        metrics["action_expert_loss_weight"] = torch.tensor(
            self.action_expert_loss_weight,
            device=reference_device,
            dtype=torch.float64,
        )

        for key, spec in TOKENIZATION_METRIC_SCHEMA.items():
            if key in self._token_stats:
                stats = self._token_stats[key]
                local_sum = stats["sum"]
                local_count = stats["count"]
                local_min = stats["min"]
                local_max = stats["max"]
            else:
                local_sum = torch.zeros((), dtype=torch.float64, device=reference_device)
                local_count = torch.zeros((), dtype=torch.float64, device=reference_device)
                local_min = torch.full((), float("inf"), dtype=torch.float64, device=reference_device)
                local_max = torch.full((), float("-inf"), dtype=torch.float64, device=reference_device)
            global_sum = self._reduce(accelerator, local_sum, "sum")
            global_count = self._reduce(accelerator, local_count, "sum")
            global_min = self._reduce(accelerator, local_min, "min")
            global_max = self._reduce(accelerator, local_max, "max")
            if global_count.item() <= 0:
                continue
            mean = global_sum / global_count
            if spec.aggregation == "ratio":
                metrics[f"{key}_ratio"] = mean
                metrics[f"{key}_count"] = global_sum
            else:
                metrics[key] = mean
                metrics[f"{key}_sum"] = global_sum
                metrics[f"{key}_min"] = global_min
                metrics[f"{key}_max"] = global_max

        diagnostics = {}
        for key, value in self._diagnostic_stats.items():
            diagnostics[key] = self._reduce(accelerator, value, "sum")
            metrics[key] = diagnostics[key]
        if diagnostics.get("data_read_samples", torch.tensor(0)).item() > 0:
            metrics["data_read_retry_sample_ratio"] = (
                diagnostics["data_read_retried_samples"]
                / diagnostics["data_read_samples"]
            )
        if diagnostics.get("action_total_elements", torch.tensor(0)).item() > 0:
            metrics["action_invalid_element_ratio"] = 1.0 - (
                diagnostics["action_valid_elements"]
                / diagnostics["action_total_elements"]
            )
        if diagnostics.get("action_total_timesteps", torch.tensor(0)).item() > 0:
            metrics["action_invalid_timestep_ratio"] = 1.0 - (
                diagnostics["action_valid_timesteps"]
                / diagnostics["action_total_timesteps"]
            )
        return metrics
