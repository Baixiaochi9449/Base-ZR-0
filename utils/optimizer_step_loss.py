"""Optimizer-step loss normalization and detached metric aggregation."""

from __future__ import annotations

from dataclasses import dataclass
import copy

import torch

from utils.training_tokenization import (
    TOKENIZATION_METRIC_SCHEMA,
    token_metric_validity_key,
)


AR_LOSS_TYPES = {"vlm", "vlm_and_action"}
FM_LOSS_TYPES = {"action", "vlm_and_action"}


def optimizer_step_result(*, applied: bool, reason: str = "none") -> dict:
    if type(applied) is not bool or reason not in {
        "none", "no_supervision", "amp_overflow", "optimizer_step_skipped"
    } or applied != (reason == "none"):
        raise RuntimeError("invalid optimizer step result")
    return {"optimizer_update_applied": applied, "optimizer_update_skipped": not applied,
            "optimizer_skip_reason": reason}


def optimizer_update_applied(metrics: dict) -> bool:
    expected = optimizer_step_result(applied=metrics.get("optimizer_update_applied"),
                                     reason=metrics.get("optimizer_skip_reason"))
    if type(metrics.get("optimizer_update_skipped")) is not bool or metrics["optimizer_update_skipped"] != expected["optimizer_update_skipped"]:
        raise RuntimeError("missing or inconsistent optimizer step result")
    return expected["optimizer_update_applied"]


class OptimizerWindowStep:
    """Resolve the backend's update result before advancing any external scheduler."""

    def __init__(self, *, model, optimizer, lr_scheduler, accelerator):
        from accelerate.optimizer import AcceleratedOptimizer
        from accelerate.utils import DistributedType
        self.model, self.optimizer = model, optimizer
        self.scheduler, self.accelerator = lr_scheduler, accelerator
        self.deepspeed = accelerator.distributed_type == DistributedType.DEEPSPEED
        if accelerator.distributed_type not in {DistributedType.NO, DistributedType.MULTI_CPU,
                                               DistributedType.MULTI_GPU, DistributedType.DEEPSPEED}:
            raise RuntimeError("unsupported backend: optimizer update status has not been verified")
        self.engine_scheduler = getattr(model, "lr_scheduler", None) if self.deepspeed else None
        if (isinstance(optimizer, (list, tuple)) or len(getattr(accelerator, "_optimizers", [])) > 1
                or len(getattr(lr_scheduler, "optimizers", [])) > 1):
            raise RuntimeError("optimizer windows require one optimizer; partial multi-optimizer updates are unsupported")
        if self.deepspeed:
            if not callable(getattr(model, "was_step_applied", None)):
                raise RuntimeError("DeepSpeed must expose was_step_applied(); optimizer update status is unknown")
            if self.engine_scheduler is not None:
                if lr_scheduler is not self.engine_scheduler and getattr(lr_scheduler, "scheduler", None) is not self.engine_scheduler:
                    raise RuntimeError("DeepSpeed and external scheduler ownership do not match")
                self.scheduler_before = copy.deepcopy(self.engine_scheduler.state_dict())
        else:
            self.accelerated = isinstance(optimizer, AcceleratedOptimizer)
            native = optimizer.optimizer if self.accelerated else optimizer
            # These native optimizers have no implicit skip path outside AMP.
            if type(native) not in {torch.optim.AdamW, torch.optim.Adam, torch.optim.SGD}:
                raise RuntimeError("unsupported optimizer: no verified optimizer step-result contract")
            if not self.accelerated and getattr(accelerator, "scaler", None) is not None:
                raise RuntimeError("AMP requires AcceleratedOptimizer.step_was_skipped")
            if self.accelerated and not optimizer.gradient_state.sync_gradients:
                raise RuntimeError("optimizer window must end at a synchronized update boundary")

    def finish(self):
        if self.deepspeed:
            # Accelerate has already called engine.step() inside backward.
            applied = self.model.was_step_applied()
            overflow = getattr(getattr(self.model, "optimizer", None), "overflow", False)
            reason = "amp_overflow" if overflow else "optimizer_step_skipped"
        else:
            if self.accelerated and not self.optimizer.gradient_state.sync_gradients:
                raise RuntimeError("optimizer window is no longer at a synchronized update boundary")
            self.optimizer.step()
            skipped = self.optimizer.step_was_skipped if self.accelerated else False
            if type(skipped) is not bool:
                raise RuntimeError("optimizer step_was_skipped is unknown/non-boolean")
            applied = not skipped
            reason = "amp_overflow" if self.accelerated and self.optimizer.scaler is not None else "optimizer_step_skipped"
        if type(applied) is not bool:
            raise RuntimeError("backend optimizer update status is unknown/non-boolean")
        total = self.accelerator.reduce(
            torch.tensor([int(applied), int(not applied and reason == "amp_overflow")],
                         device=self.accelerator.device, dtype=torch.long), reduction="sum")
        if int(total[0].item()) not in {0, self.accelerator.num_processes}:
            raise RuntimeError("optimizer update results differ across ranks; refusing to advance training state")
        if not applied:
            reason = "amp_overflow" if total[1].item() else "optimizer_step_skipped"
        if not applied and self.engine_scheduler is not None:
            from torch.utils._pytree import tree_flatten
            before, before_spec = tree_flatten(self.scheduler_before)
            after, after_spec = tree_flatten(self.engine_scheduler.state_dict())
            equal = before_spec == after_spec and all(
                torch.equal(a, b) if isinstance(a, torch.Tensor) else a == b for a, b in zip(before, after))
            if not equal:
                raise RuntimeError("DeepSpeed scheduler changed during a skipped update; refusing completed-window state")
        if applied and self.engine_scheduler is None:
            self.scheduler.step()
        return optimizer_step_result(applied=applied, reason="none" if applied else reason)


@dataclass(frozen=True)
class GlobalSupervisionCounts:
    ar_tokens: torch.Tensor
    action_elements: torch.Tensor
    flow_samples: torch.Tensor | None = None


def global_supervision_counts(
    batches: list[dict],
    *,
    loss_type: str,
    accelerator,
    optical_flow_config=None,
) -> GlobalSupervisionCounts:
    if not batches:
        raise ValueError("optimizer-step accumulation window must not be empty")
    device = accelerator.device
    flow_enabled = optical_flow_config is not None and optical_flow_config.enabled
    local = torch.zeros(3 if flow_enabled else 2, dtype=torch.float64, device=device)
    for batch in batches:
        if flow_enabled:
            from utils.optical_flow_loss import prepare_flow_targets
            local[2] += len(prepare_flow_targets(batch, optical_flow_config, device)[0])
        if loss_type in AR_LOSS_TYPES:
            labels = batch.get("labels")
            if not isinstance(labels, torch.Tensor):
                raise ValueError("labels are required to count optimizer-step AR tokens")
            local[0] += labels[..., 1:].ne(-100).sum().to(device=device, dtype=torch.float64)
        if loss_type in FM_LOSS_TYPES:
            action_mask = batch.get("action_mask")
            if not isinstance(action_mask, torch.Tensor):
                raise ValueError(
                    "action_mask is required to count optimizer-step action elements"
                )
            local[1] += action_mask.to(dtype=torch.bool).sum().to(device=device, dtype=torch.float64)
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
        flow_samples=global_counts[2] if flow_enabled else None,
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
    optical_flow_loss_weight: float = 0.0,
) -> torch.Tensor:
    scale = float(gradient_accumulation_steps * data_parallel_world_size)
    terms = []
    if optical_flow_loss_weight > 0:
        flow_sum = outputs["optical_flow_loss_sum"]
        terms.append(optical_flow_loss_weight * flow_sum / counts.flow_samples.to(flow_sum.dtype).clamp_min(1))
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
        device=None,
    ):
        self.loss_type = loss_type
        self.vlm_loss_weight = float(vlm_loss_weight)
        self.action_expert_loss_weight = float(action_expert_loss_weight)
        self.collect_diagnostics = bool(collect_diagnostics)
        self.device = device
        self._loss_sums = {}
        self._loss_counts = {}
        self._token_stats = {}
        self._diagnostic_stats = {}
        self._stage_metadata = {}

    def _detached_scalar(self, value, *, dtype=torch.float64):
        if not isinstance(value, torch.Tensor):
            raise ValueError("optimizer-step loss statistics must be tensors")
        return value.detach().to(device=self.device, dtype=dtype).reshape(())

    def update(self, outputs: dict, batch: dict) -> None:
        for key in ("training_stage", "provisional", "slot_loss_computed", "ar_loss_computed",
                    "flow_loss_computed", "fm_loss_computed"):
            if key in outputs:
                value = outputs[key]
                if key.endswith("_computed"):
                    self._stage_metadata[key] = self._stage_metadata.get(key, False) or bool(value)
                else:
                    if key in self._stage_metadata and self._stage_metadata[key] != value:
                        raise ValueError(f"optimizer window has inconsistent {key}")
                    self._stage_metadata[key] = value
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
            values = value.detach().reshape(-1).to(device=self.device, dtype=torch.float64)
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
            retries = retry_counts.detach().reshape(-1).to(device=self.device, dtype=torch.float64)
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
            valid = action_mask.detach().to(device=self.device, dtype=torch.bool)
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
            value = torch.as_tensor(value, dtype=torch.float64, device=self.device)
        value = value.detach().to(device=self.device, dtype=torch.float64).reshape(())
        self._diagnostic_stats[key] = self._diagnostic_stats.get(
            key, torch.zeros_like(value)
        ) + value

    @staticmethod
    def _reduce(accelerator, value: torch.Tensor, reduction: str) -> torch.Tensor:
        value = value.to(device=accelerator.device)
        if reduction in {"sum", "mean"}:
            return accelerator.reduce(value, reduction=reduction).detach()
        gathered = accelerator.gather(value.reshape(1)).detach()
        if reduction == "min":
            return gathered.min()
        if reduction == "max":
            return gathered.max()
        raise ValueError(f"unsupported metric reduction {reduction!r}")

    def finalize(self, accelerator) -> dict:
        metrics = {}
        branch_means = {}
        device = accelerator.device
        for branch in ("ar", "fm"):
            if branch not in self._loss_sums:
                continue
            global_sum = self._reduce(accelerator, self._loss_sums[branch], "sum")
            global_count = self._reduce(accelerator, self._loss_counts[branch], "sum")
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
        reference_device = device
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
        for key, value in self._stage_metadata.items():
            if key.endswith("_computed"):
                flag = torch.tensor(int(value), device=accelerator.device)
                value = bool(self._reduce(accelerator, flag, "sum").item())
            metrics[key] = value
        return metrics
