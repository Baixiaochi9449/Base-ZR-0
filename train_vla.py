import os
import math
import time
import json
import hashlib
from pathlib import Path
import torch
import torch.distributed as dist
from contextlib import nullcontext

from utils.load_training_dataset import (
    build_concat_streaming_dataset,
    create_dataloader_for_concat,
    resolve_dataloader_num_workers,
)
from model import FlowmatchingActionHeadConfig
from model import ZR0Model
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.utils import DistributedType
from torch.optim import AdamW
from accelerate.utils import set_seed
from transformers import get_cosine_with_min_lr_schedule_with_warmup_lr_rate, get_constant_schedule_with_warmup
from torch.utils.tensorboard import SummaryWriter
from utils.training_numerics import assert_all_finite
from utils.wandb_training_logger import WandbTrainingLogger
from utils.action_expert_config import load_action_expert_config, read_vlm_hidden_size
from utils.stage05_checkpoint_contract import (
    DOWNSTREAM_FINETUNE,
    INFERENCE,
    STAGE05_AR_RESUME,
    STAGE05_AR_TO_JOINT,
    STAGE05_JOINT_RESUME,
    _checkpoint_has_stage05_identity,
    validate_checkpoint_load_purpose_arguments,
    validate_generic_action_expert_contract,
    validate_stage05_checkpoint_for_purpose,
    validate_stage05_resume_artifacts,
)
from utils.cli_options import parse_train_options
from utils.training_checkpoint import (
    checkpoint_model_optimizer_scheduler,
    resume_model_optimizer_scheduler,
)
from utils.dataset_manifest import (
    resolved_manifest_json,
    validate_resume_manifest,
    write_resolved_dataset_manifest,
)
from utils.training_tokenization import (
    TOKENIZATION_METRIC_SCHEMA,
    token_metric_validity_key,
)
from utils.optimizer_step_loss import (
    OptimizerStepMetricAccumulator,
    global_supervision_counts,
    scaled_microbatch_loss,
)
from utils.dataset_seen_tracker import DatasetSeenTracker

def parse_option(args=None):
    return parse_train_options(args)


def should_skip_resumed_batch(
    *, epoch: int, batch_idx: int, resume_epoch: int, resume_batch_idx: int
) -> bool:
    return epoch == resume_epoch and batch_idx < resume_batch_idx


def resume_data_position(
    *, global_completed_steps: int, dataloader_length: int, gradient_accumulation_steps: int
):
    if global_completed_steps < 0:
        raise ValueError("global_completed_steps must be non-negative")
    if dataloader_length < 1 or gradient_accumulation_steps < 1:
        raise ValueError("dataloader_length and gradient_accumulation_steps must be positive")
    steps_per_epoch = math.ceil(dataloader_length / gradient_accumulation_steps)
    resume_epoch = global_completed_steps // steps_per_epoch
    completed_steps_in_epoch = global_completed_steps % steps_per_epoch
    resume_batch_idx = min(
        completed_steps_in_epoch * gradient_accumulation_steps, dataloader_length
    )
    return resume_epoch, resume_batch_idx


def synchronize_global_step(global_completed_steps: int, accelerator) -> int:
    if accelerator.num_processes == 1:
        return global_completed_steps
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError(
            "multi-process global-step synchronization requires an initialized "
            "torch.distributed process group"
        )
    step_tensor = torch.tensor(
        global_completed_steps,
        device=accelerator.device,
        dtype=torch.long,
    )
    dist.broadcast(step_tensor, src=0)
    return int(step_tensor.item())


def get_trainable_parameters(model):
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("No trainable parameters were found for the optimizer")
    return parameters


PARAMETER_OWNERS = ("vlm", "difference_query", "action_expert")


def trainable_parameter_owner(name: str) -> str:
    if name == "backbone.difference_query.weight" or name.startswith(
        "backbone.difference_query."
    ):
        return "difference_query"
    if name.startswith("backbone."):
        return "vlm"
    if name.startswith("action_expert."):
        return "action_expert"
    raise ValueError(f"trainable parameter has no diagnostic owner: {name}")


def validate_trainable_parameter_ownership(model) -> dict[str, int]:
    counts = {owner: 0 for owner in PARAMETER_OWNERS}
    seen = set()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if id(parameter) in seen:
            raise ValueError(f"trainable parameter is registered more than once: {name}")
        seen.add(id(parameter))
        counts[trainable_parameter_owner(name)] += parameter.numel()
    if not seen:
        raise ValueError("No trainable parameters were found for diagnostics")
    return counts


def tensor_initialization_statistics(tensor: torch.Tensor) -> dict:
    value = tensor.detach().cpu().contiguous()
    numeric = value.float()
    raw_bytes = value.view(torch.uint8).numpy().tobytes()
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "mean": numeric.mean().item(),
        "std": numeric.std(unbiased=False).item(),
        "min": numeric.min().item(),
        "max": numeric.max().item(),
        "l2_norm": numeric.square().sum().sqrt().item(),
        "sha256": hashlib.sha256(raw_bytes).hexdigest(),
    }


def module_parameter_sha256(module) -> str | None:
    if module is None:
        return None
    digest = hashlib.sha256()
    for name, parameter in module.named_parameters():
        value = parameter.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def write_wandb_finish_diagnostics(metrics_log, writer, *, step: int, diagnostics: dict):
    record = {"event": "wandb_finish", "step": int(step), **diagnostics}
    if writer is not None:
        for name, value in diagnostics.items():
            writer.add_scalar(name.replace("_", "-"), value, step)
    if metrics_log is not None:
        metrics_log.write(
            json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n"
        )
        metrics_log.flush()
    return record


def write_initialization_manifest(
    model, opt, ownership_counts, resolved_action_expert_config=None
) -> str:
    query = model.backbone.difference_query
    query_config_path = os.path.join(
        opt.vlm_name_or_path, "difference_query_config.json"
    )
    action_source = None
    if model.action_expert is not None:
        action_source = (
            "checkpoint"
            if opt.action_expert_name_or_path
            else f"random_seed_{opt.seed}"
        )
    manifest = {
        "version": 1,
        "seed": opt.seed,
        "resolved_options": vars(opt),
        "trainable_parameter_ownership": ownership_counts,
        "difference_query": {
            "source": (
                "checkpoint"
                if os.path.isfile(query_config_path)
                else f"random_seed_{opt.seed}"
            ),
            "statistics": (
                tensor_initialization_statistics(query.weight)
                if query is not None
                else None
            ),
        },
        "action_expert": {
            "constructed": model.action_expert is not None,
            "source": action_source,
            "config": (
                {
                    "path": str(resolved_action_expert_config.path),
                    "source_sha256": resolved_action_expert_config.source_sha256,
                    "resolved_sha256": resolved_action_expert_config.parsed_sha256,
                    "source_action_horizon": (
                        resolved_action_expert_config.source_action_horizon
                    ),
                    "resolved_action_horizon": (
                        resolved_action_expert_config.config.action_horizon
                    ),
                    "action_horizon_overridden": (
                        resolved_action_expert_config.action_horizon_overridden
                    ),
                }
                if resolved_action_expert_config is not None
                else None
            ),
            "parameter_count": (
                sum(parameter.numel() for parameter in model.action_expert.parameters())
                if model.action_expert is not None
                else 0
            ),
            "parameter_sha256": module_parameter_sha256(model.action_expert),
        },
    }
    filename = (
        "initialization_manifest_resume.json"
        if opt.resume_training
        else "initialization_manifest_fresh.json"
    )
    path = os.path.join(opt.output_ckpt_dir, filename)
    with open(path, "w", encoding="utf-8") as output:
        json.dump(manifest, output, ensure_ascii=True, indent=2, sort_keys=True)
        output.write("\n")
    return path


def module_gradient_norms(model, accelerator) -> dict[str, torch.Tensor]:
    """Compute owner norms without changing optimizer parameter groups."""
    unwrapped = accelerator.unwrap_model(model)
    is_deepspeed = accelerator.distributed_type == DistributedType.DEEPSPEED
    squared = {
        owner: torch.zeros((), dtype=torch.float64, device=accelerator.device)
        for owner in PARAMETER_OWNERS
    }
    parameter_counts = {owner: 0 for owner in PARAMETER_OWNERS}
    fragment_counts = {owner: 0 for owner in PARAMETER_OWNERS}
    parameter_owners = {}
    for name, parameter in unwrapped.named_parameters():
        if not parameter.requires_grad:
            continue
        owner = trainable_parameter_owner(name)
        parameter_counts[owner] += 1
        parameter_owners[id(parameter)] = owner

    if is_deepspeed:
        zero_optimizer = getattr(model, "optimizer", None)
        gradients_by_group = getattr(zero_optimizer, "averaged_gradients", None)
        parameters_by_group = getattr(zero_optimizer, "params_in_partition", None)
        if not isinstance(gradients_by_group, dict) or parameters_by_group is None:
            raise RuntimeError("DeepSpeed ZeRO-2 partition gradients are unavailable")
        for group_index, parameters in enumerate(parameters_by_group):
            gradients = gradients_by_group.get(group_index)
            if gradients is None:
                raise RuntimeError(
                    f"DeepSpeed ZeRO-2 gradient partition {group_index} is unavailable"
                )
            if len(gradients) < len(parameters):
                raise RuntimeError(
                    "DeepSpeed ZeRO-2 gradient/parameter partition lengths differ: "
                    f"group={group_index}, gradients={len(gradients)}, "
                    f"parameters={len(parameters)}"
                )
            for parameter, gradient in zip(parameters, gradients):
                owner = parameter_owners.get(id(parameter))
                if owner is None:
                    raise RuntimeError(
                        "DeepSpeed ZeRO-2 partition parameter has no ownership mapping"
                    )
                fragment_counts[owner] += 1
                squared[owner] += gradient.detach().to(torch.float32).square().sum().to(
                    torch.float64
                )
    else:
        for name, parameter in unwrapped.named_parameters():
            if not parameter.requires_grad or parameter.grad is None:
                continue
            owner = trainable_parameter_owner(name)
            fragment_counts[owner] += 1
            squared[owner] += parameter.grad.detach().to(torch.float32).square().sum().to(
                torch.float64
            )

    metrics = {}
    for owner in PARAMETER_OWNERS:
        if parameter_counts[owner] == 0:
            continue
        if is_deepspeed:
            global_squared = accelerator.reduce(squared[owner], reduction="sum")
            global_fragments = accelerator.reduce(
                torch.tensor(
                    fragment_counts[owner],
                    dtype=torch.long,
                    device=accelerator.device,
                ),
                reduction="sum",
            )
            if global_fragments.item() == 0:
                raise RuntimeError(
                    f"DeepSpeed exposed no gradient fragments for {owner}"
                )
        else:
            global_squared = accelerator.reduce(squared[owner], reduction="mean")
        metrics[f"{owner}_grad_norm"] = global_squared.clamp_min(0).sqrt()
    return metrics


def build_adamw_optimizer(model, *, learning_rate, beta1, beta2, epsilon):
    return AdamW(
        get_trainable_parameters(model),
        lr=learning_rate,
        betas=(beta1, beta2),
        eps=epsilon,
        weight_decay=0.01,
    )


def effective_total_optimizer_steps(epoch_horizon: int, max_train_steps):
    if epoch_horizon < 1:
        raise ValueError("epoch_horizon must be positive")
    if max_train_steps is None:
        return epoch_horizon
    if max_train_steps < 1:
        raise ValueError("max_train_steps must be positive")
    return min(epoch_horizon, max_train_steps)


def calculate_warmup_steps(total_optimizer_steps: int, num_processes: int, warmup_ratio):
    if warmup_ratio is None:
        per_process_steps = min(20000, int(total_optimizer_steps * 0.08))
    else:
        per_process_steps = int(total_optimizer_steps * warmup_ratio)
    return per_process_steps * num_processes


def loss_output_metrics(outputs):
    metric_keys = (
        "loss",
        "total_loss",
        "vlm_loss",
        "action_expert_loss",
        "ar_loss",
        "flow_matching_loss",
        "weighted_vlm_loss",
        "weighted_action_expert_loss",
        "weighted_ar_loss",
        "weighted_flow_matching_loss",
        "vlm_loss_weight",
        "action_expert_loss_weight",
    )
    reference = next(
        (value for value in outputs.values() if isinstance(value, torch.Tensor)),
        None,
    )
    metrics = {}
    for key in metric_keys:
        if key not in outputs:
            continue
        value = outputs[key]
        metrics[key] = (
            value
            if isinstance(value, torch.Tensor)
            else torch.as_tensor(
                value, device=reference.device if reference is not None else None
            )
        )
    return metrics


def batch_token_metrics(batch):
    metrics = {}
    for key, spec in TOKENIZATION_METRIC_SCHEMA.items():
        value = batch.get(key)
        validity = batch.get(token_metric_validity_key(key))
        if not isinstance(value, torch.Tensor) or not isinstance(validity, torch.Tensor):
            continue
        values = value.reshape(-1)
        valid = validity.to(device=value.device, dtype=torch.bool).reshape(-1)
        if values.numel() != valid.numel():
            raise ValueError(f"token metric {key!r} and validity mask shapes differ")
        selected = values[valid].float()
        if selected.numel() == 0:
            continue
        if spec.aggregation == "ratio":
            metrics[f"{key}_ratio"] = selected.mean()
        else:
            metrics[key] = selected.mean()
            metrics[f"{key}_min"] = selected.min()
            metrics[f"{key}_max"] = selected.max()
    return metrics


def iter_optimizer_step_windows(indexed_batches, gradient_accumulation_steps: int):
    if gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be positive")
    window = []
    for item in indexed_batches:
        window.append(item)
        if len(window) == gradient_accumulation_steps:
            yield window
            window = []
    if window:
        yield window


def run_optimizer_step_window(
    *,
    model,
    batches,
    accelerator,
    optimizer,
    lr_scheduler,
    training_progress: float,
    loss_type: str,
    vlm_loss_weight: float,
    action_expert_loss_weight: float,
    next_global_step: int,
    collect_training_diagnostics: bool = False,
):
    batches = list(batches)
    if not batches:
        raise ValueError("optimizer-step accumulation window must not be empty")
    if collect_training_diagnostics and accelerator.device.type == "cuda":
        torch.cuda.synchronize(accelerator.device)
    step_started = time.perf_counter()
    counts = global_supervision_counts(
        batches, loss_type=loss_type, accelerator=accelerator
    )
    accumulator = OptimizerStepMetricAccumulator(
        loss_type=loss_type,
        vlm_loss_weight=vlm_loss_weight,
        action_expert_loss_weight=action_expert_loss_weight,
        collect_diagnostics=collect_training_diagnostics,
    )
    optimizer.zero_grad()
    is_deepspeed = accelerator.distributed_type == DistributedType.DEEPSPEED
    diagnostic_metrics = {}
    original_deepspeed_step = None
    deepspeed_diagnostic_calls = 0
    if collect_training_diagnostics and is_deepspeed:
        zero_optimizer = getattr(model, "optimizer", None)
        original_deepspeed_step = getattr(zero_optimizer, "step", None)
        if original_deepspeed_step is None:
            raise RuntimeError("DeepSpeed optimizer step is unavailable for diagnostics")

        def diagnostic_deepspeed_step(*args, **kwargs):
            nonlocal deepspeed_diagnostic_calls
            deepspeed_diagnostic_calls += 1
            diagnostic_metrics.update(module_gradient_norms(model, accelerator))
            return original_deepspeed_step(*args, **kwargs)

        zero_optimizer.step = diagnostic_deepspeed_step

    try:
        for microbatch_index, batch in enumerate(batches):
            is_boundary = microbatch_index == len(batches) - 1
            if is_deepspeed:
                if not hasattr(model, "set_gradient_accumulation_boundary"):
                    raise RuntimeError(
                        "DeepSpeed optimizer-step windows require the engine boundary API"
                    )
                model.set_gradient_accumulation_boundary(is_boundary)
            sync_context = (
                nullcontext()
                if is_deepspeed or is_boundary
                else accelerator.no_sync(model)
            )
            with sync_context:
                outputs = model(
                    batch,
                    training_progress,
                    vlm_loss_weight=vlm_loss_weight,
                    action_expert_loss_weight=action_expert_loss_weight,
                )
                backward_loss = scaled_microbatch_loss(
                    outputs,
                    counts=counts,
                    loss_type=loss_type,
                    vlm_loss_weight=vlm_loss_weight,
                    action_expert_loss_weight=action_expert_loss_weight,
                    gradient_accumulation_steps=accelerator.gradient_accumulation_steps,
                    data_parallel_world_size=accelerator.num_processes,
                )
                assert_all_finite(
                    accelerator, backward_loss, "loss", next_global_step
                )
                accumulator.update(outputs, batch)
                accelerator.backward(backward_loss)
    finally:
        if original_deepspeed_step is not None:
            zero_optimizer.step = original_deepspeed_step

    if collect_training_diagnostics and is_deepspeed:
        if deepspeed_diagnostic_calls != 1:
            raise RuntimeError(
                "DeepSpeed optimizer-step diagnostics must run exactly once, got "
                f"{deepspeed_diagnostic_calls}"
            )
    elif collect_training_diagnostics:
        diagnostic_metrics.update(module_gradient_norms(model, accelerator))
    optimizer.step()
    lr_scheduler.step()
    optimizer.zero_grad()
    metrics = accumulator.finalize(accelerator)
    if collect_training_diagnostics:
        metrics["optimizer_microbatches_per_rank"] = torch.tensor(
            len(batches), dtype=torch.float64, device=accelerator.device
        )
        if accelerator.device.type == "cuda":
            torch.cuda.synchronize(accelerator.device)
        local_elapsed = torch.tensor(
            time.perf_counter() - step_started,
            dtype=torch.float64,
            device=accelerator.device,
        )
        elapsed = accelerator.gather(local_elapsed.reshape(1)).max()
        local_samples = sum(
            int(batch["input_ids"].shape[0]) for batch in batches
        )
        global_samples = accelerator.reduce(
            torch.tensor(
                local_samples, dtype=torch.float64, device=accelerator.device
            ),
            reduction="sum",
        )
        diagnostic_metrics.update(
            optimizer_step_seconds=elapsed,
            optimizer_step_global_samples=global_samples,
            samples_per_second=global_samples / elapsed.clamp_min(1e-12),
        )
        if accelerator.device.type == "cuda":
            memory = torch.tensor(
                [
                    torch.cuda.max_memory_allocated(accelerator.device),
                    torch.cuda.max_memory_reserved(accelerator.device),
                ],
                dtype=torch.float64,
                device=accelerator.device,
            )
            memory = accelerator.gather(memory.reshape(1, 2)).max(dim=0).values
            diagnostic_metrics.update(
                gpu_peak_memory_allocated_gib=memory[0] / (1024**3),
                gpu_peak_memory_reserved_gib=memory[1] / (1024**3),
            )
    metrics.update(diagnostic_metrics)
    return metrics


def tensorboard_loss_value(value) -> float:
    return float(value.detach().float()) if isinstance(value, torch.Tensor) else float(value)


def json_scalar_metrics(metrics: dict) -> dict[str, float]:
    return {
        name: tensorboard_loss_value(value) for name, value in metrics.items()
    }


def tensorboard_loss_metric_name(output_name: str) -> str:
    return {
        "vlm_loss": "vlm-loss",
        "action_expert_loss": "action-expert-loss",
    }.get(output_name, output_name)

def get_absolute_path(path):
    if os.path.isabs(path):
        return path
    else:
        current_path = os.getcwd()
        return os.path.join(current_path, path)

def print_model_parameters(model):
    for name, p in model.named_parameters():
        print(f"{name}: {p.numel()}")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"# all params: {total_params}")
    print(f"# trainable params: {trainable_params}")

def integrity_check(batch_data, processor):
    labels = batch_data.get("labels")
    if labels is None:
        for input_id, attn_mask in zip(
            batch_data["input_ids"][0], batch_data["attention_mask"][0]
        ):
            print(
                f"{input_id} -> {repr(processor.tokenizer.decode(input_id))} -> {attn_mask}"
            )
    else:
        for input_id, label, attn_mask in zip(
            batch_data["input_ids"][0], labels[0], batch_data["attention_mask"][0]
        ):
            print(
                f"{input_id} -> {repr(processor.tokenizer.decode(input_id))} -> {label} -> {attn_mask}"
            )
    print("integrity check ends.")

def save_model(accelerator: Accelerator, model, output_ckpt_dir, tag):
    accelerator.print(f"save model at {tag}")
    # accelerator.wait_for_everyone()
    # save checkpoint
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(os.path.join(output_ckpt_dir, tag))
    # accelerator.wait_for_everyone()

def resolve_action_expert_config(opt, *, return_resolved=False):
    purpose = getattr(opt, "checkpoint_load_purpose", None)
    validate_checkpoint_load_purpose_arguments(
        purpose, resume_training=getattr(opt, "resume_training", False)
    )
    if purpose == INFERENCE:
        raise ValueError("checkpoint_load_purpose=inference is not a training mode")
    explicit_path = getattr(opt, "action_expert_config_path", None)
    if purpose is None:
        for candidate in (
            getattr(opt, "vlm_name_or_path", None),
            getattr(opt, "action_expert_name_or_path", None),
        ):
            if not candidate:
                continue
            candidate_path = Path(candidate)
            if candidate_path.is_dir() and _checkpoint_has_stage05_identity(candidate_path):
                raise ValueError(
                    "Stage05 checkpoint requires an explicit checkpoint_load_purpose; "
                    "booleans and legacy config loading cannot bypass its contract"
                )
            metadata_path = candidate_path / "zr0_checkpoint_metadata.json"
            if metadata_path.is_file():
                try:
                    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                except Exception as error:
                    raise ValueError(
                        f"failed to read checkpoint metadata {metadata_path}: {error}"
                    ) from error
                if isinstance(metadata, dict) and "action_expert_contract" in metadata:
                    # Validate newer generic contracts before any legacy
                    # config lookup or optional horizon override.
                    validate_generic_action_expert_contract(candidate_path)
    checkpoint_path = None
    if opt.action_expert_name_or_path:
        checkpoint_path = os.path.join(
            opt.action_expert_name_or_path, "action_expert_config.json"
        )
    config_source = checkpoint_path or explicit_path
    if config_source is None:
        candidate = os.path.join(opt.vlm_name_or_path, "action_expert_config.json")
        if os.path.isfile(candidate):
            config_source = candidate
    if config_source is None:
        raise ValueError(
            "Action Expert configuration has no authoritative source; pass "
            "--action_expert_config_path or load a checkpoint containing it"
        )
    hidden_size = read_vlm_hidden_size(opt.vlm_name_or_path)
    if purpose == STAGE05_AR_RESUME:
        if opt.loss_type != "vlm" or opt.action_expert_name_or_path:
            raise ValueError("stage05_ar_resume requires loss_type=vlm and no Action Expert weights")
        validated = validate_stage05_checkpoint_for_purpose(
            opt.vlm_name_or_path, purpose=purpose, resume_training=opt.resume_training,
            external_config_path=explicit_path,
            requested_action_horizon=opt.action_horizon,
            expected_action_dim=opt.max_pad_state_and_action_length,
            expected_state_dim=opt.max_pad_state_and_action_length,
            expected_num_difference_queries=getattr(opt, "num_difference_queries", 32) or 32,
        )
        from model.difference_query import resolve_difference_query_config

        resolve_difference_query_config(
            opt.vlm_name_or_path, None,
            use_difference_query=opt.use_difference_query,
            num_difference_queries=opt.num_difference_queries,
            vlm_attention_backend=opt.vlm_attention_backend,
        )
        return validated if return_resolved else validated.config
    if purpose == STAGE05_AR_TO_JOINT:
        if explicit_path is None:
            raise ValueError(
                "stage05_ar_to_joint requires an AR checkpoint and explicit Expert config"
            )
        validated = validate_stage05_checkpoint_for_purpose(
            opt.vlm_name_or_path,
            purpose=purpose,
            external_config_path=explicit_path,
            requested_action_horizon=opt.action_horizon,
            expected_action_dim=opt.max_pad_state_and_action_length,
            expected_state_dim=opt.max_pad_state_and_action_length,
            expected_num_difference_queries=getattr(opt, "num_difference_queries", 32) or 32,
        )
        return validated if return_resolved else validated.config
    if purpose == STAGE05_JOINT_RESUME:
        if not checkpoint_path:
            raise ValueError("stage05_joint_resume requires a Joint checkpoint")
        validated = validate_stage05_resume_artifacts(
            opt.vlm_name_or_path,
            external_config_path=explicit_path,
            requested_action_horizon=opt.action_horizon,
            expected_action_dim=opt.max_pad_state_and_action_length,
            expected_state_dim=opt.max_pad_state_and_action_length,
            expected_num_difference_queries=getattr(opt, "num_difference_queries", 32) or 32,
        )
        return validated if return_resolved else validated.config
    if purpose == DOWNSTREAM_FINETUNE:
        if not opt.action_expert_name_or_path:
            raise ValueError(
                "downstream_finetune requires --action_expert_name_or_path so "
                "pretrained Action Expert weights are loaded"
            )
        source_checkpoint = opt.action_expert_name_or_path or opt.vlm_name_or_path
        validated = validate_stage05_checkpoint_for_purpose(
            source_checkpoint,
            purpose=purpose,
            external_config_path=explicit_path,
            requested_action_horizon=opt.action_horizon,
            resume_training=getattr(opt, "resume_training", False),
            expected_action_dim=opt.max_pad_state_and_action_length,
            expected_state_dim=opt.max_pad_state_and_action_length,
        )
        return validated if return_resolved else validated.config

    allow_horizon_override = (
        checkpoint_path is not None
        and not getattr(opt, "resume_training", False)
        and getattr(opt, "loss_type", None) in ("action", "vlm_and_action")
        and purpose is None
    )
    horizon_arguments = (
        {"action_horizon_override": opt.action_horizon}
        if allow_horizon_override
        else {"expected_action_horizon": opt.action_horizon}
    )
    resolved = load_action_expert_config(
        config_source,
        expected_action_dim=opt.max_pad_state_and_action_length,
        expected_state_dim=opt.max_pad_state_and_action_length,
        expected_vlm_hidden_size=hidden_size,
        **horizon_arguments,
    )
    if checkpoint_path and explicit_path:
        explicit = load_action_expert_config(
            explicit_path,
            expected_action_dim=opt.max_pad_state_and_action_length,
            expected_state_dim=opt.max_pad_state_and_action_length,
            expected_vlm_hidden_size=hidden_size,
            **horizon_arguments,
        )
        if explicit.parsed_sha256 != resolved.parsed_sha256:
            raise ValueError(
                "explicit Action Expert config does not match the loaded checkpoint config"
            )
    return resolved if return_resolved else resolved.config


def train(opt):
    set_seed(opt.seed)

    # Resolve and, for checkpoint-backed modes, validate the load contract before
    # constructing Accelerator, DeepSpeed engines, models, or allocating GPUs.
    resolved_action_expert_config = resolve_action_expert_config(
        opt, return_resolved=True
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=opt.gradient_accumulation_steps,
        dataloader_config=DataLoaderConfiguration(even_batches=False),
    )
    accelerator.print(opt)

    total_batch_size = (
        opt.per_device_train_batch_size
        * accelerator.num_processes
        * accelerator.gradient_accumulation_steps
    )
    if accelerator.gradient_accumulation_steps != opt.gradient_accumulation_steps:
        raise RuntimeError(
            "Accelerator gradient accumulation mismatch: "
            f"actual={accelerator.gradient_accumulation_steps}, "
            f"requested={opt.gradient_accumulation_steps}"
        )
    if (
        opt.expected_global_batch_size is not None
        and total_batch_size != opt.expected_global_batch_size
    ):
        raise RuntimeError(
            "global batch size mismatch: "
            f"actual={total_batch_size} = {accelerator.num_processes} x "
            f"{opt.per_device_train_batch_size} x "
            f"{accelerator.gradient_accumulation_steps}, "
            f"expected={opt.expected_global_batch_size}"
        )
    accelerator.print(
        "requested training scale: "
        f"world_size={accelerator.num_processes}, "
        f"micro_batch={opt.per_device_train_batch_size}, "
        f"gradient_accumulation_steps={accelerator.gradient_accumulation_steps}, "
        f"nominal_global_batch={total_batch_size}"
    )
    if accelerator.distributed_type == DistributedType.DEEPSPEED:
        deepspeed_config = accelerator.state.deepspeed_plugin.deepspeed_config
        runtime_batch_config = {
            "gradient_accumulation_steps": opt.gradient_accumulation_steps,
            "train_micro_batch_size_per_gpu": opt.per_device_train_batch_size,
            "train_batch_size": total_batch_size,
        }
        configured_gas = deepspeed_config.get("gradient_accumulation_steps")
        if configured_gas != opt.gradient_accumulation_steps:
            raise RuntimeError(
                "DeepSpeed plugin gradient accumulation mismatch before prepare: "
                f"plugin={configured_gas}, requested={opt.gradient_accumulation_steps}"
            )
        deepspeed_config.update(runtime_batch_config)
        accelerator.print(
            "DeepSpeed batch config before prepare: "
            f"{runtime_batch_config}"
        )
    writer = (
        SummaryWriter(opt.tensorboard_log_dir)
        if accelerator.is_main_process
        else None
    )

    concat_dataset = build_concat_streaming_dataset(
        dataset_entries=opt.dataset_entries,
        model_name_or_path=opt.vlm_name_or_path,
        fast_tokenizer_path=opt.FAST_tokenizer_path,
        window_size=opt.window_size,
        action_horizon=opt.action_horizon,
        accelerator=accelerator,
        process_mode="train",
        max_pad_state_and_action_length=opt.max_pad_state_and_action_length,
        loss_type=opt.loss_type,
        max_length=opt.max_length,
        dataset_sample_ratios=opt.dataset_sample_ratios,
    )
    resolved_dataset_manifest = concat_dataset.resolved_dataset_manifest
    seen_tracker = DatasetSeenTracker(
        concat_dataset,
        accelerator,
        resume_directory=opt.vlm_name_or_path if opt.resume_training else None,
    )
    if accelerator.is_main_process:
        accelerator.print("resolved dataset manifest:")
        accelerator.print(resolved_manifest_json(resolved_dataset_manifest))
        write_resolved_dataset_manifest(
            opt.output_ckpt_dir, resolved_dataset_manifest
        )
    if opt.resume_training:
        validate_resume_manifest(
            resolved_dataset_manifest,
            opt.vlm_name_or_path,
            allow_legacy_missing=opt.allow_legacy_checkpoint_without_manifest,
            allow_legacy_missing_observation_contract=(
                opt.allow_legacy_checkpoint_without_observation_contract
            ),
        )
    accelerator.print("len(concat_dataset):", len(concat_dataset))

    dataloader_num_workers = resolve_dataloader_num_workers(
        concat_dataset, opt.dataloader_num_workers
    )
    accelerator.print("dataloader workers:", dataloader_num_workers)
    dataloader = create_dataloader_for_concat(
        concat_dataset,
        batch_size_per_device=opt.per_device_train_batch_size,
        num_processes=accelerator.num_processes,
        num_workers=dataloader_num_workers,
        prefetch_factor=3,
        seed=opt.seed,
    )
    epoch_sampler = dataloader.batch_sampler
    accelerator.wait_for_everyone()
    accelerator.print("len(dataloader):", len(dataloader))

    if opt.action_expert_name_or_path:
        accelerator.print(f"using a pre-trained action expert from {opt.action_expert_name_or_path}, thus we will load its configurations.")
    else:
        accelerator.print(
            "no Action Expert weights requested; reusing a saved config when "
            "available and allocating random weights only for action-capable modes."
        )
    action_expert_config = resolved_action_expert_config.config
    accelerator.print(
        "resolved Action Expert config: "
        f"path={resolved_action_expert_config.path}, "
        f"source_action_horizon={resolved_action_expert_config.source_action_horizon}, "
        f"resolved_action_horizon={action_expert_config.action_horizon}, "
        "action_horizon_overridden="
        f"{resolved_action_expert_config.action_horizon_overridden}, "
        f"source_sha256={resolved_action_expert_config.source_sha256}, "
        f"resolved_sha256={resolved_action_expert_config.parsed_sha256}"
    )

    if opt.use_lora:
        lora_args = {
            "use_lora": opt.use_lora,
            "target_modules": opt.target_modules,
            "r": opt.r,
            "lora_alpha": opt.lora_alpha,
            "lora_dropout": opt.lora_dropout
        }
    else:
        lora_args = None
    
    # initialize ZR-0 model
    model = ZR0Model(
        vlm_name_or_path = opt.vlm_name_or_path,
        action_expert_name_or_path = opt.action_expert_name_or_path,
        action_expert_config = action_expert_config,
        tune_vlm = opt.tune_vlm,
        tune_action_expert = opt.tune_action_expert,
        detach_vlm_outputs_for_action_expert = opt.detach_vlm_outputs_for_action_expert,
        lora_args = lora_args,
        use_difference_query = opt.use_difference_query,
        num_difference_queries = opt.num_difference_queries,
        vlm_attention_backend = opt.vlm_attention_backend,
        loss_type=opt.loss_type,
        checkpoint_load_purpose=getattr(opt, "checkpoint_load_purpose", None),
        resume_training=getattr(opt, "resume_training", False),
        action_expert_config_path=getattr(opt, "action_expert_config_path", None),
    )
    model.resolved_dataset_manifest = resolved_dataset_manifest
    model.action_expert_source_action_horizon = int(
        resolved_action_expert_config.source_action_horizon
    )
    model.action_expert_config_source_bytes = (
        resolved_action_expert_config.path.read_bytes()
    )
    model.action_expert_config_source_sha256 = (
        resolved_action_expert_config.source_sha256
    )
    accelerator.wait_for_everyone()

    ownership_counts = validate_trainable_parameter_ownership(model)
    accelerator.print("trainable parameter ownership:", ownership_counts)
    if accelerator.is_main_process:
        initialization_manifest_path = write_initialization_manifest(
            model, opt, ownership_counts, resolved_action_expert_config
        )
        accelerator.print(
            "initialization manifest:", initialization_manifest_path
        )

    if accelerator.is_main_process:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters in model: {total_params}")

        total_backbone_params = sum(p.numel() for p in model.backbone.parameters())
        print(f"Total parameters in model.backbone: {total_backbone_params}")

        total_action_expert_params = sum(
            p.numel() for p in model.action_expert.parameters()
        ) if model.action_expert is not None else 0
        print(f"Total parameters in model.action_expert: {total_action_expert_params}")

        total_dit_params = sum(
            p.numel() for p in model.action_expert.dit.parameters()
        ) if model.action_expert is not None else 0
        print(f"Total parameters in model.action_expert.DiT: {total_dit_params}")

        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        trainable_ratio = trainable_params / total_params

        print(f"# Trainable parameters: {trainable_params}")
        print(f"Trainable ratio: {trainable_ratio:.2%}")

    if accelerator.is_main_process:
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f"Trainable parameter: {name} | shape: {param.shape}")

    accelerator.wait_for_everyone()
    # '''grouped lr (start) '''
    # param_groups = []
    # if opt.tune_vlm:
    #     param_groups.append({"params": [p for p in model.backbone.parameters()], "lr": opt.vlm_peak_learning_rate, "weight_decay": 1e-3, "name": "backbone"})
    # if opt.tune_action_expert:
    #     param_groups.append({"params": [p for p in model.action_expert.parameters()], "lr": opt.action_expert_peak_learning_rate, "weight_decay": 1e-5, "name": "action_expert"})
    # # load AdamW optimizer
    # optimizer = AdamW(
    #     # model.parameters(),
    #     param_groups,
    #     # lr = opt.vlm_peak_learning_rate, # not working
    #     betas = (0.95, 0.9995), # (0.9, 0.95),
    #     eps = 1e-6
    # )
    # for i, group in enumerate(optimizer.param_groups):
    #     accelerator.print(f"Group {i} ({group.get('name', 'unnamed')}): lr = {group['lr']}")
    # '''grouped lr (end) '''

    '''single lr (start)'''
    optimizer = build_adamw_optimizer(
        model,
        learning_rate=opt.peak_learning_rate,
        beta1=opt.adam_beta1,
        beta2=opt.adam_beta2,
        epsilon=opt.adam_epsilon,
    )
    '''single lr (end)'''
    
    epoch_optimizer_steps = math.ceil(opt.epochs * math.ceil(len(concat_dataset) / total_batch_size))
    num_total_batches = effective_total_optimizer_steps(
        epoch_optimizer_steps, opt.max_train_steps
    )
    warmup_steps = calculate_warmup_steps(
        num_total_batches, accelerator.num_processes, opt.warmup_ratio
    )

    wandb_logger = WandbTrainingLogger(
        accelerator,
        project=opt.wandb_project,
        run_name=opt.wandb_run_name,
        run_id=opt.wandb_run_id,
        resume=opt.wandb_resume,
        log_dir=opt.wandb_dir,
        group=opt.wandb_group,
        tags=opt.wandb_tags,
        failure_policy=opt.wandb_failure_policy,
        pending_capacity=opt.wandb_pending_capacity,
        retry_base_steps=opt.wandb_retry_base_steps,
        retry_max_steps=opt.wandb_retry_max_steps,
        finish_max_attempts=opt.wandb_finish_max_attempts,
        finish_timeout_seconds=opt.wandb_finish_timeout_seconds,
        config={
            **vars(opt),
            "global_batch_size": total_batch_size,
            "num_processes": accelerator.num_processes,
            "num_total_steps": num_total_batches,
            "warmup_steps": warmup_steps // accelerator.num_processes,
        },
    )
    accelerator.wait_for_everyone()

    if opt.lr_scheduler == "cosine":
        # learning rate scheduler (linear warm up and cosine decay)
        lr_scheduler = get_cosine_with_min_lr_schedule_with_warmup_lr_rate(
            optimizer = optimizer,
            num_warmup_steps = warmup_steps,
            num_training_steps = num_total_batches * accelerator.num_processes,
            min_lr_rate = opt.min_lr_rate
        )
    elif opt.lr_scheduler == "constant":
        # learning rate scheduler (linear warm up and keeping constant)
        lr_scheduler = get_constant_schedule_with_warmup(
            optimizer, 
            num_warmup_steps = warmup_steps
        )
    else:
        raise ValueError("lr_scheduler should be in [cosine, constant].")

    accelerator.wait_for_everyone()
    optimizer, model, dataloader, lr_scheduler = accelerator.prepare(
        optimizer, model, dataloader, lr_scheduler
    )
    actual_gradient_accumulation_steps = accelerator.gradient_accumulation_steps
    actual_micro_batch = opt.per_device_train_batch_size
    if accelerator.distributed_type == DistributedType.DEEPSPEED:
        engine_gradient_accumulation_steps = int(
            model.gradient_accumulation_steps()
        )
        engine_micro_batch = int(model.train_micro_batch_size_per_gpu())
        if engine_gradient_accumulation_steps != opt.gradient_accumulation_steps:
            raise RuntimeError(
                "DeepSpeed gradient accumulation mismatch after prepare: "
                f"engine={engine_gradient_accumulation_steps}, "
                f"requested={opt.gradient_accumulation_steps}"
            )
        if engine_micro_batch != opt.per_device_train_batch_size:
            raise RuntimeError(
                "DeepSpeed micro-batch mismatch after prepare: "
                f"engine={engine_micro_batch}, "
                f"requested={opt.per_device_train_batch_size}"
            )
        actual_gradient_accumulation_steps = engine_gradient_accumulation_steps
        actual_micro_batch = engine_micro_batch
    actual_global_batch = (
        accelerator.num_processes
        * actual_micro_batch
        * actual_gradient_accumulation_steps
    )
    if actual_global_batch != total_batch_size:
        raise RuntimeError(
            "prepared global batch size changed: "
            f"actual={actual_global_batch}, requested={total_batch_size}"
        )
    accelerator.print(
        "prepared training scale: "
        f"world_size={accelerator.num_processes}, "
        f"micro_batch={actual_micro_batch}, "
        f"gradient_accumulation_steps={actual_gradient_accumulation_steps}, "
        f"nominal_global_batch={actual_global_batch}"
    )
    if accelerator.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(accelerator.device)
    metrics_log = None
    if accelerator.is_main_process:
        metrics_log = open(
            os.path.join(opt.output_ckpt_dir, "training_metrics.jsonl"),
            "a",
            encoding="utf-8",
            buffering=1,
        )

    global_completed_steps = 0
    # set model to the train() mode
    model.train()
    
    if opt.resume_training:
        old_output_ckpt_dir = os.path.dirname(opt.vlm_name_or_path)
        global_completed_steps = resume_model_optimizer_scheduler(
            model,
            old_output_ckpt_dir,
            lr_scheduler,
        )
        
        resume_epoch, resume_batch_idx = resume_data_position(
            global_completed_steps=global_completed_steps,
            dataloader_length=len(dataloader),
            gradient_accumulation_steps=accelerator.gradient_accumulation_steps,
        )

        accelerator.print("resume epoch:", resume_epoch)
        accelerator.print("resume batch index:", resume_batch_idx)
        accelerator.print("resume training from {} steps.".format(global_completed_steps))

        accelerator.print("resumed lr scheduler state dict:", lr_scheduler.state_dict())

    accelerator.wait_for_everyone()
    st = time.time()
    reached_max_train_steps = global_completed_steps >= num_total_batches
    if reached_max_train_steps:
        accelerator.print(
            f"Training is already complete at step {global_completed_steps}; "
            f"effective limit is {num_total_batches}."
        )
    for epoch in range(opt.epochs):
        if reached_max_train_steps:
            break
        # set the epoch into each dataset
        for ds in concat_dataset.datasets:
            ds.set_epoch(epoch)
        epoch_sampler.set_epoch(epoch)

        if opt.resume_training and resume_epoch > epoch:
            accelerator.print("skip {}-th epoch".format(epoch))
            accelerator.wait_for_everyone()
            continue

        accelerator.print(f"{epoch=}")

        def resumed_batches():
            for batch_idx, batch in enumerate(dataloader):
                if opt.resume_training and should_skip_resumed_batch(
                    epoch=epoch,
                    batch_idx=batch_idx,
                    resume_epoch=resume_epoch,
                    resume_batch_idx=resume_batch_idx,
                ):
                    accelerator.print("skip {}-th batch".format(batch_idx))
                    continue
                yield batch_idx, batch

        for indexed_window in iter_optimizer_step_windows(
            resumed_batches(), accelerator.gradient_accumulation_steps
        ):
            first_batch_idx, first_batch = indexed_window[0]
            seen_tracker.update([batch for _, batch in indexed_window])
            if accelerator.is_main_process and first_batch_idx == 0 and epoch == 0:
                integrity_check(first_batch, model.backbone.processor)
            training_progress = global_completed_steps / num_total_batches
            next_global_step = global_completed_steps + 1
            collect_training_diagnostics = opt.log_training_diagnostics and (
                next_global_step == 1
                or next_global_step % opt.logging_steps == 0
                or next_global_step >= num_total_batches
            )
            output_metrics = run_optimizer_step_window(
                model=model,
                batches=[batch for _, batch in indexed_window],
                accelerator=accelerator,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                training_progress=training_progress,
                loss_type=opt.loss_type,
                vlm_loss_weight=opt.vlm_loss_weight,
                action_expert_loss_weight=opt.action_expert_loss_weight,
                next_global_step=next_global_step,
                collect_training_diagnostics=collect_training_diagnostics,
            )
            for metric_name, metric_value in output_metrics.items():
                if metric_name.endswith("grad_norm"):
                    assert_all_finite(
                        accelerator,
                        metric_value,
                        metric_name.replace("_", " "),
                        next_global_step,
                    )

            global_grad_norm = model.get_global_grad_norm()
            if global_grad_norm is not None:
                assert_all_finite(
                    accelerator,
                    global_grad_norm,
                    "global gradient norm",
                    next_global_step,
                )

            # Only synchronized optimizer boundaries advance the durable training step.
            if accelerator.is_main_process:
                global_completed_steps += 1
            global_completed_steps = synchronize_global_step(
                global_completed_steps, accelerator
            )
            reached_max_train_steps = global_completed_steps >= num_total_batches

            do_save = global_completed_steps > 0 and (
                global_completed_steps % opt.save_step_interval == 0
                or reached_max_train_steps
            )
            if do_save:
                accelerator.wait_for_everyone()
                save_model(accelerator, model, opt.output_ckpt_dir, f"step-{global_completed_steps}")
                accelerator.wait_for_everyone()
                if opt.save_optimizer_and_lr_states:
                    checkpoint_model_optimizer_scheduler(model, opt.output_ckpt_dir, global_completed_steps, lr_scheduler, accelerator)
                seen_tracker.save(
                    os.path.join(opt.output_ckpt_dir, f"step-{global_completed_steps}"),
                    epoch=epoch,
                    global_step=global_completed_steps,
                )
                if opt.save_optimizer_and_lr_states:
                    seen_tracker.save(
                        os.path.join(opt.output_ckpt_dir, "latest-model-optimizer-lr"),
                        epoch=epoch,
                        global_step=global_completed_steps,
                    )
                seen_tracker.save(
                    opt.output_ckpt_dir, epoch=epoch, global_step=global_completed_steps
                )
                accelerator.wait_for_everyone()

            do_log = (
                global_completed_steps == 1
                or global_completed_steps % opt.logging_steps == 0
                or reached_max_train_steps
            )
            if do_log:
                if accelerator.is_main_process and writer is not None:
                    writer.add_scalar('learning-rate', lr_scheduler.get_last_lr()[0], global_completed_steps)
                    if global_grad_norm is not None:
                        writer.add_scalar('grad-norm', global_grad_norm, global_completed_steps)
                    writer.add_scalar('training progress', training_progress, global_completed_steps)

                    for name, value in output_metrics.items():
                        writer.add_scalar(
                            'train-{}'.format(tensorboard_loss_metric_name(name)),
                            tensorboard_loss_value(value),
                            global_completed_steps,
                        )

                mean_metrics = {
                    f"train/{name}": value for name, value in output_metrics.items()
                }
                scalar_metrics = {
                    "train/learning_rate": lr_scheduler.get_last_lr()[0],
                    "train/progress": global_completed_steps / num_total_batches,
                    "train/epoch": epoch,
                }
                if global_grad_norm is not None:
                    scalar_metrics["train/grad_norm"] = float(global_grad_norm)

                wandb_diagnostics = wandb_logger.log(
                    step=global_completed_steps,
                    mean_metrics=mean_metrics,
                    scalar_metrics=scalar_metrics,
                )

                if accelerator.is_main_process:
                    local_record = {
                        "step": global_completed_steps,
                        "epoch": epoch,
                        "learning_rate": lr_scheduler.get_last_lr()[0],
                        **json_scalar_metrics(output_metrics),
                        **wandb_diagnostics,
                    }
                    if writer is not None:
                        for name, value in wandb_diagnostics.items():
                            writer.add_scalar(name.replace("_", "-"), value, global_completed_steps)
                    line = json.dumps(local_record, ensure_ascii=True, sort_keys=True)
                    print(f"optimizer_step_metrics={line}", flush=True)
                    metrics_log.write(line + "\n")

            if reached_max_train_steps:
                break

        if reached_max_train_steps:
            accelerator.wait_for_everyone()
            accelerator.print(
                f"Reached effective max training steps: {num_total_batches}"
            )
            break
        
        accelerator.wait_for_everyone()

        if accelerator.is_main_process:
            accelerator.print(f"Epoch {epoch} finished, saving checkpoint")

        do_epoch_save = (
            (epoch + 1) % opt.save_ckpt_interval == 0
            or epoch == opt.epochs - 1
        )

        if do_epoch_save:
            accelerator.wait_for_everyone()
            save_model(accelerator, model, opt.output_ckpt_dir, f"step-{global_completed_steps}")
            accelerator.wait_for_everyone()
            if opt.save_optimizer_and_lr_states:
                checkpoint_model_optimizer_scheduler(model, opt.output_ckpt_dir, global_completed_steps, lr_scheduler, accelerator)
            seen_tracker.save(
                os.path.join(opt.output_ckpt_dir, f"step-{global_completed_steps}"),
                epoch=epoch,
                global_step=global_completed_steps,
            )
            if opt.save_optimizer_and_lr_states:
                seen_tracker.save(
                    os.path.join(opt.output_ckpt_dir, "latest-model-optimizer-lr"),
                    epoch=epoch,
                    global_step=global_completed_steps,
                )
            seen_tracker.save(
                opt.output_ckpt_dir, epoch=epoch, global_step=global_completed_steps
            )
            accelerator.wait_for_everyone()

    try:
        finish_diagnostics = wandb_logger.finish(exit_code=0)
        if accelerator.is_main_process:
            write_wandb_finish_diagnostics(
                metrics_log,
                writer,
                step=global_completed_steps,
                diagnostics=finish_diagnostics,
            )
    finally:
        if writer is not None:
            writer.close()
        if metrics_log is not None:
            metrics_log.close()

if __name__ == "__main__":
    opt = parse_option()
    train(opt)
