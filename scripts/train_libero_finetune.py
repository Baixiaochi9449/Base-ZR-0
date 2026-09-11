"""Opt-in LIBERO initialization checks around the existing production trainer."""

import json
import os
from pathlib import Path
import shutil
import sys
import uuid


TAG = "latest-model-optimizer-lr"


def inventory(directory):
    return {str(p.relative_to(directory)): [p.stat().st_size, p.stat().st_mtime_ns]
            for p in sorted(directory.rglob("*")) if p.is_file()}


def validate_options(options):
    expected = dict(training_stage=None, loss_type="action", tune_vlm=True,
                    tune_action_expert=True, use_difference_query=True,
                    num_difference_queries=32, component_optimizer_groups=False,
                    action_horizon=10, per_device_train_batch_size=16,
                    gradient_accumulation_steps=1, expected_global_batch_size=64,
                    epochs=8, max_train_steps=None, seed=42,
                    vlm_loss_weight=0.0, slot_loss_weight=0.0,
                    optical_flow_loss_weight=0.0, action_expert_loss_weight=1.0,
                    checkpoint_load_purpose="downstream_finetune")
    for key, value in expected.items():
        if getattr(options, key, None) != value:
            raise ValueError(f"LIBERO contract mismatch: {key} must be {value!r}")
    if Path(options.vlm_name_or_path).resolve() != Path(options.action_expert_name_or_path).resolve():
        raise ValueError("VLM, Query and Expert must inherit the same LIBERO initialization")
    if options.init_from_checkpoint or options.resume_from_checkpoint:
        raise ValueError("three-stage loading flags are not valid for this downstream entrypoint")


def verify_model(model, source, compare):
    if model.slot_aux is not None or model.optical_flow_aux is not None:
        raise ValueError("LIBERO action-only must not construct auxiliary Heads")
    if model.training_stage is not None or model.num_difference_queries != 32:
        raise ValueError("unexpected LIBERO model stage or Query count")
    records = {}
    for name, module, weights in (
        ("vlm", model.backbone.model, None),
        ("query", model.backbone.difference_query, "difference_query.safetensors"),
        ("action_expert", model.action_expert, "action_expert.safetensors"),
    ):
        if module is None or not all(p.requires_grad for p in module.parameters()):
            raise ValueError(f"missing or frozen LIBERO component: {name}")
        records[name] = compare(module, source, weights=weights)
    return records


def verify_optimizer(optimizer):
    if len(optimizer.param_groups) != 1 or optimizer.state:
        raise ValueError("LIBERO requires one newly initialized AdamW parameter group")
    group = optimizer.param_groups[0]
    if group["betas"] != (0.9, 0.95) or group["eps"] != 1e-6 or group["weight_decay"] != 0.01:
        raise ValueError("LIBERO AdamW hyperparameters changed")


def verify_update_position(engine, scheduler, previous_step):
    state = scheduler.state_dict()
    if engine.global_steps != previous_step or state["last_epoch"] != previous_step * 4:
        raise ValueError("LIBERO engine/scheduler did not start at the expected update")
    if state["_step_count"] != previous_step * 4 + 1:
        raise ValueError("LIBERO legacy scheduler counter mismatch")
    return {"global_step_before": previous_step, "scheduler_last_epoch": state["last_epoch"],
            "scheduler_step_count": state["_step_count"], "lr": scheduler.get_last_lr()}


def seal_checkpoint(output, step):
    source = output / TAG
    destination = output / "recovery_checkpoints" / f"step-{step:06d}"
    before = inventory(source)
    pending = destination.parent / (".pending-" + uuid.uuid4().hex)
    pending.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, pending / TAG)
    if before != inventory(source) or before != inventory(pending / TAG):
        raise ValueError(f"LIBERO checkpoint changed during archive: {pending}")
    receipt = dict(step=step, source=str(source), files=before, checkpoint_kind="action_only")
    (pending / "checkpoint_complete.json").write_text(json.dumps(receipt, indent=2) + "\n")
    pending.rename(destination)
    return str(destination / TAG)


def main():
    runtime = Path(os.environ["ZR0_RUNTIME_ROOT"]).resolve()
    sys.path[:0] = [str(runtime), str(runtime / "lerobot")]
    os.chdir(runtime)
    import torch.distributed as distributed
    import train_vla as trainer
    from utils.three_stage_sources import compare_component

    options = trainer.parse_option()
    validate_options(options)
    output = Path(options.output_ckpt_dir)
    original_optimizer = trainer.build_adamw_optimizer
    original_window = trainer.run_optimizer_step_window
    original_checkpoint = trainer.checkpoint_model_optimizer_scheduler
    checked = False

    def build_optimizer(*args, **kwargs):
        optimizer = original_optimizer(*args, **kwargs)
        verify_optimizer(optimizer)
        return optimizer

    def window(**kwargs):
        nonlocal checked
        accelerator = kwargs["accelerator"]
        if not checked:
            previous = kwargs["next_global_step"] - 1
            errors = [None] * accelerator.num_processes
            local_error = None
            try:
                counters = verify_update_position(kwargs["model"], kwargs["lr_scheduler"], previous)
                if not options.resume_training and previous != 0:
                    raise ValueError("fresh LIBERO initialization retained pretraining global step")
                if accelerator.is_main_process:
                    evidence = verify_model(accelerator.unwrap_model(kwargs["model"]),
                                            options.vlm_name_or_path, compare_component)
                    evidence.update(counters, tolerance=0, runtime_root=str(runtime),
                                    new_optimizer_before_prepare=True, resume=options.resume_training)
                    with (output / "libero_initialization_verified.json").open("x") as stream:
                        json.dump(evidence, stream, indent=2)
            except Exception as error:
                local_error = repr(error)
            distributed.all_gather_object(errors, local_error)
            if any(errors):
                raise RuntimeError(f"LIBERO verification failed before optimizer update: {errors}")
            if accelerator.is_main_process:
                (output / "optimizer_windows_started").touch(exist_ok=False)
            checked = True
        return original_window(**kwargs)

    def checkpoint(model, output_ckpt_dir, step, scheduler, accelerator):
        original_checkpoint(model, output_ckpt_dir, step, scheduler, accelerator)
        accelerator.wait_for_everyone()
        result = [None]
        if accelerator.is_main_process:
            try:
                result[0] = {"checkpoint": seal_checkpoint(Path(output_ckpt_dir), step)}
            except Exception as error:
                result[0] = {"error": repr(error)}
        distributed.broadcast_object_list(result, src=0)
        if "error" in result[0]:
            raise RuntimeError(f"LIBERO checkpoint archive failed: {result[0]['error']}")
        accelerator.print(f"libero_checkpoint_preserved={json.dumps(result[0])}")

    trainer.build_adamw_optimizer = build_optimizer
    trainer.run_optimizer_step_window = window
    trainer.checkpoint_model_optimizer_scheduler = checkpoint
    trainer.train(options)


if __name__ == "__main__":
    main()
