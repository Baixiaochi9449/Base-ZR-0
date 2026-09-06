import os
import random
import numpy as np

import torch


CHECKPOINT_TAG = "latest-model-optimizer-lr"
SCHEDULER_FILENAME = "scheduler.pt"


def capture_rng_state():
    numpy_state = np.random.get_state()
    return {"python": random.getstate(), "torch": torch.get_rng_state(),
            "numpy": (numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else []}


def restore_rng_state(state):
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    numpy_state = state["numpy"]
    np.random.set_state((numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32), *numpy_state[2:]))
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def register_deepspeed_checkpoint_safe_globals() -> tuple[type, ...]:
    from deepspeed.runtime.fp16.loss_scaler import LossScaler
    from deepspeed.runtime.zero.config import ZeroStageEnum
    from deepspeed.utils.tensor_fragment import fragment_address

    safe_globals = (LossScaler, ZeroStageEnum, fragment_address)
    if hasattr(torch.serialization, "add_safe_globals"):
        torch.serialization.add_safe_globals(list(safe_globals))
    return safe_globals


def checkpoint_model_optimizer_scheduler(
    model,
    output_ckpt_dir,
    global_completed_steps,
    lr_scheduler,
    accelerator,
) -> None:
    checkpoint_state = {"last_global_step": int(global_completed_steps)}
    unwrapped_model = accelerator.unwrap_model(model)
    from utils.optical_flow_checkpoint import reject_flow_zero3
    reject_flow_zero3(model=unwrapped_model, accelerator=accelerator)
    if getattr(unwrapped_model, "training_stage", None):
        checkpoint_state["training_data_cursor"] = unwrapped_model.training_data_cursor
    accelerator.print("==> saving model and optimizer <==")
    model.save_checkpoint(
        output_ckpt_dir,
        tag=CHECKPOINT_TAG,
        client_state=checkpoint_state,
        save_latest=False,
    )

    checkpoint_directory = os.path.join(output_ckpt_dir, CHECKPOINT_TAG)
    if getattr(unwrapped_model, "training_stage", None):
        runtime = {"rng": capture_rng_state(), "cursor": unwrapped_model.training_data_cursor,
                   "global_step": int(global_completed_steps),
                   "world_size": accelerator.num_processes,
                   "sampler_contract": unwrapped_model.training_sampler_contract,
                   "skipped_flow_batches": getattr(unwrapped_model, "skipped_flow_batches", 0),
                   "scaler": accelerator.scaler.state_dict() if accelerator.scaler is not None else None}
        torch.save(runtime, os.path.join(checkpoint_directory, f"training_runtime_rank{accelerator.process_index}.pt"))
    accelerator.print("==> saving lr scheduler <==")
    accelerator.save(
        lr_scheduler.state_dict(),
        os.path.join(checkpoint_directory, SCHEDULER_FILENAME),
    )

    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(checkpoint_directory)


def _load_scheduler_state(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def resume_model_optimizer_scheduler(
    model,
    old_output_ckpt_dir,
    lr_scheduler,
) -> int:
    register_deepspeed_checkpoint_safe_globals()
    load_path, checkpoint_state = model.load_checkpoint(
        old_output_ckpt_dir,
        tag=CHECKPOINT_TAG,
    )
    if load_path is None or checkpoint_state is None:
        raise RuntimeError(
            f"failed to load DeepSpeed checkpoint {CHECKPOINT_TAG!r} "
            f"from {old_output_ckpt_dir!r}"
        )

    scheduler_path = os.path.join(
        old_output_ckpt_dir,
        CHECKPOINT_TAG,
        SCHEDULER_FILENAME,
    )
    lr_scheduler.load_state_dict(_load_scheduler_state(scheduler_path))
    return int(checkpoint_state["last_global_step"])


def restore_stage_runtime(directory, model, accelerator, global_step):
    path = os.path.join(directory, f"training_runtime_rank{accelerator.process_index}.pt")
    runtime = _load_scheduler_state(path)
    if runtime["world_size"] != accelerator.num_processes or runtime["global_step"] != global_step:
        raise ValueError("resume runtime world size/global step mismatch")
    if runtime["sampler_contract"] != model.training_sampler_contract:
        raise ValueError("resume sampler seed/batch/accumulation contract mismatch")
    if runtime["scaler"] is not None:
        if accelerator.scaler is None:
            raise ValueError("resume requires saved AMP scaler")
        accelerator.scaler.load_state_dict(runtime["scaler"])
    model.skipped_flow_batches = runtime["skipped_flow_batches"]
    model.training_data_cursor = runtime["cursor"]
    model.pending_resume_rng = runtime["rng"]
    return runtime["cursor"]["epoch"], runtime["cursor"]["batch_idx"]
