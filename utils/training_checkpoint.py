import os

import torch


CHECKPOINT_TAG = "latest-model-optimizer-lr"
SCHEDULER_FILENAME = "scheduler.pt"


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
    accelerator.print("==> saving model and optimizer <==")
    model.save_checkpoint(
        output_ckpt_dir,
        tag=CHECKPOINT_TAG,
        client_state=checkpoint_state,
        save_latest=False,
    )

    checkpoint_directory = os.path.join(output_ckpt_dir, CHECKPOINT_TAG)
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
