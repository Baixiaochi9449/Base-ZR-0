"""CPU wrapper semantics only, not a distributed DeepSpeed recovery test."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import yaml
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.scheduler import AcceleratedScheduler

from test_stage05_checkpoint_contract import (
    _make_checkpoint, _make_joint_checkpoint, _write_resume_state,
)
from test_stage05_necessary_repairs import _validate_ar, _write_ar_training_state
from train_vla import (
    calculate_warmup_steps, get_cosine_with_min_lr_schedule_with_warmup_lr_rate,
)
from utils.stage05_checkpoint_contract import validate_stage05_resume_artifacts
from utils.training_checkpoint import _load_scheduler_state, checkpoint_model_optimizer_scheduler


def _make_scheduler(world_size):
    production = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "accelerate_configs/accelerate_config.yaml").read_text()
    )
    assert production["num_processes"] == 4
    accelerator = Accelerator(
        cpu=True,
        gradient_accumulation_steps=production["deepspeed_config"]["gradient_accumulation_steps"],
        dataloader_config=DataLoaderConfiguration(even_batches=False),
    )
    parameter = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.AdamW([parameter], lr=1e-5, betas=(0.9, 0.95), eps=1e-8)
    scheduler = get_cosine_with_min_lr_schedule_with_warmup_lr_rate(
        optimizer,
        num_warmup_steps=calculate_warmup_steps(20, world_size, 0.05),
        num_training_steps=20 * world_size,
        min_lr_rate=0.1,
    )
    optimizer = accelerator.prepare_optimizer(optimizer)
    scheduler = accelerator.prepare_scheduler(scheduler)
    assert isinstance(scheduler, AcceleratedScheduler)
    assert scheduler.step_with_optimizer and not scheduler.split_batches
    return SimpleNamespace(
        accelerator=accelerator, parameter=parameter, optimizer=optimizer,
        scheduler=scheduler, world_size=world_size,
    )


def _advance(state):
    state.parameter.grad = torch.ones_like(state.parameter)
    state.optimizer.step()
    # Only the process-count lookup is replaced; the installed wrapper steps.
    with patch("accelerate.scheduler.AcceleratorState",
               return_value=SimpleNamespace(num_processes=state.world_size)):
        state.scheduler.step()
    state.optimizer.zero_grad()


def _save_scheduler_checkpoint(checkpoint, state):
    _write_ar_training_state(checkpoint)
    # Exercise the production save helper; the tiny facade preserves fixture
    # model files while the real wrapper/optimizer states are serialized below.
    facade = SimpleNamespace(save_checkpoint=lambda *args, **kwargs: None,
                             save_pretrained=lambda *args, **kwargs: None)
    with patch("utils.training_checkpoint.CHECKPOINT_TAG", checkpoint.name), \
         patch.object(state.accelerator, "unwrap_model", return_value=facade):
        checkpoint_model_optimizer_scheduler(
            facade, str(checkpoint.parent), 1, state.scheduler, state.accelerator,
        )
    model_path = checkpoint / "mp_rank_00_model_states.pt"
    model_state = torch.load(model_path, weights_only=False)
    model_state["dp_world_size"] = state.world_size
    torch.save(model_state, model_path)
    optimizer_path = checkpoint / "zero_pp_rank_0_mp_rank_00_optim_states.pt"
    optimizer_state = torch.load(optimizer_path, weights_only=False)
    zero_state = optimizer_state["optimizer_state_dict"]
    zero_state["partition_count"] = [state.world_size]
    zero_state["base_optimizer_state"] = state.optimizer.state_dict()
    for rank in range(state.world_size):
        torch.save(optimizer_state, checkpoint / f"zero_pp_rank_{rank}_mp_rank_00_optim_states.pt")


@pytest.fixture(params=["ar", "joint"])
def resume_case(request, tmp_path):
    if request.param == "ar":
        return _make_checkpoint(tmp_path), _validate_ar
    return _make_joint_checkpoint(tmp_path), validate_stage05_resume_artifacts


@pytest.mark.parametrize("world_size", [1, 4])
def test_ar_resume_accepts_saved_scheduler_steps(resume_case, world_size):
    checkpoint, validate = resume_case
    state = _make_scheduler(world_size)
    _advance(state)
    _save_scheduler_checkpoint(checkpoint, state)
    saved = _load_scheduler_state(checkpoint / "scheduler.pt")
    assert saved["last_epoch"] == world_size
    assert saved["_step_count"] == world_size + 1
    assert validate(checkpoint).config.action_horizon == 32


@pytest.mark.parametrize("world_size", [1, 4])
@pytest.mark.parametrize("fault", ["last_epoch", "_step_count", "both_counts", "global_step"])
def test_ar_resume_rejects_independent_step_corruption(resume_case, world_size, fault):
    checkpoint, validate = resume_case
    state = _make_scheduler(world_size)
    _advance(state)
    _save_scheduler_checkpoint(checkpoint, state)
    if fault == "global_step":
        path = checkpoint / "mp_rank_00_model_states.pt"
        payload = torch.load(path, weights_only=False)
        payload["last_global_step"] += 1
        payload["global_steps"] += 1
    else:
        path = checkpoint / "scheduler.pt"
        payload = _load_scheduler_state(path)
        for name in ("last_epoch", "_step_count") if fault == "both_counts" else (fault,):
            payload[name] += world_size
    torch.save(payload, path)
    with pytest.raises(ValueError, match="scheduler/global step mismatch"):
        validate(checkpoint)


@pytest.mark.parametrize("saved_world_size", [None, 0, True, 2])
def test_ar_resume_requires_saved_process_count(resume_case, saved_world_size):
    checkpoint, validate = resume_case
    state = _make_scheduler(1)
    _advance(state)
    _save_scheduler_checkpoint(checkpoint, state)
    path = checkpoint / "mp_rank_00_model_states.pt"
    payload = torch.load(path, weights_only=False)
    if saved_world_size is None:
        del payload["dp_world_size"]
    else:
        payload["dp_world_size"] = saved_world_size
    torch.save(payload, path)
    with pytest.raises(ValueError, match="saved dp_world_size"):
        validate(checkpoint)


@pytest.mark.parametrize("world_size", [1, 4])
def test_ar_scheduler_resume_matches_uninterrupted_next_step(resume_case, world_size):
    checkpoint, validate = resume_case
    control = _make_scheduler(world_size)
    _advance(control)
    _save_scheduler_checkpoint(checkpoint, control)
    validate(checkpoint)

    resumed = _make_scheduler(world_size)
    with torch.no_grad():
        resumed.parameter.copy_(control.parameter)
    saved_optimizer = torch.load(
        checkpoint / "zero_pp_rank_0_mp_rank_00_optim_states.pt", weights_only=False,
    )["optimizer_state_dict"]["base_optimizer_state"]
    resumed.optimizer.load_state_dict(saved_optimizer)
    resumed.scheduler.load_state_dict(_load_scheduler_state(checkpoint / "scheduler.pt"))
    _advance(control)
    _advance(resumed)
    assert resumed.scheduler.state_dict() == control.scheduler.state_dict()
    assert resumed.scheduler.get_last_lr() == control.scheduler.get_last_lr()
    torch.testing.assert_close(resumed.parameter, control.parameter, rtol=0, atol=0)


def test_joint_scheduler_rejects_unscaled_four_process_counts(tmp_path):
    checkpoint = _make_joint_checkpoint(tmp_path)
    _write_resume_state(checkpoint, global_step=1)
    path = checkpoint / "mp_rank_00_model_states.pt"
    payload = torch.load(path, weights_only=False)
    payload["dp_world_size"] = 4
    torch.save(payload, path)
    optimizer_path = checkpoint / "zero_pp_rank_0_mp_rank_00_optim_states.pt"
    optimizer_state = torch.load(optimizer_path, weights_only=False)
    for rank in range(1, 4):
        torch.save(optimizer_state, checkpoint / f"zero_pp_rank_{rank}_mp_rank_00_optim_states.pt")
    with pytest.raises(ValueError, match="scheduler/global step mismatch"):
        validate_stage05_resume_artifacts(checkpoint)
    torch.save({"last_epoch": 4, "_step_count": 5}, checkpoint / "scheduler.pt")
    validate_stage05_resume_artifacts(checkpoint)
