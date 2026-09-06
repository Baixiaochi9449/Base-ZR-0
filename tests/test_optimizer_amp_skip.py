"""Real CPU GradScaler skips must not become completed training windows."""

import copy
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from accelerate import Accelerator
from accelerate.optimizer import AcceleratedOptimizer
from accelerate.utils import DistributedType

from test_optical_flow_aux import tiny_batch, tiny_stage_model
from test_optical_flow_review import fake_logger
from train_vla import advance_global_step, json_scalar_metrics, run_optimizer_step_window
from utils.optical_flow_checkpoint import checkpoint_stage_metadata, read_stage_training_state
from utils.optimizer_step_loss import OptimizerWindowStep, optimizer_update_applied


@pytest.mark.parametrize("stage", ["stage1_ar", "stage2_aux", "stage3_joint"])
@pytest.mark.parametrize("scheduler_kind", ["step", "constant", "cosine", "accelerated"])
def test_real_cpu_amp_overflow_then_success(stage, scheduler_kind):
    torch.manual_seed(42)
    model = tiny_stage_model(stage)
    accelerator = Accelerator(cpu=True)
    original_scaler = accelerator.scaler
    accelerator.scaler = torch.amp.GradScaler("cpu", init_scale=128.)
    native = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.1)
    if scheduler_kind == "constant":
        scheduler = torch.optim.lr_scheduler.LambdaLR(native, lambda _: 1.)
    elif scheduler_kind == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(native, T_max=10)
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(native, step_size=1, gamma=.9)
    optimizer = AcceleratedOptimizer(native, scaler=accelerator.scaler)
    if scheduler_kind == "accelerated":
        from accelerate.scheduler import AcceleratedScheduler
        scheduler = AcceleratedScheduler(scheduler, optimizer)
    global_step = 0
    initial_scheduler = copy.deepcopy(scheduler.state_dict())
    before = [p.detach().clone() for p in model.parameters()]
    logger, run = fake_logger(accelerator)

    def window(data):
        return run_optimizer_step_window(model=model, batches=[data], accelerator=accelerator,
            optimizer=optimizer, lr_scheduler=scheduler, training_progress=0., loss_type=model.loss_type,
            vlm_loss_weight=1., action_expert_loss_weight=1., next_global_step=global_step + 1,
            optical_flow_config=model.optical_flow_config, training_stage=stage)

    try:
        # Loss remains finite; Inf is introduced only after autograd computed a gradient.
        parameter = model.backbone.difference_query.weight
        handle = parameter.register_hook(lambda gradient: torch.full_like(gradient, float("inf")))
        try:
            overflow = window(tiny_batch())
        finally:
            handle.remove()
        assert optimizer.step_was_skipped is True
        assert torch.isfinite(overflow["loss"])
        assert overflow["optimizer_skip_reason"] == "amp_overflow"
        assert overflow["optimizer_update_skipped"] is True
        assert optimizer_update_applied(overflow) is False
        assert all(torch.equal(a, b) for a, b in zip(before, model.parameters()))
        assert scheduler.state_dict() == initial_scheduler
        global_step = advance_global_step(global_step, overflow, accelerator)
        assert global_step == 0
        state = checkpoint_stage_metadata(model)
        assert state["completed_optimizer_windows"] == 0
        assert not any(state[key] for key in ("ar_loss_computed", "flow_loss_computed", "slot_loss_computed", "fm_loss_computed"))
        logger.log(step=global_step, mean_metrics=overflow, scalar_metrics={})
        assert run.logs[-1][0] == json_scalar_metrics(overflow)

        if stage == "stage2_aux":
            assert overflow["flow_loss_computed"] is True  # Per-window evidence, not cumulative state.
            data = dict(tiny_batch(), flow_supervision_available=torch.tensor([False]), flow_target={}, flow_valid_mask={})
            with patch.object(accelerator, "backward", side_effect=AssertionError("empty backward")), \
                 patch.object(optimizer, "step", side_effect=AssertionError("empty optimizer step")), \
                 patch.object(scheduler, "step", side_effect=AssertionError("empty scheduler step")):
                empty = window(data)
            assert empty["optimizer_skip_reason"] == "no_supervision"
            assert empty["optimizer_update_applied"] is False and empty["flow_loss_computed"] is False
            assert advance_global_step(global_step, empty, accelerator) == 0
            assert checkpoint_stage_metadata(model) == state
            assert all(torch.equal(a, b) for a, b in zip(before, model.parameters()))
            assert scheduler.state_dict() == initial_scheduler
            logger.log(step=0, mean_metrics=empty, scalar_metrics={})
            assert run.logs[-1][0]["optimizer_skip_reason"] == "no_supervision"

        finite = window(tiny_batch())
        assert optimizer.step_was_skipped is False
        assert optimizer_update_applied(finite) is True
        assert finite["optimizer_skip_reason"] == "none"
        assert any(not torch.equal(a, b) for a, b in zip(before, model.parameters()))
        assert scheduler.state_dict()["last_epoch"] == initial_scheduler["last_epoch"] + 1
        global_step = advance_global_step(global_step, finite, accelerator)
        assert global_step == 1
        state = checkpoint_stage_metadata(model)
        assert state["completed_optimizer_windows"] == 1
        assert state["ar_loss_computed"] is (stage != "stage2_aux")
        assert state["flow_loss_computed"] is (stage != "stage1_ar")
        assert state["fm_loss_computed"] is (stage == "stage3_joint")
        assert state["slot_loss_computed"] is False
        logger.log(step=1, mean_metrics=finite, scalar_metrics={})
        assert run.logs[-1][0]["optimizer_update_applied"] is True
    finally:
        accelerator.scaler = original_scaler


@pytest.mark.parametrize("applied,overflow", [(False, True), (False, False), (True, False)])
def test_deepspeed_engine_result_and_scheduler_ownership_simulation(applied, overflow):
    parameter = torch.nn.Parameter(torch.ones(1))
    native = torch.optim.SGD([parameter], lr=.1)
    scheduler = torch.optim.lr_scheduler.StepLR(native, 1)
    engine = SimpleNamespace(was_step_applied=lambda: applied,
                             optimizer=SimpleNamespace(overflow=overflow), lr_scheduler=scheduler)
    accelerator = SimpleNamespace(distributed_type=DistributedType.DEEPSPEED, device=torch.device("cpu"),
                                  num_processes=1, reduce=lambda value, **_: value)
    tracker = OptimizerWindowStep(model=engine, optimizer=object(), lr_scheduler=scheduler, accelerator=accelerator)
    before = scheduler.last_epoch
    # Simulate the engine-owned boundary; the trainer must not step this scheduler twice.
    if applied:
        parameter.grad = torch.ones_like(parameter)
        native.step()
        scheduler.step()
    result = tracker.finish()
    assert result["optimizer_update_applied"] is applied
    assert scheduler.last_epoch == before + int(applied)
    assert result["optimizer_skip_reason"] == ("none" if applied else "amp_overflow" if overflow else "optimizer_step_skipped")


def test_engine_scheduler_cannot_advance_on_skip():
    parameter = torch.nn.Parameter(torch.ones(1))
    native = torch.optim.SGD([parameter], lr=.1)
    scheduler = torch.optim.lr_scheduler.StepLR(native, 1)
    engine = SimpleNamespace(was_step_applied=lambda: False, optimizer=SimpleNamespace(overflow=True),
                             lr_scheduler=scheduler)
    accelerator = SimpleNamespace(distributed_type=DistributedType.DEEPSPEED, device=torch.device("cpu"),
                                  num_processes=1, reduce=lambda value, **_: value)
    tracker = OptimizerWindowStep(model=engine, optimizer=object(), lr_scheduler=scheduler, accelerator=accelerator)
    scheduler.last_epoch += 1
    with pytest.raises(RuntimeError, match="scheduler changed during a skipped"):
        tracker.finish()


def test_rank_update_disagreement_cannot_advance_scheduler():
    parameter = torch.nn.Parameter(torch.ones(1))
    native = torch.optim.SGD([parameter], lr=.1)
    scheduler = torch.optim.lr_scheduler.StepLR(native, 1)
    accelerator = SimpleNamespace(distributed_type=DistributedType.MULTI_CPU, device=torch.device("cpu"),
                                  num_processes=2, reduce=lambda value, **_: torch.tensor([1, 0]))
    tracker = OptimizerWindowStep(model=None, optimizer=native, lr_scheduler=scheduler, accelerator=accelerator)
    before = scheduler.state_dict()
    parameter.grad = torch.ones_like(parameter)
    with pytest.raises(RuntimeError, match="results differ across ranks"):
        tracker.finish()
    assert scheduler.state_dict() == before


def test_unknown_backend_and_multiple_optimizers_fail_fast():
    accelerator = Accelerator(cpu=True)
    parameter = torch.nn.Parameter(torch.ones(1))
    native = torch.optim.SGD([parameter], lr=.1)
    scheduler = torch.optim.lr_scheduler.StepLR(native, 1)
    with pytest.raises(RuntimeError, match="one optimizer"):
        OptimizerWindowStep(model=None, optimizer=[native, native], lr_scheduler=scheduler, accelerator=accelerator)
    with pytest.raises(RuntimeError, match="no verified"):
        OptimizerWindowStep(model=None, optimizer=object(), lr_scheduler=scheduler, accelerator=accelerator)
    fake_ds = SimpleNamespace(distributed_type=DistributedType.DEEPSPEED)
    with pytest.raises(RuntimeError, match="status is unknown"):
        OptimizerWindowStep(model=None, optimizer=native, lr_scheduler=scheduler, accelerator=fake_ds)
    with pytest.raises(RuntimeError, match="invalid optimizer step result"):
        optimizer_update_applied({})


def test_v1_checkpoint_does_not_claim_verified_updates():
    model = tiny_stage_model("stage2_aux")
    old = checkpoint_stage_metadata(model)
    old.update(stage_metadata_version=1, flow_loss_computed=True)
    del old["completed_optimizer_windows"]
    with pytest.warns(RuntimeWarning, match="history is unknown"):
        restored = read_stage_training_state(old, model.optical_flow_config)
    assert restored["stage_metadata_version"] == 2
    assert restored["completed_optimizer_windows"] == 0
    assert restored["flow_loss_computed"] is None
    assert restored["legacy_history_unknown"] is True
    assert restored["slot_loss_computed"] is False
