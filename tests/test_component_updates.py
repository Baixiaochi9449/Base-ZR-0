"""Momentum, masked objectives, and native partition-boundary update regressions."""

import copy
import json
from types import SimpleNamespace

import pytest
import torch

from utils.component_updates import ComponentUpdateGuard, component_activity
from utils.bounded_validation import execution_limit, consecutive_skips, check_next_update


def equal_state(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            equal_state(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            equal_state(x, y)
    else:
        assert a == b


@pytest.mark.parametrize("partitioned", [False, True])
@pytest.mark.parametrize("diagnostics_mode", [None, "full", "inactive"])
def test_momentum_inactive_first_active_reactivation_and_resume(partitioned, diagnostics_mode, tmp_path):
    parameters = [torch.nn.Parameter(torch.ones(3)) for _ in range(3)]
    groups = [dict(params=[p], component=n) for p, n in zip(parameters,
              ("difference_query", "slot_aux", "optical_flow_aux"))]
    optimizer = torch.optim.AdamW(groups, lr=.01, weight_decay=.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: .9 ** step)
    log_path = tmp_path / "component_updates_rank0.jsonl"
    guard = ComponentUpdateGuard(optimizer, scheduler, diagnostics=True, log_path=log_path,
                                 diagnostics_mode=diagnostics_mode)
    def window(slot, flow):
        guard.begin(dict(ar_active_token_count=1, slot_active_supervision_count=slot,
                         flow_active_sample_count=flow), global_step=scheduler.last_epoch)
        for p in parameters:
            p.grad = torch.ones_like(p) if slot else torch.zeros_like(p)
        original = optimizer.param_groups
        if partitioned:
            for group in original:
                optimizer.param_groups = [group]
                optimizer.step()
            optimizer.param_groups = original
        else:
            optimizer.step()
        scheduler.step()
        return guard.finish(applied=True)
    first = window(1, 0)
    assert first["optical_flow_aux_optimizer_step_max"] == 0
    assert parameters[2] not in optimizer.state
    before = parameters[1].detach().clone(), copy.deepcopy(optimizer.state[parameters[1]])
    shared = parameters[0].detach().clone()
    inactive = window(0, 1)
    equal_state(before, (parameters[1], optimizer.state[parameters[1]]))
    assert not torch.equal(shared, parameters[0])
    assert inactive["slot_aux_parameter_delta"] == inactive["slot_aux_optimizer_state_delta"] == 0
    assert inactive["slot_aux_lr_after"] == first["slot_aux_lr_after"]
    assert inactive["optical_flow_aux_optimizer_step_max"] == 1
    # Same-stage optimizer/scheduler restoration must retain per-group time.
    saved_opt, saved_sched = copy.deepcopy(optimizer.state_dict()), copy.deepcopy(scheduler.state_dict())
    optimizer.load_state_dict(saved_opt)
    scheduler.load_state_dict(saved_sched)
    for handle in guard.handles:
        handle.remove()
    guard = ComponentUpdateGuard(optimizer, scheduler, diagnostics=True, log_path=log_path,
                                 diagnostics_mode=diagnostics_mode)
    active = window(1, 1)
    assert active["slot_aux_optimizer_step_max"] == 2
    assert active["slot_aux_lr_used"] == pytest.approx(.01 * .9 ** 2)
    assert active["difference_query_optimizer_step_max"] == 3
    assert scheduler.last_epoch == 3
    records = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert [row["global_step_after"] for row in records] == [1, 2, 3]
    assert not records[1]["slot_aux_update_applied"] and records[1]["slot_aux_optimizer_state_delta"] == 0


def test_zero_numerical_gradient_is_still_active():
    parameter = torch.nn.Parameter(torch.ones(2))
    optimizer = torch.optim.AdamW([dict(params=[parameter], component="slot_aux")], lr=.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1)
    guard = ComponentUpdateGuard(optimizer, scheduler)
    guard.begin({"slot_active_supervision_count": 1})
    parameter.grad = torch.zeros_like(parameter)
    optimizer.step()
    scheduler.step()
    assert guard.finish(applied=True)["slot_aux_update_applied"]
    assert optimizer.state[parameter]["step"] == 1


def test_global_skip_retains_states_and_no_independent_head_clock():
    p = torch.nn.Parameter(torch.ones(1))
    opt = torch.optim.AdamW([dict(params=[p], component="action_expert")])
    scheduler = torch.optim.lr_scheduler.StepLR(opt, 1)
    guard = ComponentUpdateGuard(opt, scheduler)
    before = copy.deepcopy(opt.state_dict())
    guard.begin({"fm_active_element_count": 4})
    assert not guard.finish(applied=False)["action_expert_update_applied"]
    equal_state(before, opt.state_dict())
    assert scheduler.last_epoch == 0
    assert component_activity({"fm_active_element_count": 1}, detach_action=True) == {
        "vlm": False, "difference_query": False, "action_expert": True,
        "slot_aux": False, "optical_flow_aux": False}


def test_independent_execution_cap_and_skip_limit():
    options = SimpleNamespace(save_and_exit_after_updates=50, max_consecutive_skipped_windows=20,
        bounded_three_stage_validation=True, training_stage="stage1_ar", max_train_steps=100,
        save_optimizer_and_lr_states=True, component_optimizer_groups=True,
        wandb_project="validation", wandb_failure_policy="required")
    assert execution_limit(options, 100) == 50
    options.save_and_exit_after_updates = 100
    assert execution_limit(options, 150000) == 100
    options.max_train_steps = 10000
    with pytest.raises(ValueError, match="scheduler length 100"):
        execution_limit(options, 10000)
    with pytest.raises(RuntimeError, match="budget exhausted"):
        check_next_update(100, 100)
    assert consecutive_skips(19, True, 20) == 0
    with pytest.raises(RuntimeError, match="20 consecutive"):
        consecutive_skips(19, False, 20)


@pytest.mark.parametrize("diagnostics_mode", ["full", "inactive"])
def test_production_loss_window_preserves_unsupervised_slot_momentum(diagnostics_mode):
    from test_slot_integration import tiny_slot_model, sample, run_window
    from train_vla import build_adamw_optimizer
    torch.set_num_threads(2)
    model = tiny_slot_model("stage3_joint")
    model.backbone.model.requires_grad_(True)
    optimizer = build_adamw_optimizer(model, learning_rate=.001, beta1=.9, beta2=.95,
                                     epsilon=1e-6, component_groups=True)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: .9 ** step)
    model.component_update_guard = ComponentUpdateGuard(optimizer, scheduler, diagnostics=True,
                                                        diagnostics_mode=diagnostics_mode)
    first = run_window(model, [sample(True)], optimizer, scheduler)
    slot = list(model.slot_aux.parameters())
    before = [(p.detach().clone(), copy.deepcopy(optimizer.state.get(p, {}))) for p in slot]
    query = model.backbone.difference_query.weight.detach().clone()
    second = run_window(model, [sample(False), sample(False)], optimizer, scheduler)
    equal_state(before, [(p, optimizer.state.get(p, {})) for p in slot])
    assert first["slot_aux_update_applied"] and not second["slot_aux_update_applied"]
    assert second["optimizer_update_applied"] and second["action_expert_update_applied"]
    assert second["vlm_update_applied"] and second["difference_query_update_applied"]
    assert not torch.equal(query, model.backbone.difference_query.weight)
    assert scheduler.last_epoch == 2


def test_inactive_mode_never_snapshots_active_components(monkeypatch):
    import utils.component_updates as updates
    parameters = [torch.nn.Parameter(torch.ones(3)) for _ in range(2)]
    optimizer = torch.optim.AdamW([dict(params=[p], component=name) for p, name in
                                  zip(parameters, ("vlm", "slot_aux"))], lr=.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1)
    original, snapshots = updates._snapshot, []

    def snapshot(group, native):
        assert group["component"] == "slot_aux"
        snapshots.append(group["component"])
        return original(group, native)

    monkeypatch.setattr(updates, "_snapshot", snapshot)
    guard = ComponentUpdateGuard(optimizer, scheduler, diagnostics=True, diagnostics_mode="inactive")
    guard.begin({"ar_active_token_count": 1})
    for p in parameters:
        p.grad = torch.ones_like(p)
    optimizer.step()
    scheduler.step()
    metrics = guard.finish(applied=True)
    assert snapshots == ["slot_aux", "slot_aux"]
    assert "vlm_parameter_delta" not in metrics
    assert not any(name.endswith("native_grad_norm") for name in metrics)
    assert metrics["slot_aux_parameter_delta"] == metrics["slot_aux_optimizer_state_delta"] == 0
    assert metrics["vlm_update_applied"] and not metrics["slot_aux_update_applied"]


@pytest.mark.parametrize("partitioned", [False, True])
def test_diagnostic_modes_have_exact_same_updates_and_resumed_state(partitioned):
    names = ("vlm", "difference_query", "action_expert", "slot_aux", "optical_flow_aux")

    def setup(mode):
        parameters = [torch.nn.Parameter(torch.arange(4, dtype=torch.float32) + i + 1)
                      for i in range(len(names))]
        optimizer = torch.optim.AdamW([dict(params=[p], component=name) for p, name in zip(parameters, names)],
                                     lr=.002, betas=(.9, .95), eps=1e-6, weight_decay=.01)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: .95 ** step)
        guard = ComponentUpdateGuard(optimizer, scheduler, diagnostics=True, diagnostics_mode=mode)
        return parameters, optimizer, scheduler, guard

    full, inactive = setup("full"), setup("inactive")
    windows = [(1, 1, 1, 0, True), (1, 0, 0, 1, True), (0, 0, 0, 0, False),
               (1, 1, 1, 1, False), (1, 1, 1, 1, True), (0, 1, 0, 0, True)]
    for index, (ar, fm, slot, flow, applied) in enumerate(windows):
        metrics = []
        for parameters, optimizer, scheduler, guard in (full, inactive):
            if index == 4:
                optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
                scheduler.load_state_dict(copy.deepcopy(scheduler.state_dict()))
            guard.begin(dict(ar_active_token_count=ar, fm_active_element_count=fm,
                             slot_active_supervision_count=slot, flow_active_sample_count=flow))
            for p in parameters:
                p.grad = p.detach().square() * (.001 if index % 2 else 0.)
            if applied:
                original = optimizer.param_groups
                if partitioned:
                    for group in original:
                        optimizer.param_groups = [group]
                        optimizer.step()
                    optimizer.param_groups = original
                else:
                    optimizer.step()
                scheduler.step()
            metrics.append(guard.finish(applied=applied))
        equal_state(full[0], inactive[0])
        equal_state(full[1].state_dict(), inactive[1].state_dict())
        equal_state(full[2].state_dict(), inactive[2].state_dict())
        for key in metrics[1]:
            assert metrics[0][key] == metrics[1][key]


def test_diagnostic_mode_defaults_and_bounded_validation_contract():
    from utils.cli_options import parse_train_options
    from utils.bounded_validation import validate_update_limits
    assert parse_train_options([]).component_update_diagnostics is None
    with pytest.raises(SystemExit) as error:
        parse_train_options(["--component_update_diagnostics", "inactive"])
    assert error.value.code == 2
    options = SimpleNamespace(save_and_exit_after_updates=50, max_consecutive_skipped_windows=20,
        bounded_three_stage_validation=True, training_stage="stage1_ar", max_train_steps=100,
        save_optimizer_and_lr_states=True, component_optimizer_groups=True,
        wandb_project="validation", wandb_failure_policy="required", component_update_diagnostics="inactive")
    with pytest.raises(ValueError, match="requires full component diagnostics"):
        validate_update_limits(options)
    options.component_update_diagnostics = "full"
    validate_update_limits(options)
