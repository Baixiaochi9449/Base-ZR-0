import copy

import pytest
import torch
from accelerate import Accelerator
from accelerate.state import AcceleratorState
from accelerate.utils import DistributedType
from torch import nn
from transformers import BatchFeature

from model.difference_query import DifferenceQuery
from train_vla import (
    build_adamw_optimizer,
    module_gradient_norms,
    run_optimizer_step_window,
    validate_trainable_parameter_ownership,
)


class _Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Linear(1, 1, bias=False)
        self.difference_query = DifferenceQuery(1, 1, initializer_std=0.0)


class _OwnedObjective(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = _Backbone()
        self.action_expert = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            self.backbone.model.weight.fill_(0.5)
            self.backbone.difference_query.weight.fill_(0.25)
            self.action_expert.weight.fill_(0.75)

    def forward(
        self,
        batch,
        training_progress,
        vlm_loss_weight=1.0,
        action_expert_loss_weight=1.0,
    ):
        del training_progress, vlm_loss_weight, action_expert_loss_weight
        ar_value = (
            self.backbone.model.weight
            + self.backbone.difference_query.weight
        ).square().sum()
        fm_value = self.action_expert.weight.square().sum()
        return BatchFeature(
            {
                "ar_loss_sum": ar_value,
                "ar_loss_count": torch.tensor(1.0),
                "flow_matching_loss_sum": fm_value,
                "flow_matching_loss_count": torch.tensor(1.0),
            }
        )


def _batches():
    batch = {
        "input_ids": torch.zeros((1, 2), dtype=torch.long),
        "labels": torch.tensor([[-100, 1]]),
        "action_mask": torch.ones((1, 1, 1), dtype=torch.bool),
        "data_read_retry_count": torch.tensor([0]),
    }
    return [copy.deepcopy(batch), copy.deepcopy(batch)]


def _run(collect_diagnostics):
    accelerator = Accelerator(cpu=True, gradient_accumulation_steps=2)
    model = _OwnedObjective()
    ownership = validate_trainable_parameter_ownership(model)
    optimizer = build_adamw_optimizer(
        model,
        learning_rate=1e-3,
        beta1=0.9,
        beta2=0.95,
        epsilon=1e-8,
    )
    assert len(optimizer.param_groups) == 1
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
    metrics = run_optimizer_step_window(
        model=model,
        batches=_batches(),
        accelerator=accelerator,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        training_progress=0.0,
        loss_type="vlm_and_action",
        vlm_loss_weight=1.0,
        action_expert_loss_weight=5.0,
        next_global_step=1,
        collect_training_diagnostics=collect_diagnostics,
    )
    parameters = {
        name: parameter.detach().clone()
        for name, parameter in accelerator.unwrap_model(model).named_parameters()
    }
    AcceleratorState._reset_state(reset_partial_state=True)
    return ownership, metrics, parameters


def test_diagnostics_preserve_single_optimizer_group_loss_and_update():
    ownership_off, metrics_off, parameters_off = _run(False)
    ownership_on, metrics_on, parameters_on = _run(True)

    assert ownership_on == ownership_off == {
        "vlm": 1,
        "difference_query": 1,
        "action_expert": 1,
    }
    for key in ("ar_loss", "flow_matching_loss", "total_loss"):
        torch.testing.assert_close(metrics_on[key], metrics_off[key])
    for name in parameters_on:
        torch.testing.assert_close(parameters_on[name], parameters_off[name])
    for owner in ("vlm", "difference_query", "action_expert"):
        norm = metrics_on[f"{owner}_grad_norm"]
        assert torch.isfinite(norm)
        assert norm.item() > 0
    assert metrics_on["optimizer_microbatches_per_rank"].item() == 2
    assert metrics_on["optimizer_step_global_samples"].item() == 2


def test_parameter_ownership_rejects_unmapped_trainable_parameters():
    with pytest.raises(ValueError, match="no diagnostic owner"):
        validate_trainable_parameter_ownership(nn.Linear(1, 1))


def test_zero2_gradient_diagnostics_use_current_partition_ownership():
    model = _OwnedObjective()
    parameters = list(model.parameters())
    engine = type("Engine", (), {})()
    engine.module = model
    engine.optimizer = type("ZeroOptimizer", (), {})()
    engine.optimizer.params_in_partition = [parameters]
    engine.optimizer.averaged_gradients = {
        0: [torch.full_like(parameter, index + 1.0) for index, parameter in enumerate(parameters)]
    }

    accelerator = type("DeepSpeedAccelerator", (), {})()
    accelerator.distributed_type = DistributedType.DEEPSPEED
    accelerator.device = torch.device("cpu")
    accelerator.unwrap_model = lambda candidate: candidate.module
    accelerator.reduce = lambda tensor, reduction: tensor

    metrics = module_gradient_norms(engine, accelerator)

    torch.testing.assert_close(metrics["vlm_grad_norm"], torch.tensor(1.0, dtype=torch.float64))
    torch.testing.assert_close(
        metrics["difference_query_grad_norm"], torch.tensor(2.0, dtype=torch.float64)
    )
    torch.testing.assert_close(
        metrics["action_expert_grad_norm"], torch.tensor(3.0, dtype=torch.float64)
    )
