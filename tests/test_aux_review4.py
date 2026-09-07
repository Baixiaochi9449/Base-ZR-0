"""Regression probes for the fourth independent auxiliary-training review."""

from dataclasses import replace
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def _slot_model(stage="stage2_aux"):
    from test_slot_integration import tiny_slot_model
    return tiny_slot_model(stage)


def _slot_sample(valid=True):
    from test_slot_integration import sample
    return sample(valid)


def _run(model, batches, optimizer, scheduler, *, accelerator=None, ar_weight=None, fm_weight=None):
    from train_vla import run_optimizer_step_window
    from accelerate import Accelerator
    return run_optimizer_step_window(
        model=model,
        batches=batches,
        accelerator=accelerator or Accelerator(cpu=True, gradient_accumulation_steps=len(batches)),
        optimizer=optimizer,
        lr_scheduler=scheduler,
        training_progress=0.,
        loss_type=model.loss_type,
        vlm_loss_weight=(0. if model.loss_type == "aux" else 1.) if ar_weight is None else ar_weight,
        action_expert_loss_weight=(0. if model.loss_type == "aux" else 1.) if fm_weight is None else fm_weight,
        next_global_step=1,
        training_stage=model.training_stage,
        slot_config=model.slot_config,
        optical_flow_config=model.optical_flow_config,
    )


def test_explicit_route_is_used_when_both_auxiliary_heads_are_disabled(tmp_path, monkeypatch):
    from utils import load_training_dataset as loader

    route = tmp_path / "aux.json"
    route.write_text(json.dumps({"version": 1, "datasets": {
        "source": {"dataset_path": "/explicit/source", "ar_sidecar_path": "/explicit/joint"}
    }}))
    seen = []

    class FakeDataset:
        def __init__(self, *, entry, **kwargs):
            seen.append((entry["dataset_path"], entry["ar_sidecar_path"], entry["flow_enabled"], entry["slot_enabled"]))
            self.spec = SimpleNamespace(dataset_entry=entry["dataset_entry"])
        def __len__(self):
            return 1

    monkeypatch.setattr(loader, "AutoProcessor", SimpleNamespace(from_pretrained=lambda *a, **k: object()))
    monkeypatch.setitem(loader.DATASET2FEATURE, "source", {
        "dataset_path": "/legacy/source", "dataset_type": "vla", "dataset_adapter": "lerobot_v3_future_difference",
        "sample_ratio": 1., "use_quantile": True,
    })
    monkeypatch.setattr(loader, "DATASET_ADAPTERS", {**loader.DATASET_ADAPTERS, "lerobot_v3_future_difference": FakeDataset})
    monkeypatch.setattr(loader, "build_resolved_dataset_manifest", lambda specs, loss_type: {"entries": []})
    dataset = loader.build_concat_streaming_dataset(
        ["source"], "model", None, 1, 32, None, loss_type="vlm_and_action",
        slot_config=SimpleNamespace(enabled=False), optical_flow_config=SimpleNamespace(enabled=False),
        aux_dataset_config=str(route),
    )
    assert len(dataset) == 1
    assert seen == [("/explicit/source", "/explicit/joint", False, False)]


def _optimizer(model):
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.001, weight_decay=.1)
    return optimizer, torch.optim.lr_scheduler.StepLR(optimizer, 1)


def test_zero_weight_slot_task_does_not_update_or_advance_scheduler():
    model = _slot_model()
    weights = dict(model.slot_config.slot_task_weights)
    weights["Q1"] = 0.
    model.slot_config = replace(model.slot_config, slot_task_weights=weights).validate()
    batch = _slot_sample()
    batch["slot_Q7_mask"].zero_()
    optimizer, scheduler = _optimizer(model)
    before = [p.detach().clone() for p in model.parameters()]
    state = copy.deepcopy(optimizer.state_dict())
    result = _run(model, [batch], optimizer, scheduler)
    assert result["optimizer_skip_reason"] == "no_supervision"
    assert result["optimizer_update_applied"] is False
    assert result["slot_available"] is True
    assert result["slot_sample_coverage"] == 1
    assert result["slot_active_loss_computed"] is False
    assert result["slot_Q1_valid_count"] == 1 and result["slot_Q1_active_count"] == 0
    assert all(torch.equal(a, b) for a, b in zip(before, model.parameters()))
    assert optimizer.state_dict() == state and scheduler.last_epoch == 0


@pytest.mark.parametrize("component", ["presence", "bbox", "risk"])
def test_zero_weight_q9_component_is_not_active(component):
    model = _slot_model()
    field = {"presence": "slot_presence_weight", "bbox": "slot_obstacle_bbox_weight", "risk": "slot_risk_weight"}[component]
    model.slot_config = replace(model.slot_config, **{field: 0.}).validate()
    batch = _slot_sample(False)
    batch[f"slot_Q9_{component}_mask"].fill_(True)
    optimizer, scheduler = _optimizer(model)
    before = copy.deepcopy(model.state_dict())
    result = _run(model, [batch], optimizer, scheduler)
    assert result["optimizer_update_applied"] is False
    assert result["slot_Q9_valid_count"] == 1 and result["slot_Q9_active_count"] == 0
    assert scheduler.last_epoch == 0 and not optimizer.state
    for key, value in model.state_dict().items():
        assert torch.equal(value, before[key])


def test_zero_outer_slot_weight_keeps_labels_but_skips_update():
    model = _slot_model()
    model.slot_config = replace(model.slot_config, slot_loss_weight=0.).validate()
    optimizer, scheduler = _optimizer(model)
    before = [p.detach().clone() for p in model.parameters()]
    result = _run(model, [_slot_sample()], optimizer, scheduler)
    assert result["optimizer_update_applied"] is False
    assert result["slot_Q1_valid_count"] == 1 and result["slot_active_supervision_count"] == 0
    assert all(torch.equal(a, b) for a, b in zip(before, model.parameters()))
    assert scheduler.last_epoch == 0


def test_positive_weight_zero_value_loss_still_updates():
    model = _slot_model()
    original = model.forward
    def exact_zero(*args, **kwargs):
        outputs = original(*args, **kwargs)
        outputs["slot_Q1_loss_sum"] = outputs["slot_Q1_loss_sum"] * 0.
        return outputs
    model.forward = exact_zero
    batch = _slot_sample()
    batch["slot_Q7_mask"].zero_()
    optimizer, scheduler = _optimizer(model)
    before = [p.detach().clone() for p in model.parameters()]
    result = _run(model, [batch], optimizer, scheduler)
    assert result["optimizer_update_applied"] is True
    assert any(not torch.equal(a, b) for a, b in zip(before, model.parameters()))
    assert scheduler.last_epoch == 1
    assert result["loss"] == 0 and result["slot_Q1_active_count"] == 1
    assert optimizer.state and all(state["step"] == 1 for state in optimizer.state.values())


def test_gas_mixed_empty_microbatch_keeps_valid_gradient_and_stage3_other_objectives():
    model = _slot_model()
    optimizer, scheduler = _optimizer(model)
    before = [p.detach().clone() for p in model.parameters()]
    result = _run(model, [_slot_sample(False), _slot_sample(True)], optimizer, scheduler)
    assert result["optimizer_update_applied"] is True
    assert any(not torch.equal(a, b) for a, b in zip(before, model.parameters()))

    joint = _slot_model("stage3_joint")
    optimizer, scheduler = _optimizer(joint)
    result = _run(joint, [_slot_sample(False)], optimizer, scheduler)
    assert result["optimizer_update_applied"] is True
    assert result["slot_active_loss_computed"] is False
    assert result["ar_loss_count"] > 0 and result["flow_matching_loss_count"] > 0


def assert_state_equal(left, right):
    from torch.utils._pytree import tree_flatten
    a, a_spec = tree_flatten(left)
    b, b_spec = tree_flatten(right)
    assert a_spec == b_spec
    for x, y in zip(a, b):
        assert torch.equal(x, y) if isinstance(x, torch.Tensor) else x == y


@pytest.mark.parametrize("kind", ["zero_task", "zero_outer", "missing", "zero_q9_component"])
def test_inactive_window_preserves_existing_adam_moments_and_global_step(kind):
    from accelerate import Accelerator
    from train_vla import advance_global_step, json_scalar_metrics
    from utils.optical_flow_checkpoint import checkpoint_stage_metadata
    model = _slot_model()
    optimizer, scheduler = _optimizer(model)
    accelerator = Accelerator(cpu=True)
    first = _run(model, [_slot_sample()], optimizer, scheduler, accelerator=accelerator)
    assert advance_global_step(0, first, accelerator) == 1
    data = _slot_sample(kind != "missing")
    if kind == "zero_task":
        model.slot_config = replace(model.slot_config, slot_task_weights={**model.slot_config.slot_task_weights, "Q1": 0.})
        data["slot_Q7_mask"].zero_()
    elif kind == "zero_outer":
        model.slot_config = replace(model.slot_config, slot_loss_weight=0.)
    elif kind == "zero_q9_component":
        data = _slot_sample(False)
        data["slot_Q9_risk_mask"].fill_(True)
        model.slot_config = replace(model.slot_config, slot_risk_weight=0.)
    before = copy.deepcopy((model.state_dict(), optimizer.state_dict(), scheduler.state_dict(), checkpoint_stage_metadata(model)))
    result = _run(model, [data], optimizer, scheduler, accelerator=accelerator)
    after = (model.state_dict(), optimizer.state_dict(), scheduler.state_dict(), checkpoint_stage_metadata(model))
    assert_state_equal(before, after)
    assert advance_global_step(1, result, accelerator) == 1
    logged = json_scalar_metrics(result)
    assert logged["optimizer_update_applied"] is False and logged["optimizer_skip_reason"] == "no_supervision"
    assert logged["slot_active_supervision_count"] == 0
    assert logged["slot_sample_coverage"] == (0 if kind == "missing" else 1)


@pytest.mark.parametrize("order", ["first", "last", "middle"])
def test_gas_with_only_zero_weight_labels_matches_single_active_sample(order):
    torch.manual_seed(42)
    model = _slot_model()
    model.slot_config = replace(model.slot_config, slot_task_weights={**model.slot_config.slot_task_weights, "Q1": 0.})
    control = copy.deepcopy(model)
    inactive = _slot_sample()
    inactive["slot_Q7_mask"].zero_()
    data = [copy.deepcopy(inactive), copy.deepcopy(inactive)]
    data.insert({"first": 0, "middle": 1, "last": 2}[order], _slot_sample())
    for instance, values in ((model, data), (control, [_slot_sample()])):
        optimizer, scheduler = _optimizer(instance)
        result = _run(instance, values, optimizer, scheduler)
        assert result["optimizer_update_applied"] and scheduler.last_epoch == 1
        assert result["slot_Q7_active_count"] == 1
    for a, b in zip(model.parameters(), control.parameters()):
        torch.testing.assert_close(a, b, rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("stage,ar_valid,fm_valid,flow", [
    ("stage2_aux", False, False, True), ("stage3_joint", True, False, False),
    ("stage3_joint", False, True, False), ("stage3_joint", False, False, False)])
def test_other_active_objectives_determine_update(stage, ar_valid, fm_valid, flow):
    model = _slot_model(stage)
    model.slot_config = replace(model.slot_config, slot_loss_weight=0.)
    data = _slot_sample()
    if not ar_valid:
        data["labels"].fill_(-100)
    if not fm_valid:
        data["action_mask"].zero_()
        data["action_supervision_available"] = torch.tensor([False])
    if flow:
        from model.optical_flow_aux_head import build_optical_flow_head
        from test_optical_flow_aux import config, batch
        model.optical_flow_config = config(num_flow_queries=8)
        model.optical_flow_aux = build_optical_flow_head(32, model.optical_flow_config)
        data.update({key: value for key, value in batch((True,)).items() if key.startswith("flow_")})
    optimizer, scheduler = _optimizer(model)
    before = copy.deepcopy(model.state_dict())
    result = _run(model, [data], optimizer, scheduler)
    active = bool(ar_valid or fm_valid or flow)
    assert result["optimizer_update_applied"] is active and scheduler.last_epoch == int(active)
    assert result["slot_Q1_valid_count"] == 1 and result["slot_active_supervision_count"] == 0
    assert any(not torch.equal(before[key], value) for key, value in model.state_dict().items()) is active
    assert bool(optimizer.state) is active


@pytest.mark.parametrize("routes", [None, {}, [], {"other": {}}, {"source": []}, {"source": {}}])
def test_bad_explicit_route_never_falls_back(tmp_path, monkeypatch, routes):
    from utils import load_training_dataset as loader
    path = tmp_path / "route.json"
    path.write_text(json.dumps({"version": 1, "datasets": routes}))
    monkeypatch.setitem(loader.DATASET2FEATURE, "source", {"dataset_path": "/legacy"})
    monkeypatch.setattr(loader.AutoProcessor, "from_pretrained", lambda *a, **k: pytest.fail("route failed to reject before processor"))
    with pytest.raises(ValueError, match="route"):
        loader.build_concat_streaming_dataset(["source"], "unused", None, 1, 32, None, aux_dataset_config=path)


def distributed_active_worker(rank, rendezvous, output):
    import torch.distributed as dist
    from accelerate.utils import DistributedType
    from train_vla import advance_global_step
    torch.set_num_threads(2)
    dist.init_process_group("gloo", init_method="file://" + rendezvous, rank=rank, world_size=2)
    class CPUAccelerator:
        is_main_process = rank == 0
        device = torch.device("cpu")
        distributed_type = DistributedType.MULTI_CPU
        num_processes = 2
        gradient_accumulation_steps = 2
        scaler = None
        def reduce(self, value, reduction="sum"):
            value = value.clone()
            dist.all_reduce(value)
            return value / 2 if reduction == "mean" else value
        def gather(self, value):
            values = [torch.zeros_like(value) for _ in range(2)]
            dist.all_gather(values, value)
            return torch.cat(values)
        def no_sync(self, model):
            return model.no_sync()
        def unwrap_model(self, model):
            return model.module
        def backward(self, value):
            (value / 2).backward()
    torch.manual_seed(42)
    bare = _slot_model()
    bare.slot_config = replace(bare.slot_config, slot_task_weights={**bare.slot_config.slot_task_weights, "Q1": 0.})
    model = torch.nn.parallel.DistributedDataParallel(bare)
    optimizer, scheduler = _optimizer(model)
    from train_vla import run_optimizer_step_window
    accelerator = CPUAccelerator()
    inactive = _slot_sample()
    inactive["slot_Q7_mask"].zero_()
    args = dict(model=model, accelerator=accelerator, optimizer=optimizer, lr_scheduler=scheduler,
        training_progress=0., loss_type="aux", vlm_loss_weight=0., action_expert_loss_weight=0., next_global_step=1,
        training_stage="stage2_aux", slot_config=bare.slot_config, optical_flow_config=bare.optical_flow_config)
    result = run_optimizer_step_window(batches=[_slot_sample() if rank == 0 else inactive, inactive], **args)
    assert result["slot_Q1_valid_count"] == 4 and result["slot_Q7_active_count"] == 1
    assert result["optimizer_update_applied"] and advance_global_step(0, result, accelerator) == 1
    before = copy.deepcopy((bare.state_dict(), optimizer.state_dict(), scheduler.state_dict()))
    skipped = run_optimizer_step_window(batches=[inactive, inactive], **args)
    assert skipped["slot_Q1_valid_count"] == 4 and skipped["slot_active_supervision_count"] == 0
    assert advance_global_step(1, skipped, accelerator) == 1
    assert_state_equal(before, (bare.state_dict(), optimizer.state_dict(), scheduler.state_dict()))
    if rank == 0:
        torch.save(bare.state_dict(), output)
    dist.destroy_process_group()


def test_two_rank_inactive_labels_preserve_other_rank_gradient_and_skip_global_empty(tmp_path):
    import torch.multiprocessing as mp
    output = tmp_path / "model.pt"
    mp.spawn(distributed_active_worker, args=(str(tmp_path / "rendezvous"), str(output)), nprocs=2, join=True)
    torch.manual_seed(42)
    model = _slot_model()
    model.slot_config = replace(model.slot_config, slot_task_weights={**model.slot_config.slot_task_weights, "Q1": 0.})
    optimizer, scheduler = _optimizer(model)
    result = _run(model, [_slot_sample()], optimizer, scheduler)
    assert result["optimizer_update_applied"]
    expected = torch.load(output, weights_only=True)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, expected[key], rtol=2e-5, atol=2e-6)


class SamplerDataset(torch.utils.data.Dataset):
    natural_mix_block_size = 8
    def __init__(self, offset):
        self.offset = offset
    def __len__(self):
        return 12
    def sampling_group_ranges(self):
        return [(0, 4), (4, 8), (8, 12)]
    def __getitem__(self, index):
        return {"input_ids": torch.tensor([self.offset + index]), "attention_mask": torch.ones(1, dtype=torch.long)}


@pytest.mark.parametrize("workers", [0, 2])
def test_prepared_sampler_resume_epoch_matches_uninterrupted_epochs(workers):
    from accelerate import Accelerator, DataLoaderConfiguration
    from utils.load_training_dataset import create_dataloader_for_concat, set_dataloader_epoch
    dataset = torch.utils.data.ConcatDataset([SamplerDataset(12 * i) for i in range(4)])
    accelerator = Accelerator(cpu=True, dataloader_config=DataLoaderConfiguration(even_batches=False))
    def create():
        raw = create_dataloader_for_concat(dataset, batch_size_per_device=2, num_workers=workers, prefetch_factor=3, seed=42)
        return accelerator.prepare_data_loader(raw), raw.batch_sampler
    control, control_sampler = create()
    set_dataloader_epoch(control, control_sampler, 0)
    epoch0 = [batch["input_ids"].tolist() for batch in control]
    set_dataloader_epoch(control, control_sampler, 1)
    epoch1 = [batch["input_ids"].tolist() for batch in control]
    assert epoch0 != epoch1
    resumed, resumed_sampler = create()
    set_dataloader_epoch(resumed, resumed_sampler, 1)
    continued = [batch["input_ids"].tolist() for batch in resumed]
    assert continued[3:] == epoch1[3:], "fresh prepared loader replayed the wrong epoch after a GAS=3 boundary"


@pytest.mark.parametrize("stage", ["stage1_ar", "stage3_joint"])
def test_empty_activity_gate_keeps_required_ar_fm_weight_rejection(stage):
    from test_optical_flow_aux import tiny_stage_model, tiny_batch
    from utils.optical_flow_config import OpticalFlowConfig
    from utils.slot_config import SlotConfig
    model = tiny_stage_model(stage)
    model.optical_flow_config, model.optical_flow_aux = OpticalFlowConfig(), None
    model.slot_config = SlotConfig()
    optimizer, scheduler = _optimizer(model)
    with pytest.raises(ValueError, match="greater than zero"):
        _run(model, [tiny_batch()], optimizer, scheduler, ar_weight=0., fm_weight=0.)
    assert not optimizer.state and scheduler.last_epoch == 0
