"""Slot objectives in the existing Qwen, optimizer and checkpoint paths."""
import copy
import json
from pathlib import Path
from types import MethodType
from unittest.mock import patch

import pytest
import torch
from accelerate import Accelerator
from transformers.feature_extraction_utils import BatchFeature

from model.reasoning_vla_model import ZR0Model
from model.structured_slot_head import build_slot_head
from utils.slot_config import SlotConfig, resolve_query_layout
from utils.slot_labels import normalize_slot_labels
from utils.slot_loss import slot_loss
from utils.optical_flow_config import OpticalFlowConfig, stage_loss_metadata
from utils.optical_flow_checkpoint import module_checksum
from train_vla import run_optimizer_step_window
from test_difference_query_tiny_vla import TinyVlaDifferenceQueryTest as Tiny
from test_structured_slots import config, stats


def fixture_stats():
    result = stats()
    result.update(version=1, schema_version="future_difference_training_v5", split="train",
                  rules={"max_weight_ratio": 10., "std_floor_m": .001, "time_alignment": "source_semantic_anchor_only",
                         "camera": "first_view", "risk_range": [0, 1]})
    for q in result["classes"]:
        result["classes"][q]["counts"] = [10] * len(result["classes"][q]["weights"])
    return result


def tiny_slot_model(stage="stage2_aux"):
    model = ZR0Model.__new__(ZR0Model)
    torch.nn.Module.__init__(model)
    model.backbone = Tiny.make_backbone(15)
    model.backbone.prepare_inputs = MethodType(lambda _s, inputs: BatchFeature({k: inputs[k] for k in ("input_ids", "attention_mask", "labels") if k in inputs}), model.backbone)
    model.training_stage, model.loss_type = stage, "aux" if stage == "stage2_aux" else "vlm_and_action"
    model.optical_flow_config, model.optical_flow_aux = OpticalFlowConfig(num_flow_queries=8), None
    model.slot_config, model.slot_supervision_stats = config(), fixture_stats()
    model.slot_aux = build_slot_head(32, 7, model.slot_config)
    model.action_expert = Tiny.make_action_head() if stage == "stage3_joint" else None
    model.action_expert_config = Tiny.make_action_head().config
    model.use_difference_query, model.num_difference_queries = True, 15
    model.detach_vlm_outputs_for_action_expert = False
    model.query_role_layout = resolve_query_layout(15, 8, slot_enabled=True, query_enabled=True)
    return model


def sample(valid=True):
    labels = normalize_slot_labels({"query_1": {"progress_t": .1, "progress_tK": .8},
                                   "query_7": {"gripper_transition_id": "maintain_open", "valid_mask": 1}}, is_anchor=valid)
    return {"input_ids": torch.tensor([[5, 6, 7, 8]]), "attention_mask": torch.ones(1, 4, dtype=torch.long),
            "labels": torch.tensor([[-100, -100, 7, 8]]),
            "observation.state": torch.zeros(1, 1, 4), "state_mask": torch.ones(1, 1, 4, dtype=torch.bool),
            "action": torch.zeros(1, 3, 4), "action_mask": torch.ones(1, 3, 4, dtype=torch.bool),
            **{k: v.unsqueeze(0) for k, v in labels.items()}}


def run_window(model, data, optimizer, scheduler, accelerator=None):
    return run_optimizer_step_window(model=model, batches=data, accelerator=accelerator or Accelerator(cpu=True),
        optimizer=optimizer, lr_scheduler=scheduler, training_progress=0, loss_type=model.loss_type,
        vlm_loss_weight=1 if model.loss_type != "aux" else 0, action_expert_loss_weight=1 if model.loss_type != "aux" else 0,
        next_global_step=1, training_stage=model.training_stage, slot_config=model.slot_config,
        optical_flow_config=model.optical_flow_config)


def test_frozen_vlm_query_gradients_and_empty_step():
    torch.set_num_threads(2)
    model = tiny_slot_model()
    frozen = module_checksum(model.backbone.model)
    assert model.action_expert is None
    result = model(sample(), 0)
    result["loss"].backward()
    assert model.backbone.difference_query.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.backbone.model.parameters())
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.slot_aux.parameters())
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.001, weight_decay=.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1)
    first = run_window(model, [sample()], optimizer, scheduler)
    assert first["slot_loss_computed"] and first["optimizer_update_applied"]
    assert model.stage_training_state["completed_optimizer_windows"] == 1
    before = module_checksum(model)
    with patch.object(model, "forward", side_effect=AssertionError("empty window ran forward")):
        empty = run_window(model, [sample(False)], optimizer, scheduler)
    assert empty["optimizer_skip_reason"] == "no_supervision"
    assert module_checksum(model) == before and scheduler.last_epoch == 1
    assert module_checksum(model.backbone.model) == frozen


def test_stage3_missing_slot_and_future_labels_do_not_enter_query_or_ar():
    torch.set_num_threads(2)
    model = tiny_slot_model("stage3_joint").eval()
    data = sample()
    predictions = []
    hook = model.slot_aux.register_forward_pre_hook(lambda _m, args: predictions.append(args[0].detach().clone()))
    first = model(data, 0)
    changed = copy.deepcopy(data)
    changed["slot_Q1"].fill_(.99)
    second = model(changed, 0)
    torch.testing.assert_close(predictions[0], predictions[1], rtol=0, atol=0)
    torch.testing.assert_close(first["ar_loss"], second["ar_loss"], rtol=0, atol=0)
    missing = model(sample(False), 0)
    assert not missing["slot_loss_computed"] and missing["fm_loss_computed"]
    altered_target = copy.deepcopy(data)
    altered_target["input_ids"][0, 2:] = torch.tensor([10, 11])
    altered_target["labels"][0, 2:] = torch.tensor([10, 11])
    model(altered_target, 0)
    torch.testing.assert_close(predictions[0], predictions[-1], rtol=0, atol=0)
    hook.remove()


def test_gas_equivalence():
    torch.set_num_threads(2)
    model = tiny_slot_model()
    other = copy.deepcopy(model)
    data = [sample(), sample(False), sample()]
    data[2]["slot_Q1_mask"][:, 1] = False
    merged = {k: torch.cat([s[k] for s in data]) for k in data[0]}
    for instance, batches in ((model, data), (other, [merged])):
        opt = torch.optim.SGD([p for p in instance.parameters() if p.requires_grad], lr=.01)
        run_window(instance, batches, opt, torch.optim.lr_scheduler.StepLR(opt, 1))
    for a, b in zip(model.parameters(), other.parameters()):
        torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-6)


def test_checkpoint_init_resume_and_inference(tmp_path):
    from test_query_ar_joint_checkpoint import _TinyBackbone, _TinyActionExpert, QueryArWarmStartCheckpointTest
    from utils.slot_checkpoint import read_slot_artifacts
    from safetensors.torch import save_file
    with patch("model.reasoning_vla_model.QwenVLBackbone", _TinyBackbone), patch("model.reasoning_vla_model.FlowmatchingActionHead", _TinyActionExpert), patch("model.flow_matching_action_head.FlowmatchingActionHead", _TinyActionExpert):
        kwargs = dict(action_expert_name_or_path=None, action_expert_config=QueryArWarmStartCheckpointTest.action_config(), tune_vlm=True)
        one = ZR0Model(str(tmp_path / "base"), **kwargs, training_stage="stage1_ar", use_difference_query=True, num_difference_queries=15)
        ar = tmp_path / "ar"
        one.save_pretrained(ar)
        two = ZR0Model(str(ar), **kwargs, training_stage="stage2_aux", optical_flow_config=OpticalFlowConfig(num_flow_queries=8),
            slot_config=config(), slot_supervision_stats=fixture_stats(), init_from_checkpoint=str(ar))
        assert two.action_expert is None
        assert module_checksum(one.backbone) == module_checksum(two.backbone)
        aux = tmp_path / "aux"
        two.save_pretrained(aux)
        stale = _TinyActionExpert(None, True)
        with torch.no_grad():
            for p in stale.parameters():
                p.fill_(99)
        save_file(stale.state_dict(), str(aux / "action_expert.safetensors"))
        three = ZR0Model(str(aux), **kwargs, training_stage="stage3_joint", init_from_checkpoint=str(aux))
        assert module_checksum(two.slot_aux) == module_checksum(three.slot_aux)
        assert module_checksum(stale) != module_checksum(three.action_expert)
        assert three.stage_training_state["completed_optimizer_windows"] == 0
        joint = tmp_path / "joint"
        three.save_pretrained(joint)
        restored = ZR0Model.from_pretrained(joint, tune_vlm=True, tune_action_expert=True)
        assert module_checksum(restored) == module_checksum(three)
        with patch("model.structured_slot_head.StructuredSlotHead.__init__", side_effect=AssertionError("inference constructed Slot")):
            inference = ZR0Model.from_pretrained(joint, for_action_inference=True)
            assert inference.slot_aux is None
        with pytest.raises(ValueError, match="conflicts"):
            ZR0Model(str(aux), **kwargs, training_stage="stage3_joint", optical_flow_config=OpticalFlowConfig(num_flow_queries=7))
        (joint / "slot_head.safetensors").unlink()
        with pytest.raises(ValueError, match="incomplete"):
            read_slot_artifacts(joint)


def distributed_slot_worker(rank, rendezvous, output):
    import torch.distributed as dist
    from accelerate.utils import DistributedType
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method="file://" + rendezvous, rank=rank, world_size=2)
    class CPUAccelerator:
        device = torch.device("cpu")
        num_processes = 2
        gradient_accumulation_steps = 2
        distributed_type = DistributedType.MULTI_CPU
        def reduce(self, value, reduction):
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
    model = torch.nn.parallel.DistributedDataParallel(tiny_slot_model())
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1)
    data = [sample(rank == 0), sample(rank == 0)]
    data[1]["slot_Q7_mask"].fill_(False)
    args = dict(model=model, accelerator=CPUAccelerator(), optimizer=optimizer, lr_scheduler=scheduler,
                training_progress=0, loss_type="aux", vlm_loss_weight=0, action_expert_loss_weight=0,
                next_global_step=1, slot_config=model.module.slot_config, training_stage="stage2_aux")
    result = run_optimizer_step_window(batches=data, **args)
    assert result["slot_Q1_valid_count"] == 2 and result["slot_Q7_valid_count"] == 1
    before = module_checksum(model.module)
    empty = run_optimizer_step_window(batches=[sample(False), sample(False)], **args)
    assert empty["optimizer_skip_reason"] == "no_supervision" and module_checksum(model.module) == before
    if rank == 0:
        torch.save(model.module.state_dict(), output)
    dist.destroy_process_group()


def test_gloo_empty_rank_and_global_means(tmp_path):
    import torch.multiprocessing as mp
    output = str(tmp_path / "result.pt")
    mp.spawn(distributed_slot_worker, args=(str(tmp_path / "rendezvous"), output), nprocs=2, join=True)
    torch.manual_seed(42)
    model = tiny_slot_model()
    data = [sample(), sample(), sample(False), sample(False)]
    data[1]["slot_Q7_mask"].fill_(False)
    merged = {k: torch.cat([s[k] for s in data]) for k in data[0]}
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=.01)
    run_window(model, [merged], optimizer, torch.optim.lr_scheduler.StepLR(optimizer, 1))
    expected = torch.load(output, weights_only=True)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, expected[key], rtol=2e-5, atol=2e-6)


def test_slot_amp_overflow_does_not_mark_computed():
    from accelerate.optimizer import AcceleratedOptimizer
    accelerator = Accelerator(cpu=True)
    previous = accelerator.scaler
    accelerator.scaler = torch.amp.GradScaler("cpu", init_scale=128)
    model = tiny_slot_model()
    native = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.001, weight_decay=.1)
    optimizer = AcceleratedOptimizer(native, scaler=accelerator.scaler)
    scheduler = torch.optim.lr_scheduler.StepLR(native, 1)
    before = module_checksum(model)
    hook = model.backbone.difference_query.weight.register_hook(lambda g: torch.full_like(g, float("inf")))
    try:
        result = run_window(model, [sample()], optimizer, scheduler, accelerator)
        assert result["optimizer_skip_reason"] == "amp_overflow"
        assert module_checksum(model) == before and scheduler.last_epoch == 0
        assert getattr(model, "stage_training_state", None) is None
        hook.remove()
        result = run_window(model, [sample()], optimizer, scheduler, accelerator)
        assert result["optimizer_update_applied"] and model.stage_training_state["slot_loss_computed"]
    finally:
        hook.remove()
        accelerator.scaler = previous


@pytest.mark.parametrize("mutation", ["config", "weights", "stats", "vocabulary", "weight_shape", "layout", "cli"])
def test_strict_slot_artifacts(tmp_path, mutation):
    from model.difference_query import save_difference_query_artifacts
    from utils.slot_checkpoint import save_slot_artifacts, read_slot_artifacts, resolve_slot_checkpoint
    from utils.optical_flow_checkpoint import json_hash
    from safetensors.torch import save_file, load_file
    from dataclasses import replace
    model = tiny_slot_model()
    save_slot_artifacts(model, tmp_path)
    save_difference_query_artifacts(tmp_path, enabled=True, hidden_size=32, difference_query=model.backbone.difference_query.weight)
    assert read_slot_artifacts(tmp_path)["config"] == model.slot_config.to_dict()
    if mutation in {"config", "weights", "stats"}:
        (tmp_path / {"config": "slot_aux_config.json", "weights": "slot_head.safetensors", "stats": "slot_supervision_stats.json"}[mutation]).unlink()
    elif mutation == "vocabulary":
        path = tmp_path / "slot_supervision_stats.json"
        payload = json.loads(path.read_text())
        payload["classes"]["Q2"]["vocabulary"][0] = "unknown"
        path.write_text(json.dumps(payload))
    elif mutation == "weight_shape":
        path = tmp_path / "slot_head.safetensors"
        weights = load_file(path)
        weights["q7.weight"] = weights["q7.weight"][:1]
        save_file(weights, path)
    elif mutation == "layout":
        path = tmp_path / "slot_aux_config.json"
        payload = json.loads(path.read_text())
        payload["query_layout"]["groups"]["G12"][1] += 1
        payload.pop("integrity_sha256")
        payload["integrity_sha256"] = json_hash(payload)
        path.write_text(json.dumps(payload))
    else:
        with pytest.raises(ValueError, match="conflicts"):
            resolve_slot_checkpoint(tmp_path, replace(config(), slot_loss_weight=2))
        return
    with pytest.raises(ValueError):
        read_slot_artifacts(tmp_path)


@pytest.mark.parametrize("stage", ["stage2_aux", "stage3_joint"])
@pytest.mark.parametrize("with_flow", [False, True])
def test_slot_launcher_dry_run(stage, with_flow):
    import os
    import shlex
    import subprocess
    from utils.cli_options import parse_train_options
    root = Path(__file__).resolve().parents[1]
    environment = dict(os.environ, INIT_CHECKPOINT="/fixture", OUTPUT_DIR="/fixture-output", SLOT_SUPERVISION_DIR="/fixture-supervision",
        SLOT_LOSS_WEIGHT="1", ACCELERATE_CONFIG=str(root / "accelerate_configs/structured_slots_zero2_bf16.yaml"),
        WANDB_PROJECT="fixture", WANDB_RUN_NAME="fixture", WITH_FLOW=str(int(with_flow)), OPTICAL_FLOW_LOSS_WEIGHT="1")
    printed = subprocess.check_output(["bash", str(root / "scripts/run_structured_slot_stage.sh"), stage, "--print-command"], env=environment, text=True)
    arguments = shlex.split(printed)
    options = parse_train_options(arguments[arguments.index("train_vla.py") + 1:])
    assert options.training_stage == stage and options.wandb_failure_policy == "required"
    assert options.num_difference_queries == 32 and options.num_flow_queries == 8
    assert options.tune_vlm == (stage == "stage3_joint")
    assert options.tune_action_expert == (stage == "stage3_joint")
    assert (options.optical_flow_aux_type != "none") == with_flow
    assert options.stage2_aux_sampling == ("any_aux_valid" if with_flow else "slot_valid")


@pytest.mark.parametrize("stage", ["stage2_aux", "stage3_joint"])
@pytest.mark.parametrize("slot_valid,flow_valid", [(True, True), (False, True), (True, False)])
def test_combined_slot_flow_objectives(stage, slot_valid, flow_valid):
    from model.optical_flow_aux_head import build_optical_flow_head
    from test_optical_flow_aux import config as flow_config, batch as flow_batch
    model = tiny_slot_model(stage)
    model.optical_flow_config = flow_config(num_flow_queries=8)
    model.optical_flow_aux = build_optical_flow_head(32, model.optical_flow_config)
    data = sample(slot_valid)
    data.update({k: v for k, v in flow_batch((flow_valid,)).items() if k.startswith("flow_")})
    outputs = model(data, 0)
    expected = outputs["slot_loss_weighted"] + outputs["optical_flow_loss"]
    if stage == "stage3_joint":
        expected = expected + outputs["ar_loss"] + outputs["flow_matching_loss"]
    torch.testing.assert_close(outputs["loss"], expected)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.0001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1)
    metrics = run_window(model, [data], optimizer, scheduler)
    assert metrics["optimizer_update_applied"]
    assert metrics["slot_loss_computed"] == slot_valid
    assert metrics["flow_loss_computed"] == flow_valid
    assert model.stage_training_state["slot_loss_computed"] == slot_valid


def test_legacy_unimplemented_description_is_read_as_disabled():
    from utils.optical_flow_checkpoint import initial_stage_training_state, read_stage_training_state
    from test_optical_flow_aux import config as flow_config
    state = initial_stage_training_state("stage2_aux", flow_config())
    state["stage_description"] = "stage2 OF-only / Slot not implemented"
    restored = read_stage_training_state(state, flow_config())
    assert restored["stage_description"] == "stage2 OF-only / Slot disabled"
    assert restored["slot_loss_computed"] is False
