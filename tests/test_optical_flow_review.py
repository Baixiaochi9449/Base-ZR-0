"""Regression coverage for OF device, provenance and checkpoint review findings."""

from contextlib import nullcontext
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from accelerate import Accelerator

from test_optical_flow_aux import config, tiny_batch, tiny_stage_model
from model.difference_query import resolve_difference_query_config, save_difference_query_artifacts
from model.optical_flow_aux_head import build_optical_flow_head
from utils.optical_flow_checkpoint import read_flow_artifacts, reject_flow_zero3, save_flow_artifacts
from utils.optimizer_step_loss import OptimizerStepMetricAccumulator
from utils.wandb_training_logger import WandbTrainingLogger
from train_vla import json_scalar_metrics, run_optimizer_step_window


@pytest.mark.parametrize("backbone_dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("autocast", [False, True])
def test_real_tiny_qwen_mixed_dtype_backward(backbone_dtype, autocast):
    model = tiny_stage_model("stage2_aux")
    model.backbone.model.to(backbone_dtype)
    assert model.optical_flow_aux.input_projection.weight.dtype == torch.float32
    with torch.autocast("cpu", dtype=torch.bfloat16) if autocast else nullcontext():
        result = model(tiny_batch(), 0.)
    result["loss"].backward()
    for parameter in [model.backbone.difference_query.weight, *model.optical_flow_aux.parameters()]:
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0


@pytest.mark.parametrize("query_dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("head_dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_head_explicit_parameter_dtype(query_dtype, head_dtype):
    from utils.optical_flow_loss import optical_flow_loss
    torch.manual_seed(42)
    head = build_optical_flow_head(32, config()).to(head_dtype)
    queries = torch.randn(1, 4, 32, dtype=query_dtype, requires_grad=True)
    prediction = head(queries)
    assert prediction.dtype == head_dtype
    optical_flow_loss(prediction, tiny_batch(), config())["optical_flow_loss"].backward()
    assert torch.isfinite(queries.grad).all() and queries.grad[:, -2:].abs().sum() > 0
    assert queries.grad[:, :2].count_nonzero() == 0


def flow_checkpoint(path):
    model = SimpleNamespace(optical_flow_config=config(), optical_flow_aux=build_optical_flow_head(32, config()),
                            training_stage="stage2_aux", stage_description="OF-only")
    save_flow_artifacts(model, path)
    save_difference_query_artifacts(path, enabled=True, hidden_size=32, difference_query=torch.randn(4, 32))
    return model


@pytest.mark.parametrize("missing", ["difference_query_config.json", "difference_query.safetensors", "both"])
def test_flow_requires_query_artifacts(tmp_path, missing):
    flow_checkpoint(tmp_path)
    for name in ("difference_query_config.json", "difference_query.safetensors"):
        if missing in (name, "both"):
            (tmp_path / name).unlink()
    with pytest.raises(ValueError, match="Difference Query"):
        read_flow_artifacts(tmp_path)
    with pytest.raises(ValueError, match="Difference Query"):
        resolve_difference_query_config(tmp_path, None, use_difference_query=True, num_difference_queries=4)


@pytest.mark.parametrize("field,value", [("hidden_size", 31), ("num_difference_queries", 3)])
def test_query_shape_mismatch_rejected(tmp_path, field, value):
    flow_checkpoint(tmp_path)
    path = tmp_path / "difference_query_config.json"
    data = json.loads(path.read_text())
    data[field] = value
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="shape"):
        read_flow_artifacts(tmp_path)


def test_valid_query_shape_but_wrong_flow_hidden_rejected(tmp_path):
    flow_checkpoint(tmp_path)
    save_difference_query_artifacts(tmp_path, enabled=True, hidden_size=16, difference_query=torch.randn(4, 16))
    with pytest.raises(ValueError, match="hidden size"):
        read_flow_artifacts(tmp_path)


def test_initial_creation_and_checkpoint_load_are_distinct(tmp_path):
    initial = resolve_difference_query_config(tmp_path, None, use_difference_query=True, num_difference_queries=4)
    assert initial.random_initialization and initial.checkpoint_tensor is None
    flow_checkpoint(tmp_path)
    loaded = resolve_difference_query_config(tmp_path, None)
    assert not loaded.random_initialization and loaded.checkpoint_tensor is not None
    with pytest.raises(ValueError, match="num_difference_queries conflicts"):
        resolve_difference_query_config(tmp_path, None, num_difference_queries=3)


def test_zero3_rejected_at_configuration_and_export(tmp_path):
    reject_flow_zero3(config={"zero_optimization": {"stage": 2}}, flow_enabled=True)
    reject_flow_zero3(config={}, flow_enabled=True)
    reject_flow_zero3(config={"zero_optimization": {"stage": 3}}, flow_enabled=False)
    with pytest.raises(RuntimeError, match="ZeRO-3.*ZeRO-2"):
        reject_flow_zero3(config={"zero_optimization": {"stage": 3}}, flow_enabled=True)
    with pytest.raises(RuntimeError, match="use ZeRO-2 or non-ZeRO"):
        reject_flow_zero3(config={"zero_optimization": {"stage": 1}}, flow_enabled=True)
    model = flow_checkpoint(tmp_path)
    before = (tmp_path / "optical_flow_aux.safetensors").read_bytes()
    next(model.optical_flow_aux.parameters()).ds_id = 1
    with pytest.raises(RuntimeError, match="partitioned"):
        save_flow_artifacts(model, tmp_path)
    assert (tmp_path / "optical_flow_aux.safetensors").read_bytes() == before
    from train_vla import save_model
    from utils.training_checkpoint import checkpoint_model_optimizer_scheduler
    accelerator = SimpleNamespace(unwrap_model=lambda _: model)
    with pytest.raises(RuntimeError, match="ZeRO-3"):
        save_model(accelerator, model, tmp_path, "step-1")
    with pytest.raises(RuntimeError, match="ZeRO-3"):
        checkpoint_model_optimizer_scheduler(model, tmp_path, 1, None, accelerator)


def fake_logger(accelerator):
    run = SimpleNamespace(logs=[])
    run.log = lambda metrics, step: run.logs.append((metrics, step))
    with patch.dict("sys.modules", {"wandb": SimpleNamespace(init=lambda **_: run)}):
        logger = WandbTrainingLogger(accelerator, project="test", run_name="test", run_id="test",
                                     resume="never", log_dir="/tmp", failure_policy="required")
    return logger, run


@pytest.mark.parametrize("stage,empty", [("stage1_ar", False), ("stage2_aux", False),
                                         ("stage2_aux", True), ("stage3_joint", False), ("stage3_joint", True)])
def test_window_json_and_wandb_stage_metadata(stage, empty):
    model = tiny_stage_model(stage)
    data = tiny_batch()
    if empty:
        data.update(flow_supervision_available=torch.tensor([False]), flow_target={}, flow_valid_mask={})
    accelerator = Accelerator(cpu=True)
    original_reduce, original_gather = accelerator.reduce, accelerator.gather
    def reduce(value, **kwargs):
        assert value.device == accelerator.device
        return original_reduce(value, **kwargs)
    def gather(value):
        assert value.device == accelerator.device
        return original_gather(value)
    accelerator.reduce, accelerator.gather = reduce, gather
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1)
    metrics = run_optimizer_step_window(model=model, batches=[data], accelerator=accelerator, optimizer=optimizer,
        lr_scheduler=scheduler, training_progress=0, loss_type=model.loss_type, vlm_loss_weight=1,
        action_expert_loss_weight=1, optical_flow_config=model.optical_flow_config, training_stage=stage, next_global_step=1)
    for value in metrics.values():
        if isinstance(value, torch.Tensor):
            assert value.device == accelerator.device
    record = json.loads(json.dumps(json_scalar_metrics(metrics)))
    assert record["training_stage"] == stage and record["slot_loss_computed"] is False
    assert record["ar_loss_computed"] == (stage != "stage2_aux")
    assert record["fm_loss_computed"] == (stage == "stage3_joint")
    assert record["flow_loss_computed"] == (stage != "stage1_ar" and not empty)
    assert record["provisional"] == (stage == "stage3_joint")
    assert "slot_loss" not in record
    logger, run = fake_logger(accelerator)
    logger.log(step=0 if empty and stage == "stage2_aux" else 1, mean_metrics=metrics, scalar_metrics={})
    assert run.logs[-1][0] == record


def test_device_simulation_of_only_accumulator_to_wandb():
    # Meta tensors stand in for a non-CPU collective device without allocating a GPU.
    class DeviceAccelerator:
        device = torch.device("meta")
        is_main_process = True
        num_processes = 1
        def reduce(self, value, reduction):
            assert value.device == self.device
            return torch.tensor(0.)
        def gather(self, value):
            assert value.device == self.device
            return torch.zeros(1)
    accelerator = DeviceAccelerator()
    accumulator = OptimizerStepMetricAccumulator(loss_type="aux", vlm_loss_weight=0, action_expert_loss_weight=0,
                                                  device=accelerator.device)
    metrics = accumulator.finalize(accelerator)
    assert all(value.device == accelerator.device for value in metrics.values())
    for key in ("optical_flow_loss", "weighted_optical_flow_loss", "total_loss", "loss"):
        metrics[key] = torch.ones((), device=accelerator.device)
    # The logger also repairs a caller's CPU metric before entering the collective.
    metrics["caller_cpu_metric"] = torch.tensor(1.)
    logger, run = fake_logger(accelerator)
    logger.log(step=1, mean_metrics=metrics, scalar_metrics={})
    assert run.logs


def test_computed_zero_loss_is_not_uncomputed():
    from utils.optical_flow_config import stage_loss_metadata
    accumulator = OptimizerStepMetricAccumulator(loss_type="vlm", vlm_loss_weight=1, action_expert_loss_weight=0,
                                                  device=torch.device("cpu"))
    accumulator.update({"ar_loss_sum": torch.tensor(0.), "ar_loss_count": torch.tensor(1.),
                        **stage_loss_metadata("stage1_ar", ar=True)}, {})
    result = accumulator.finalize(Accelerator(cpu=True))
    assert result["ar_loss"] == 0 and result["ar_loss_computed"] is True
    assert result["slot_loss_computed"] is False and "slot_loss" not in result


@pytest.mark.parametrize("vision_mutation", [None, "448", "missing", "height_missing", "string_width", "bool_width", "resize", "augmentation"])
def test_real_stage06_checkpoint_policy_initialization(tmp_path, monkeypatch, vision_mutation):
    from utils.constants import DATASET2FEATURE
    from utils.dataset_spec import resolve_dataset_spec
    from utils.dataset_manifest import (build_resolved_dataset_manifest, write_resolved_dataset_manifest,
                                        validate_policy_dataset_manifest, dataset_spec_to_manifest)
    from test_query_ar_joint_checkpoint import _TinyBackbone, _TinyActionExpert, QueryArWarmStartCheckpointTest
    from model.reasoning_vla_model import ZR0Model
    import policies.reasoning_vla_policy as policy_module
    entry = dict(DATASET2FEATURE["stage06_libero_flow"])
    root = Path(entry["dataset_path"])
    if not root.is_dir():
        pytest.skip("real Stage06 LIBERO metadata unavailable")
    train_entry = dict(entry, optical_flow_data_root=str(root / "stage06_flow/libero_delta10"),
                       optical_flow_manifest="manifest.2849ed69240ad542.jsonl", flow_delta_frames=10)
    action_config = QueryArWarmStartCheckpointTest.action_config()
    action_config.action_dim = action_config.state_dim = 64
    spec = resolve_dataset_spec("stage06_libero_flow", train_entry, action_horizon=3, require_action=True)
    assert spec.sidecar_sha256 == "0a39a44943bf22edc8501419d5cb8f39e31117bb23c86c1c4c48030f045d5c26"
    # Real policy and model save/load entry points; small modules keep this CPU-only.
    with patch("model.reasoning_vla_model.QwenVLBackbone", _TinyBackbone), \
         patch("model.reasoning_vla_model.FlowmatchingActionHead", _TinyActionExpert), \
         patch("model.flow_matching_action_head.FlowmatchingActionHead", _TinyActionExpert):
        model = ZR0Model(str(tmp_path / "base"), action_expert_name_or_path=None,
                        action_expert_config=action_config, training_stage="stage3_joint", optical_flow_config=config(),
                        use_difference_query=True, num_difference_queries=4, tune_vlm=True, tune_action_expert=True)
        model.resolved_dataset_manifest = build_resolved_dataset_manifest([spec], "vlm_and_action")
        checkpoint = tmp_path / "checkpoint"
        model.save_pretrained(checkpoint)
        if vision_mutation:
            saved = model.resolved_dataset_manifest
            contract = saved["entries"][0]["vision_input_contract"]
            if vision_mutation == "448":
                contract.update(image_width=448, image_height=448)
            elif vision_mutation == "missing":
                del saved["entries"][0]["vision_input_contract"]
            elif vision_mutation == "height_missing":
                del contract["image_height"]
            elif vision_mutation == "string_width":
                contract["image_width"] = "224"
            elif vision_mutation == "bool_width":
                contract["image_width"] = True
            elif vision_mutation == "resize":
                contract["do_resize"] = True
            else:
                contract["random_geometric_augmentation"] = "false"
            # Recompute the hash: semantic rejection must also catch valid JSON metadata.
            write_resolved_dataset_manifest(checkpoint, saved)
            with pytest.raises(ValueError, match="vision_input_contract"):
                validate_policy_dataset_manifest(spec, checkpoint)
        monkeypatch.setattr(policy_module.AutoProcessor, "from_pretrained", lambda *_: object())
        monkeypatch.setattr(torch, "compile", lambda model, **_: model)
        with patch("h5py.File", side_effect=AssertionError("policy must not open HDF5")), \
             patch("utils.optical_flow_reader.OpticalFlowReader", side_effect=AssertionError("policy must not initialize reader")):
            if vision_mutation:
                with pytest.raises(ValueError, match="vision_input_contract"):
                    policy_module.ZR0Policy("stage06_libero_flow", str(checkpoint), "direct_action", 1, device="cpu")
                return
            policy = policy_module.ZR0Policy("stage06_libero_flow", str(checkpoint), "direct_action", 1, device="cpu")
        assert policy.dataset_spec.sidecar_sha256 == spec.sidecar_sha256
        assert policy.dataset_spec.canonical_schema == spec.canonical_schema
        assert policy.dataset_spec.vision_input_contract == spec.vision_input_contract
        assert dataset_spec_to_manifest(policy.dataset_spec, "action") == dataset_spec_to_manifest(spec, "action")
        assert policy.model.optical_flow_aux is None
        assert policy.model.action_expert is not None
        assert policy.state_dim == 8 and policy.action_dim == 7
