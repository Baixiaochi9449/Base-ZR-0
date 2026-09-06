"""Stage metadata is cumulative evidence, separate from window diagnostics."""

import json
from unittest.mock import patch

import pytest
import torch
from accelerate import Accelerator

from model.reasoning_vla_model import ZR0Model
from test_optical_flow_aux import config, tiny_batch, tiny_stage_model
from test_query_ar_joint_checkpoint import _TinyBackbone, _TinyActionExpert, QueryArWarmStartCheckpointTest
from train_vla import run_optimizer_step_window
from utils.optical_flow_checkpoint import (
    COMPUTED_FLAGS, checkpoint_stage_metadata, read_flow_artifacts, record_stage_training_window,
)


def optimizer_window(model, *, empty=False):
    data = tiny_batch()
    if empty:
        data.update(flow_supervision_available=torch.tensor([False]), flow_target={}, flow_valid_mask={})
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0001)
    return run_optimizer_step_window(model=model, batches=[data], accelerator=Accelerator(cpu=True),
        optimizer=optimizer, lr_scheduler=torch.optim.lr_scheduler.StepLR(optimizer, 1),
        training_progress=0., loss_type=model.loss_type, vlm_loss_weight=1., action_expert_loss_weight=1.,
        optical_flow_config=model.optical_flow_config, training_stage=model.training_stage, next_global_step=1)


@pytest.mark.parametrize("empty", [False, True])
def test_explicit_aux_forward_only_computes_supervised_flow(empty):
    model = tiny_stage_model("stage2_aux")
    data = tiny_batch()
    if empty:
        data.update(flow_supervision_available=torch.tensor([False]), flow_target={}, flow_valid_mask={})
    # Fail if future refactoring accidentally routes stage2 to the Expert.
    with patch.object(model, "action_expert", side_effect=AssertionError("stage2 called Expert")), \
         patch.object(model.backbone, "forward", wraps=model.backbone.forward) as backbone:
        result = model(data, 0., loss_type="aux")
        assert backbone.call_args.kwargs["compute_vlm_loss"] is False
    assert result["slot_loss_computed"] is result["ar_loss_computed"] is result["fm_loss_computed"] is False
    assert result["flow_loss_computed"] is (not empty)
    assert not {"slot_loss", "ar_loss", "flow_matching_loss"}.intersection(result)
    torch.testing.assert_close(result["loss"], result["optical_flow_loss"])
    result["loss"].backward()
    if not empty:
        gradient = model.backbone.difference_query.weight.grad
        assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0


@pytest.mark.parametrize("stage", ["stage1_ar", "stage2_aux", "stage3_joint"])
def test_explicit_mode_conflicts_and_unknown(stage):
    model = tiny_stage_model(stage)
    with pytest.raises(ValueError, match="Unrecognized loss_type"):
        model(tiny_batch(), 0., loss_type="invalid")
    with pytest.raises(ValueError, match="mode mismatch"):
        model(tiny_batch(), 0., loss_type="vlm" if stage == "stage2_aux" else "aux")


def test_aux_cli_and_empty_training_history():
    from utils.cli_options import parse_train_options
    options = parse_train_options(["--training_stage", "stage2_aux", "--loss_type", "aux",
        "--optical_flow_aux_type", "dense_regression_v1", "--optical_flow_loss_weight", "1",
        "--num_flow_queries", "2", "--optical_flow_data_root", "/unused"])
    assert options.loss_type == "aux" and options.loss_type_explicit
    model = tiny_stage_model("stage2_aux")
    metrics = optimizer_window(model, empty=True)
    assert metrics["optimizer_update_skipped"] == 1
    assert not any(checkpoint_stage_metadata(model)[key] for key in COMPUTED_FLAGS)


def test_stage3_without_flow_records_only_ar_fm():
    from utils.optical_flow_config import OpticalFlowConfig
    model = tiny_stage_model("stage3_joint")
    model.optical_flow_aux = None
    model.optical_flow_config = OpticalFlowConfig()
    metrics = optimizer_window(model)
    assert metrics["ar_loss_computed"] is metrics["fm_loss_computed"] is True
    state = checkpoint_stage_metadata(model)
    assert state["provisional"] is True
    assert state["flow_loss_computed"] is state["slot_loss_computed"] is False


@pytest.fixture
def saveable_model(tmp_path):
    with patch("model.reasoning_vla_model.QwenVLBackbone", _TinyBackbone), \
         patch("model.reasoning_vla_model.FlowmatchingActionHead", _TinyActionExpert), \
         patch("model.flow_matching_action_head.FlowmatchingActionHead", _TinyActionExpert):
        def create(stage, source=None, resume=False):
            return ZR0Model(str(source or tmp_path / "base"), action_expert_name_or_path=None,
                action_expert_config=QueryArWarmStartCheckpointTest.action_config(), training_stage=stage,
                optical_flow_config=config() if stage != "stage1_ar" else None,
                use_difference_query=True, num_difference_queries=4, tune_vlm=True,
                tune_action_expert=stage == "stage3_joint",
                init_from_checkpoint=str(source) if source and not resume else None,
                resume_from_checkpoint=str(source) if resume else None)
        yield create


@pytest.mark.parametrize("stage", ["stage1_ar", "stage2_aux", "stage3_joint"])
def test_checkpoint_cumulative_state_save_load_and_init(tmp_path, saveable_model, stage):
    model = saveable_model(stage)
    initial = checkpoint_stage_metadata(model)
    assert not any(initial[key] for key in COMPUTED_FLAGS)
    model.save_pretrained(tmp_path / "untrained")
    assert not any(read_flow_artifacts(tmp_path / "untrained")[key] for key in COMPUTED_FLAGS)
    tiny = tiny_stage_model(stage)
    # Evaluation/forward alone cannot be mistaken for completed training windows.
    tiny(tiny_batch(), 0.)
    assert not any(checkpoint_stage_metadata(tiny)[key] for key in COMPUTED_FLAGS)
    metrics = optimizer_window(tiny)
    assert all(checkpoint_stage_metadata(tiny)[key] == metrics[key] for key in COMPUTED_FLAGS)
    record_stage_training_window(model, metrics)
    state = checkpoint_stage_metadata(model)
    assert state["ar_loss_computed"] is (stage != "stage2_aux")
    assert state["flow_loss_computed"] is (stage != "stage1_ar")
    assert state["fm_loss_computed"] is (stage == "stage3_joint")
    assert state["slot_loss_computed"] is False
    assert state["provisional"] is (stage == "stage3_joint")
    assert state["legacy_history_unknown"] is False
    if stage != "stage1_ar":
        later = optimizer_window(tiny, empty=True)
        assert later["flow_loss_computed"] is False
        record_stage_training_window(model, later)
        assert checkpoint_stage_metadata(model)["flow_loss_computed"] is True
        state = checkpoint_stage_metadata(model)
    path = tmp_path / "trained"
    model.save_pretrained(path)
    sidecar = read_flow_artifacts(path)
    ordinary = json.loads((path / "zr0_checkpoint_metadata.json").read_text())
    for key, value in state.items():
        assert sidecar[key] == ordinary[key] == value
    loaded = ZR0Model.from_pretrained(path, tune_vlm=True, tune_action_expert=stage == "stage3_joint")
    assert checkpoint_stage_metadata(loaded) == state
    assert checkpoint_stage_metadata(saveable_model(stage, path, resume=True)) == state
    initialized = saveable_model(stage, path)
    assert not any(checkpoint_stage_metadata(initialized)[key] for key in COMPUTED_FLAGS)
    assert initialized.source_stage_training_state == state
    if stage != "stage3_joint":
        next_stage = "stage2_aux" if stage == "stage1_ar" else "stage3_joint"
        transitioned = saveable_model(next_stage, path)
        assert not any(checkpoint_stage_metadata(transitioned)[key] for key in COMPUTED_FLAGS)
        assert transitioned.source_stage_training_state == state


def test_legacy_metadata_unknown_and_resume_preserves_marker(tmp_path, saveable_model):
    model = saveable_model("stage2_aux")
    model.save_pretrained(tmp_path)
    new_keys = {"stage_metadata_version", "loss_computed_scope", "legacy_history_unknown", "provisional",
                "ar_loss_computed", "flow_loss_computed", "fm_loss_computed"}
    for name in ("optical_flow_aux_config.json", "zr0_checkpoint_metadata.json"):
        path = tmp_path / name
        payload = json.loads(path.read_text())
        path.write_text(json.dumps({key: value for key, value in payload.items() if key not in new_keys}))
    with pytest.warns(RuntimeWarning, match="history is unknown"):
        restored = ZR0Model.from_pretrained(tmp_path, tune_vlm=True)
    state = checkpoint_stage_metadata(restored)
    assert state["legacy_history_unknown"] is True
    assert state["flow_loss_computed"] is state["ar_loss_computed"] is state["fm_loss_computed"] is None
    assert state["slot_loss_computed"] is False
    metrics = optimizer_window(tiny_stage_model("stage2_aux"))
    record_stage_training_window(restored, metrics)
    assert checkpoint_stage_metadata(restored)["flow_loss_computed"] is True
    assert checkpoint_stage_metadata(restored)["ar_loss_computed"] is None
    restored.save_pretrained(tmp_path / "resaved")
    assert read_flow_artifacts(tmp_path / "resaved")["legacy_history_unknown"] is True


@pytest.mark.parametrize("mutation", ["flow", "slot_integer", "missing"])
def test_checkpoint_metadata_inconsistent_copies_fail(tmp_path, saveable_model, mutation):
    saveable_model("stage3_joint").save_pretrained(tmp_path)
    path = tmp_path / "zr0_checkpoint_metadata.json"
    payload = json.loads(path.read_text())
    if mutation == "flow":
        payload["flow_loss_computed"] = True
    elif mutation == "slot_integer":
        payload["slot_loss_computed"] = 0
    else:
        del payload["ar_loss_computed"]
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="stage metadata mismatch"):
        read_flow_artifacts(tmp_path)
