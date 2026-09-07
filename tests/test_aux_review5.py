"""Saved Head removal and phase-aware static objective regressions."""

from dataclasses import replace
import hashlib
import json
import os
import shlex
import subprocess
from unittest.mock import patch

import pytest
import torch
from safetensors.torch import load_file, save_file

from model.reasoning_vla_model import ZR0Model
from utils.cli_options import parse_train_options
from utils.optical_flow_config import OpticalFlowConfig
from utils.optical_flow_checkpoint import json_hash, module_checksum
from utils.slot_config import SlotConfig
from test_query_ar_joint_checkpoint import _TinyBackbone, _TinyActionExpert
import test_query_ar_joint_checkpoint as checkpoint_helpers
from test_slot_integration import fixture_stats


@pytest.fixture
def checkpoints(tmp_path):
    with patch("model.reasoning_vla_model.QwenVLBackbone", _TinyBackbone), \
            patch("model.reasoning_vla_model.FlowmatchingActionHead", _TinyActionExpert), \
            patch("model.flow_matching_action_head.FlowmatchingActionHead", _TinyActionExpert):
        kwargs = dict(action_expert_name_or_path=None,
            action_expert_config=checkpoint_helpers.QueryArWarmStartCheckpointTest.action_config())
        one = ZR0Model(str(tmp_path / "base"), **kwargs, training_stage="stage1_ar",
            use_difference_query=True, num_difference_queries=32)
        one.save_pretrained(tmp_path / "stage1")
        models = {"stage1": one}
        for mode in ("slot", "flow", "both"):
            slot = SlotConfig(slot_aux_type="structured_slots_v1", slot_loss_weight=1.,
                stage2_aux_sampling="any_aux_valid" if mode == "both" else "slot_valid") if mode != "flow" else SlotConfig()
            flow = OpticalFlowConfig(num_flow_queries=8, flow_head_hidden_dim=16, flow_head_num_layers=1,
                optical_flow_aux_type="dense_regression_v1" if mode != "slot" else "none",
                optical_flow_loss_weight=1. if mode != "slot" else 0.)
            model = ZR0Model(str(tmp_path / "stage1"), **kwargs, training_stage="stage2_aux",
                slot_config=slot, optical_flow_config=flow, slot_supervision_stats=fixture_stats() if slot.enabled else None,
                init_from_checkpoint=str(tmp_path / "stage1"))
            model.save_pretrained(tmp_path / mode)
            models[mode] = model
        yield tmp_path, models, kwargs


def cli_args(source, mode, *, stage="stage3_joint", resume=False):
    args = ["--training_stage", stage, "--resume_from_checkpoint" if resume else "--init_from_checkpoint", str(source)]
    if stage == "stage3_joint":
        args += ["--tune_action_expert", "--tune_vlm"]
    if mode is not None:
        if mode not in {"slot", "both"}:
            args += ["--slot_aux_type", "none", "--slot_loss_weight", "0"]
        if mode not in {"flow", "both"}:
            args += ["--optical_flow_aux_type", "none", "--optical_flow_loss_weight", "0"]
    return args + ["--slot_supervision_dir", "/unused/slot", "--optical_flow_data_root", "/unused/flow"]


def configs(options):
    return tuple(cls(**{key: getattr(options, key) for key in cls.__dataclass_fields__})
                 for cls in (SlotConfig, OpticalFlowConfig))


@pytest.mark.parametrize("saved,target", [("slot", "none"), ("flow", "none"), ("both", "none"),
    ("both", "slot"), ("both", "flow"), ("both", "both")])
def test_cross_stage_removal_roundtrip_and_resume(checkpoints, saved, target):
    root, models, kwargs = checkpoints
    source = root / saved
    identity = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir() if p.is_file()}
    options = parse_train_options(cli_args(source, target))
    slot, flow = configs(options)
    model = ZR0Model(str(source), **kwargs, training_stage="stage3_joint", slot_config=slot,
        optical_flow_config=flow, init_from_checkpoint=str(source))
    assert model.slot_config.enabled == (target in {"slot", "both"})
    assert model.optical_flow_config.enabled == (target in {"flow", "both"})
    assert model.query_role_layout == models[saved].query_role_layout
    torch.testing.assert_close(model.backbone.difference_query.weight, models[saved].backbone.difference_query.weight, rtol=0, atol=0)
    assert model.num_difference_queries == 32 and model.query_role_layout["num_flow_queries"] == 8
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        expected_expert = _TinyActionExpert(None, True)
    assert module_checksum(model.action_expert) == module_checksum(expected_expert)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad])
    owned = {id(p) for group in optimizer.param_groups for p in group["params"]}
    for name, enabled in (("slot_aux", slot.enabled), ("optical_flow_aux", flow.enabled)):
        head = getattr(model, name)
        assert (head is not None) == enabled
        assert all(not key.startswith(name + ".") for key, p in model.named_parameters() if not enabled and id(p) in owned)
    destination = root / "target"
    model.save_pretrained(destination)
    restored = ZR0Model.from_pretrained(destination, tune_vlm=True, tune_action_expert=True)
    resumed = ZR0Model(str(destination), **kwargs, training_stage="stage3_joint", resume_from_checkpoint=str(destination))
    for other in (restored, resumed):
        assert module_checksum(model) == module_checksum(other)
        assert other.slot_config == slot and other.optical_flow_config == flow
        assert other.query_role_layout == model.query_role_layout
    assert {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir() if p.is_file()} == identity
    assert (destination / "slot_head.safetensors").exists() == slot.enabled
    assert (destination / "optical_flow_aux.safetensors").exists() == flow.enabled


def test_launcher_zero_is_explicit(checkpoints):
    root, _, _ = checkpoints
    env = dict(os.environ, INIT_CHECKPOINT=str(root / "both"), OUTPUT_DIR=str(root / "unused"),
        WITH_SLOT="0", WITH_FLOW="0", ACCELERATE_CONFIG="accelerate_configs/structured_slots_zero2_bf16.yaml",
        WANDB_PROJECT="test", WANDB_RUN_NAME="test")
    tokens = shlex.split(subprocess.check_output(["bash", "scripts/run_structured_slot_stage.sh", "stage3_joint", "--print-command"], env=env, text=True))
    args = tokens[tokens.index("train_vla.py") + 1:]
    for flag in ("--slot_aux_type", "--optical_flow_aux_type"):
        assert flag in args and args[args.index(flag) + 1] == "none"
    slot, flow = configs(parse_train_options(args))
    assert not slot.enabled and not flow.enabled and flow.num_flow_queries == 8


@pytest.mark.parametrize("resume", [False, True])
def test_same_stage_or_ordinary_loading_cannot_remove_heads(checkpoints, resume):
    root, models, kwargs = checkpoints
    source = root / "both"
    slot = replace(models["both"].slot_config, slot_aux_type="none", slot_loss_weight=0.)
    with pytest.raises(ValueError, match="conflict|discard"):
        ZR0Model(str(source), **kwargs, training_stage="stage2_aux" if resume else "stage3_joint",
            slot_config=slot, **({"resume_from_checkpoint": str(source)} if resume else {}))
    with pytest.raises(SystemExit):
        parse_train_options(cli_args(source, "none", stage="stage2_aux", resume=resume))


def test_unspecified_heads_inherit(checkpoints):
    root, models, kwargs = checkpoints
    slot, flow = configs(parse_train_options(cli_args(root / "both", None)))
    assert slot == models["both"].slot_config and flow == models["both"].optical_flow_config
    model = ZR0Model(str(root / "both"), **kwargs, training_stage="stage3_joint", init_from_checkpoint=str(root / "both"))
    assert model.slot_config == slot and model.optical_flow_config == flow


def test_model_explicit_none_preserves_unspecified_head_settings(checkpoints):
    root, models, kwargs = checkpoints
    source = root / "both"
    model = ZR0Model(str(source), **kwargs, training_stage="stage3_joint", init_from_checkpoint=str(source),
        slot_config=SlotConfig(), optical_flow_config=OpticalFlowConfig())
    assert not model.slot_config.enabled and not model.optical_flow_config.enabled
    assert model.query_role_layout == models["both"].query_role_layout
    assert model.optical_flow_config.flow_head_hidden_dim == 16
    assert model.slot_config.stage2_aux_sampling == "any_aux_valid"


def test_cli_none_without_explicit_weight_disables_and_preserves_roles(checkpoints):
    root, models, _ = checkpoints
    arguments = cli_args(root / "both", None) + ["--slot_aux_type", "none", "--optical_flow_aux_type", "none"]
    slot, flow = configs(parse_train_options(arguments))
    assert not slot.enabled and slot.slot_loss_weight == 0.
    assert not flow.enabled and flow.optical_flow_loss_weight == 0. and flow.num_flow_queries == 8
    assert flow.flow_head_hidden_dim == models["both"].optical_flow_config.flow_head_hidden_dim


def test_disabled_config_is_saved_and_resume_cannot_enable_missing_head(checkpoints):
    root, models, kwargs = checkpoints
    source = root / "slot"
    with pytest.raises(ValueError, match="conflict|resume"):
        ZR0Model(str(source), **kwargs, training_stage="stage2_aux", resume_from_checkpoint=str(source),
            optical_flow_config=replace(models["slot"].optical_flow_config,
                optical_flow_aux_type="dense_regression_v1", optical_flow_loss_weight=1.))


@pytest.mark.parametrize("damage", ["slot_missing", "flow_missing", "slot_hash", "flow_hash", "flow_shape", "query_layout",
    "slot_schema", "flow_schema"])
def test_head_removal_still_rejects_corrupt_source(checkpoints, damage, capsys):
    root, _, _ = checkpoints
    source = root / "both"
    if damage.endswith("missing"):
        (source / ("slot_head.safetensors" if damage.startswith("slot") else "optical_flow_aux_config.json")).unlink()
    elif damage.endswith("hash"):
        path = source / ("slot_head.safetensors" if damage.startswith("slot") else "optical_flow_aux.safetensors")
        path.write_bytes(path.read_bytes() + b"corruption")
    elif damage == "flow_shape":
        path = source / "optical_flow_aux.safetensors"
        state = load_file(path)
        key = next(key for key in state if key != "input_projection.weight" and state[key].ndim > 0)
        state[key] = state[key][:-1]
        save_file(state, path)
        config_path = source / "optical_flow_aux_config.json"
        payload = json.loads(config_path.read_text())
        payload["weights_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        config_path.write_text(json.dumps(payload))
    elif damage.endswith("schema"):
        is_slot = damage.startswith("slot")
        path = source / ("slot_aux_config.json" if is_slot else "optical_flow_aux_config.json")
        payload = json.loads(path.read_text())
        del payload["config"]["slot_presence_weight" if is_slot else "flow_motion_loss_weight"]
        if is_slot:
            payload.pop("integrity_sha256")
            payload["integrity_sha256"] = json_hash(payload)
        else:
            payload["config_sha256"] = json_hash(payload["config"])
        path.write_text(json.dumps(payload))
    else:
        path = source / "zr0_checkpoint_metadata.json"
        payload = json.loads(path.read_text())
        payload["query_role_layout"]["num_flow_queries"] = 7
        path.write_text(json.dumps(payload))
    with pytest.raises(SystemExit):
        parse_train_options(cli_args(source, "none"))
    message = capsys.readouterr().err
    expected = {"slot_missing": "incomplete", "flow_missing": "without OF configuration",
        "slot_hash": "integrity", "flow_hash": "hash", "flow_shape": "shape", "query_layout": "layout",
        "slot_schema": "config", "flow_schema": "config"}
    assert expected[damage] in message


def zero_config(case):
    slot = SlotConfig(slot_aux_type="structured_slots_v1", slot_loss_weight=1.)
    if case == "outer":
        return replace(slot, slot_loss_weight=0.)
    return replace(slot, slot_task_weights={f"Q{i}": float(i == 9) for i in range(1, 10)},
        slot_presence_weight=0., slot_obstacle_bbox_weight=0., slot_risk_weight=0.)


def config_args(slot):
    args = []
    for key, value in slot.to_dict().items():
        args += ["--" + key, json.dumps(value) if isinstance(value, dict) else str(value)]
    return args


@pytest.mark.parametrize("case", ["outer", "q9"])
@pytest.mark.parametrize("entry", ["cli", "model"])
def test_stage2_static_zero_rejected_before_backbone(checkpoints, case, entry, capsys):
    root, _, kwargs = checkpoints
    source = root / "stage1"
    slot = zero_config(case)
    with patch("model.reasoning_vla_model.QwenVLBackbone", side_effect=AssertionError("large model constructed")):
        if entry == "cli":
            with pytest.raises(SystemExit) as error:
                parse_train_options(cli_args(source, "slot", stage="stage2_aux") + config_args(slot) +
                    ["--use_difference_query", "--num_flow_queries", "8"])
            assert error.value.code == 2
            message = capsys.readouterr().err
        else:
            with pytest.raises(ValueError) as error:
                ZR0Model(str(source), **kwargs, training_stage="stage2_aux", slot_config=slot,
                    optical_flow_config=OpticalFlowConfig(num_flow_queries=8), slot_supervision_stats=fixture_stats(),
                    init_from_checkpoint=str(source))
            message = str(error.value)
    assert "stage2_aux" in message and "weight" in message and "structured_slots_v1" in message


@pytest.mark.parametrize("case", ["flow_alternative", "q1_alternative", "default", "stage3_ablation",
    "q9_presence", "q9_bbox", "q9_risk", "q1_base", "q3_base", "q5_base"])
def test_static_alternatives_accepted_by_cli_and_model(checkpoints, case):
    root, _, kwargs = checkpoints
    source = root / "stage1"
    slot = zero_config("outer" if case == "flow_alternative" else "q9")
    flow = OpticalFlowConfig(num_flow_queries=8)
    stage = "stage3_joint" if case == "stage3_ablation" else "stage2_aux"
    if case == "flow_alternative":
        flow = replace(flow, optical_flow_aux_type="dense_regression_v1", optical_flow_loss_weight=1., flow_head_hidden_dim=16, flow_head_num_layers=1)
    elif case == "q1_alternative":
        slot = replace(slot, slot_task_weights={**slot.slot_task_weights, "Q1": 1.})
    elif case == "default":
        slot = SlotConfig(slot_aux_type="structured_slots_v1", slot_loss_weight=1.)
    elif case.startswith("q9_"):
        field = {"q9_presence": "slot_presence_weight", "q9_bbox": "slot_obstacle_bbox_weight", "q9_risk": "slot_risk_weight"}[case]
        slot = replace(slot, **{field: 1.})
    elif case.endswith("_base"):
        q = case.split("_")[0].upper()
        slot = replace(slot, slot_task_weights={f"Q{i}": float(f"Q{i}" == q) for i in range(1, 10)},
            slot_progress_monotonic_weight=0., slot_bbox_giou_weight=0., slot_q5_consistency_weight=0.)
    arguments = cli_args(source, None, stage=stage) + config_args(slot) + ["--use_difference_query"]
    for key, value in flow.to_dict().items():
        arguments += ["--" + key, str(value)]
    parsed_slot, parsed_flow = configs(parse_train_options(arguments))
    model = ZR0Model(str(source), **kwargs, training_stage=stage, slot_config=parsed_slot,
        optical_flow_config=parsed_flow, slot_supervision_stats=fixture_stats(), init_from_checkpoint=str(source))
    assert model.slot_config == slot and model.optical_flow_config == flow
