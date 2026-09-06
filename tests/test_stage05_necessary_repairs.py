"""Focused regressions for AR resume and direct generic-contract loading."""

import copy
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from model.reasoning_vla_model import ZR0Model
from test_stage05_checkpoint_contract import (
    _hash_json, _make_checkpoint, _make_joint_checkpoint, _metadata,
)
from utils.action_expert_config import architecture_config_hash, load_action_expert_config
from utils.cli_options import parse_train_options
from utils.stage05_checkpoint_contract import (
    STAGE05_AR_JOINT_CONTRACT_KEY, STAGE05_AR_RESUME,
    validate_stage05_checkpoint_for_purpose,
)


def _write_json(path, payload):
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _write_ar_training_state(checkpoint):
    save_file(
        {"difference_query": torch.zeros(32, 2048)},
        checkpoint / "difference_query.safetensors",
    )
    torch.save({"last_epoch": 1, "_step_count": 2}, checkpoint / "scheduler.pt")
    torch.save(
        {
            "last_global_step": 1, "global_steps": 1,
            "dp_world_size": 1,
            "module": {"backbone.model.weight": torch.ones(1)},
            "param_shapes": [{"backbone.model.weight": torch.Size([1])}],
        },
        checkpoint / "mp_rank_00_model_states.pt",
    )
    torch.save(
        {"optimizer_state_dict": {
            "zero_stage": 2, "partition_count": [1],
            "single_partition_of_fp32_groups": [torch.ones(1)],
            "base_optimizer_state": {
                "state": {0: {"step": torch.tensor(1.), "exp_avg": torch.zeros(1),
                              "exp_avg_sq": torch.zeros(1)}},
                "param_groups": [{"params": [0]}],
            },
        }},
        checkpoint / "zero_pp_rank_0_mp_rank_00_optim_states.pt",
    )


def _options(checkpoint, *, purpose=None, loss_type="action", resume=False):
    return SimpleNamespace(
        checkpoint_load_purpose=purpose, resume_training=resume,
        vlm_name_or_path=str(checkpoint),
        action_expert_name_or_path=None if loss_type == "vlm" else str(checkpoint),
        action_expert_config_path=None, loss_type=loss_type,
        action_horizon=32, max_pad_state_and_action_length=64,
        use_difference_query=None, num_difference_queries=32,
        vlm_attention_backend="sdpa", seed=42,
    )


def _validate_ar(checkpoint, **overrides):
    return validate_stage05_checkpoint_for_purpose(
        checkpoint, **{
            "purpose": STAGE05_AR_RESUME, "resume_training": True,
            "requested_action_horizon": 32, **overrides,
        },
    )


def test_ar_resume_valid_contract_and_training_entry(tmp_path):
    from train_vla import resolve_action_expert_config

    checkpoint = _make_checkpoint(tmp_path)
    _write_ar_training_state(checkpoint)
    resolved = _validate_ar(checkpoint)
    options = _options(checkpoint, purpose=STAGE05_AR_RESUME, loss_type="vlm", resume=True)
    assert resolve_action_expert_config(options).to_dict() == resolved.config.to_dict()
    assert resolved.config.action_horizon == 32
    assert not (checkpoint / "action_expert.safetensors").exists()
    assert _metadata(checkpoint)[STAGE05_AR_JOINT_CONTRACT_KEY]["runtime_contract"]["purpose"] == "stage05_ar_to_joint"


@pytest.mark.parametrize("fault", [
    "kind", "manifest_loss", "manifest_hash", "contract_hash", "horizon",
    "dq_backend", "dq_shape", "dq_missing", "expert_weights",
    "scheduler_missing", "scheduler_corrupt", "scheduler_step",
    "client_missing", "client_corrupt", "client_step", "client_module",
    "optimizer_missing", "optimizer_corrupt", "optimizer_empty",
    "optimizer_step", "optimizer_partitions",
])
def test_ar_resume_faults_fail_before_accelerator(tmp_path, monkeypatch, fault):
    import train_vla

    checkpoint = _make_checkpoint(tmp_path)
    _write_ar_training_state(checkpoint)
    metadata_path = checkpoint / "zr0_checkpoint_metadata.json"
    metadata = _metadata(checkpoint)
    options = _options(checkpoint, purpose=STAGE05_AR_RESUME, loss_type="vlm", resume=True)
    if fault == "kind":
        metadata["checkpoint_kind"] = "joint"
        _write_json(metadata_path, metadata)
    elif fault.startswith("manifest_"):
        path = checkpoint / "resolved_dataset_manifest.json"
        manifest = json.loads(path.read_text())
        manifest["loss_type"] = "vlm_and_action" if fault == "manifest_loss" else "action"
        _write_json(path, manifest)
    elif fault == "contract_hash":
        metadata[STAGE05_AR_JOINT_CONTRACT_KEY]["content_hash"] = "broken"
        _write_json(metadata_path, metadata)
    elif fault == "horizon":
        options.action_horizon = 10
    elif fault == "dq_backend":
        path = checkpoint / "difference_query_config.json"
        payload = json.loads(path.read_text())
        payload["attention_backend"] = "eager"
        _write_json(path, payload)
    elif fault == "dq_shape":
        save_file({"difference_query": torch.zeros(31, 2048)}, checkpoint / "difference_query.safetensors")
    elif fault == "dq_missing":
        (checkpoint / "difference_query.safetensors").unlink()
    elif fault == "expert_weights":
        save_file({"unexpected": torch.zeros(1)}, checkpoint / "action_expert.safetensors")
    else:
        category, problem = fault.split("_", 1)
        path = checkpoint / {
            "scheduler": "scheduler.pt", "client": "mp_rank_00_model_states.pt",
            "optimizer": "zero_pp_rank_0_mp_rank_00_optim_states.pt",
        }[category]
        if problem == "missing":
            path.unlink()
        elif problem == "corrupt":
            path.write_bytes(b"corrupt")
        else:
            state = torch.load(path, weights_only=False)
            if category == "scheduler":
                state["last_epoch"] = 2
            elif category == "client":
                state["global_steps" if problem == "step" else "module"] = 2 if problem == "step" else {}
            elif problem == "empty":
                state = {"irrelevant": True}
            elif problem == "partitions":
                state["optimizer_state_dict"]["partition_count"] = [4]
            else:
                state["optimizer_state_dict"]["base_optimizer_state"]["state"][0]["step"] = 2
            torch.save(state, path)
    accelerator = Mock(side_effect=AssertionError("Accelerator must not initialize"))
    monkeypatch.setattr(train_vla, "Accelerator", accelerator)
    with pytest.raises(ValueError):
        train_vla.train(options)
    accelerator.assert_not_called()


@pytest.mark.parametrize("arguments", [[], ["--resume_training", "--loss_type", "action"]])
def test_ar_resume_cli_rejects_wrong_mode(arguments):
    with pytest.raises(SystemExit):
        parse_train_options(["--checkpoint_load_purpose", STAGE05_AR_RESUME, *arguments])


def test_ar_resume_cli_and_explicit_query_conflicts(tmp_path):
    from train_vla import resolve_action_expert_config

    parsed = parse_train_options([
        "--checkpoint_load_purpose", STAGE05_AR_RESUME,
        "--resume_training", "--loss_type", "vlm", "--tune_vlm",
    ])
    assert parsed.checkpoint_load_purpose == STAGE05_AR_RESUME
    checkpoint = _make_checkpoint(tmp_path)
    _write_ar_training_state(checkpoint)
    with pytest.raises(ValueError, match="resume_training"):
        _validate_ar(checkpoint, resume_training=False)
    options = _options(checkpoint, purpose=STAGE05_AR_RESUME, loss_type="vlm", resume=True)
    options.use_difference_query = False
    with pytest.raises(ValueError):
        resolve_action_expert_config(options)
    with pytest.raises(ValueError, match="loss_type=vlm"):
        ZR0Model(str(checkpoint), None, load_action_expert_config(checkpoint / "action_expert_config.json").config,
                 checkpoint_load_purpose=STAGE05_AR_RESUME, resume_training=True)


def _generic_checkpoint(tmp_path):
    checkpoint = _make_joint_checkpoint(tmp_path)
    metadata = _metadata(checkpoint)
    metadata.pop(STAGE05_AR_JOINT_CONTRACT_KEY)
    (checkpoint / "resolved_dataset_manifest.json").unlink()
    resolved = load_action_expert_config(checkpoint / "action_expert_config.json")
    with safe_open(str(checkpoint / "action_expert.safetensors"), framework="pt") as source:
        shapes = {name: source.get_slice(name).get_shape() for name in source.keys()}
    runtime = {"purpose": "legacy", "checkpoint_kind": "joint", "action_horizon": 32,
               "source_action_horizon": 32, "target_action_horizon": 32}
    contract = {
        "version": 1, "architecture_hash": architecture_config_hash(resolved.payload),
        "runtime_contract": runtime, "runtime_contract_hash": _hash_json(runtime),
        "config_file": "action_expert_config.json", "raw_file_sha256": resolved.source_sha256,
        "canonical_sha256": resolved.parsed_sha256, "state_dict_shapes": shapes,
    }
    contract["content_hash"] = _hash_json(contract)
    metadata["action_expert_contract"] = contract
    _write_json(checkpoint / "zr0_checkpoint_metadata.json", metadata)
    save_file({"difference_query": torch.zeros(32, 4)}, checkpoint / "difference_query.safetensors")
    return checkpoint, resolved.config


@pytest.mark.parametrize("fault", [
    "raw_file_sha256", "canonical_sha256", "content_hash", "architecture_hash",
    "runtime_contract_hash", "runtime_horizon", "hidden_size", "weights_keys",
    "weights_shape", "forged_shapes", "null_contract", "config_dimension",
])
def test_generic_corruption_fails_before_model_and_accelerator(tmp_path, monkeypatch, fault):
    import model.reasoning_vla_model as model_module
    import train_vla

    checkpoint, config = _generic_checkpoint(tmp_path)
    metadata = _metadata(checkpoint)
    contract = metadata["action_expert_contract"]
    if fault in contract:
        contract[fault] = "broken"
    elif fault == "runtime_horizon":
        contract["runtime_contract"]["action_horizon"] = 10
        contract["runtime_contract_hash"] = _hash_json(contract["runtime_contract"])
    elif fault == "hidden_size":
        _write_json(checkpoint / "config.json", {"text_config": {"hidden_size": 8}})
    elif fault in {"weights_keys", "weights_shape", "forged_shapes"}:
        from safetensors.torch import load_file

        weights = load_file(checkpoint / "action_expert.safetensors")
        name = next(iter(weights))
        if fault == "weights_keys":
            weights.pop(name)
        else:
            weights[name] = torch.zeros(1)
        save_file(weights, checkpoint / "action_expert.safetensors")
        if fault == "forged_shapes":
            contract["state_dict_shapes"][name] = [1]
    elif fault == "null_contract":
        metadata["action_expert_contract"] = None
    elif fault == "config_dimension":
        payload = json.loads((checkpoint / "action_expert_config.json").read_text())
        payload["action_dim"] = 63
        _write_json(checkpoint / "action_expert_config.json", payload)
    if fault != "content_hash":
        contract["content_hash"] = _hash_json({key: value for key, value in contract.items() if key != "content_hash"})
    _write_json(checkpoint / "zr0_checkpoint_metadata.json", metadata)
    backbone = Mock(side_effect=AssertionError("Backbone must not initialize"))
    accelerator = Mock(side_effect=AssertionError("Accelerator must not initialize"))
    monkeypatch.setattr(model_module, "QwenVLBackbone", backbone)
    monkeypatch.setattr(train_vla, "Accelerator", accelerator)
    with pytest.raises(ValueError):
        ZR0Model(str(checkpoint), str(checkpoint), config)
    with pytest.raises(ValueError):
        ZR0Model.from_pretrained(checkpoint, allow_ar_warm_start=True)
    with pytest.raises(ValueError):
        train_vla.train(_options(checkpoint))
    backbone.assert_not_called()
    accelerator.assert_not_called()


def test_direct_generic_uses_verified_config_and_preserves_fresh_horizon(tmp_path, monkeypatch):
    import model.reasoning_vla_model as model_module

    checkpoint, config = _generic_checkpoint(tmp_path)
    seen = []
    validator = model_module.validate_generic_action_expert_contract

    def capture(*args, **kwargs):
        verified = validator(*args, **kwargs)
        seen.append(verified.config)
        return verified

    class Backbone(torch.nn.Module):
        def __init__(self, *_args, **_kwargs):
            super().__init__()
            self.model = SimpleNamespace(config=SimpleNamespace(text_config=SimpleNamespace(hidden_size=4)))

    monkeypatch.setattr(model_module, "validate_generic_action_expert_contract", capture)
    monkeypatch.setattr(model_module, "QwenVLBackbone", Backbone)
    for horizon in (32, 10):
        requested = copy.deepcopy(config)
        requested.action_horizon = horizon
        model = ZR0Model(str(checkpoint), str(checkpoint), requested)
        assert model.action_expert_config is seen[-1]
        assert model.action_expert_config is not requested
        assert model.action_expert_config.action_horizon == horizon
        assert model.action_expert_source_action_horizon == 32
    invalid = copy.deepcopy(config)
    invalid.diffusion_transformer_cfg["dropout"] = 0.1
    with pytest.raises(ValueError, match="requested Action Expert config"):
        ZR0Model(str(checkpoint), str(checkpoint), invalid)
