import copy
import hashlib
import json
from pathlib import Path

import pytest

from utils.action_expert_config import load_action_expert_config
from utils.dataset_manifest import build_resolved_dataset_manifest, write_resolved_dataset_manifest
from utils.dataset_spec import ObservationContract, ResolvedDatasetSpec
from utils.stage05_checkpoint_contract import (
    DOWNSTREAM_FINETUNE,
    INFERENCE,
    STAGE05_AR_TO_JOINT,
    STAGE05_AR_JOINT_CONTRACT_KEY,
    STAGE05_JOINT_RESUME,
    STAGE05_DATASET_ENTRIES,
    build_stage05_ar_joint_contract,
    validate_checkpoint_load_purpose_arguments,
    validate_stage05_joint_warm_start,
    validate_stage05_checkpoint_for_purpose,
    validate_stage05_resume_artifacts,
)


ROOT = Path(__file__).resolve().parents[1]
AUTHORITY_CONFIG = ROOT / "configs/stage05_four_dataset_action_expert.json"


def _hash_json(value):
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=True, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def _stage05_manifest(loss_type="vlm"):
    specs = [
        ResolvedDatasetSpec(
            dataset_entry=name,
            dataset_path=f"/fixture/{name}",
            dataset_type="vla",
            adapter="stage05_mixed_pretraining",
            target_text_field="train_data",
            camera_keys=("main", "wrist"),
            grounding_camera_keys=(),
            state_key="observation.state",
            action_key="action",
            state_dim=7,
            action_dim=7,
            action_horizon=32,
            stats_path=None,
            stats_key=name,
            normalization="q01_q99",
            normalization_stats=None,
            sample_ratio=1.0,
            training_eligibility_exists=True,
            training_eligibility_used=True,
            data_version="fixture",
            observation_contract=ObservationContract(
                version=1,
                window_size=1,
                history_order="single_current_frame",
                history_stride="not_applicable",
            ),
        )
        for name in STAGE05_DATASET_ENTRIES
    ]
    return build_resolved_dataset_manifest(specs, loss_type)


def _make_checkpoint(tmp_path: Path) -> Path:
    checkpoint = tmp_path / "stage05-ar"
    checkpoint.mkdir(parents=True)
    source = AUTHORITY_CONFIG.read_bytes()
    (checkpoint / "action_expert_config.json").write_bytes(source)
    (checkpoint / "config.json").write_text(
        json.dumps({"text_config": {"hidden_size": 2048}}), encoding="utf-8"
    )
    (checkpoint / "difference_query_config.json").write_text(
        json.dumps(
            {
                "version": 1,
                "enabled": True,
                "num_difference_queries": 32,
                "hidden_size": 2048,
                "attention_backend": "sdpa",
            }
        ),
        encoding="utf-8",
    )
    (checkpoint / "difference_query.safetensors").write_bytes(b"query fixture")
    manifest = _stage05_manifest()
    write_resolved_dataset_manifest(checkpoint, manifest)
    resolved = load_action_expert_config(AUTHORITY_CONFIG).payload
    contract = build_stage05_ar_joint_contract(
        checkpoint_directory=checkpoint,
        resolved_action_expert_config=resolved,
        source_config_sha256=hashlib.sha256(source).hexdigest(),
        resolved_dataset_manifest=manifest,
        vlm_hidden_size=2048,
        num_difference_queries=32,
    )
    metadata = {
        "version": 1,
        "checkpoint_kind": "ar_only",
        "action_expert": {
            "status": "not_constructed_future_joint_config_reference",
            "config_file": "action_expert_config.json",
            "config_sha256": contract["action_expert_config"]["canonical_sha256"],
            "weights_file": None,
        },
        STAGE05_AR_JOINT_CONTRACT_KEY: contract,
    }
    (checkpoint / "zr0_checkpoint_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return checkpoint


def _make_joint_checkpoint(tmp_path: Path) -> Path:
    # Use a small but complete Action Expert so the resume validator can read
    # and compare a real safetensors state_dict without allocating the 2B
    # production head in this unit test.
    checkpoint = tmp_path / "stage05-joint"
    checkpoint.mkdir(parents=True)
    payload = json.loads(AUTHORITY_CONFIG.read_text(encoding="utf-8"))
    payload.update(
        {
            "vlm_output_embedding_dim": 4,
            "action_or_state_token_embedding_dim": 4,
            "mlp_hidden_size": 4,
            "max_seq_len": 64,
        }
    )
    payload["diffusion_transformer_cfg"].update(
        {
            "num_attention_heads": 1,
            "attention_head_dim": 4,
            "output_dim": 4,
            "num_layers": 1,
            "dropout": 0.0,
            "max_num_positional_embeddings": 64,
            "final_dropout": False,
        }
    )
    config_bytes = (json.dumps(payload, sort_keys=True) + "\n").encode()
    (checkpoint / "action_expert_config.json").write_bytes(config_bytes)
    (checkpoint / "config.json").write_text(
        json.dumps({"text_config": {"hidden_size": 4}}), encoding="utf-8"
    )
    (checkpoint / "difference_query_config.json").write_text(
        json.dumps(
            {
                "version": 1,
                "enabled": True,
                "num_difference_queries": 32,
                "hidden_size": 4,
                "attention_backend": "sdpa",
            }
        ),
        encoding="utf-8",
    )
    (checkpoint / "difference_query.safetensors").write_bytes(b"query fixture")
    metadata_path = checkpoint / "zr0_checkpoint_metadata.json"
    metadata = {
        "version": 1,
        "checkpoint_kind": "joint",
        "action_expert": {
            "status": "constructed",
            "config_file": "action_expert_config.json",
            "config_sha256": _hash_json(payload),
            "weights_file": "action_expert.safetensors",
        },
    }
    joint_manifest = _stage05_manifest("vlm_and_action")
    write_resolved_dataset_manifest(checkpoint, joint_manifest)
    contract = build_stage05_ar_joint_contract(
        checkpoint_directory=checkpoint,
        resolved_action_expert_config=load_action_expert_config(checkpoint / "action_expert_config.json").payload,
        source_config_sha256=hashlib.sha256(config_bytes).hexdigest(),
        resolved_dataset_manifest=joint_manifest,
        vlm_hidden_size=4,
        num_difference_queries=32,
        checkpoint_kind="joint",
        runtime_purpose=STAGE05_JOINT_RESUME,
    )
    metadata[STAGE05_AR_JOINT_CONTRACT_KEY] = contract
    from model.flow_matching_action_head import FlowmatchingActionHead
    from safetensors.torch import save_file

    expert = FlowmatchingActionHead(
        load_action_expert_config(checkpoint / "action_expert_config.json").config,
        True,
    )
    save_file(expert.state_dict(), checkpoint / "action_expert.safetensors")
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return checkpoint


def _metadata(checkpoint):
    return json.loads(
        (checkpoint / "zr0_checkpoint_metadata.json").read_text(encoding="utf-8")
    )


def _write_metadata(checkpoint, metadata, *, refresh_contract_hash=True):
    if refresh_contract_hash:
        contract = metadata[STAGE05_AR_JOINT_CONTRACT_KEY]
        contract["content_hash"] = _hash_json(
            {key: value for key, value in contract.items() if key != "content_hash"}
        )
    (checkpoint / "zr0_checkpoint_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def test_valid_stage05_contract_accepts_matching_external_config(tmp_path):
    checkpoint = _make_checkpoint(tmp_path)
    resolved = validate_stage05_joint_warm_start(checkpoint, AUTHORITY_CONFIG)
    assert resolved.config.action_horizon == 32
    assert resolved.payload["diffusion_transformer_cfg"]["num_layers"] == 9
    assert resolved.payload["diffusion_transformer_cfg"]["dropout"] == 0.2
    assert not (checkpoint / "action_expert.safetensors").exists()
    contract_runtime = _metadata(checkpoint)[STAGE05_AR_JOINT_CONTRACT_KEY]["runtime_contract"]
    assert contract_runtime["training_stage"] == "ar_only"
    assert contract_runtime["loss_type"] == "vlm"


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.pop("raw_file_sha256"), "metadata is incomplete"),
        (lambda value: value.pop("canonical_sha256"), "metadata is incomplete"),
    ],
)
def test_missing_config_hashes_fail_fast(tmp_path, mutation, message):
    checkpoint = _make_checkpoint(tmp_path)
    metadata = _metadata(checkpoint)
    mutation(metadata[STAGE05_AR_JOINT_CONTRACT_KEY]["action_expert_config"])
    _write_metadata(checkpoint, metadata)
    with pytest.raises(ValueError, match=message):
        validate_stage05_joint_warm_start(checkpoint, AUTHORITY_CONFIG)


def test_missing_or_tampered_checkpoint_config_fails_fast(tmp_path):
    checkpoint = _make_checkpoint(tmp_path)
    (checkpoint / "action_expert_config.json").unlink()
    with pytest.raises(ValueError, match="config is missing"):
        validate_stage05_joint_warm_start(checkpoint, AUTHORITY_CONFIG)

    checkpoint = _make_checkpoint(tmp_path / "tampered")
    (checkpoint / "action_expert_config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="raw hash mismatch"):
        validate_stage05_joint_warm_start(checkpoint, AUTHORITY_CONFIG)


def test_tampered_contract_metadata_hash_fails_fast(tmp_path):
    checkpoint = _make_checkpoint(tmp_path)
    metadata = _metadata(checkpoint)
    metadata[STAGE05_AR_JOINT_CONTRACT_KEY]["content_hash"] = "0" * 64
    _write_metadata(checkpoint, metadata, refresh_contract_hash=False)
    with pytest.raises(ValueError, match="metadata hash mismatch"):
        validate_stage05_joint_warm_start(checkpoint, AUTHORITY_CONFIG)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda payload: payload["diffusion_transformer_cfg"].__setitem__(
            "num_layers", 8
        ),
        lambda payload: payload["diffusion_transformer_cfg"].__setitem__(
            "dropout", 0.1
        ),
        lambda payload: (
            payload["diffusion_transformer_cfg"].__setitem__(
                "num_attention_heads", 16
            ),
            payload["diffusion_transformer_cfg"].__setitem__(
                "attention_head_dim", 128
            ),
        ),
        lambda payload: payload["diffusion_transformer_cfg"].__setitem__(
            "max_num_positional_embeddings", 256
        ),
    ],
)
def test_any_external_structural_drift_fails_canonical_hash(tmp_path, mutate):
    checkpoint = _make_checkpoint(tmp_path)
    payload = copy.deepcopy(json.loads(AUTHORITY_CONFIG.read_text(encoding="utf-8")))
    mutate(payload)
    external = tmp_path / "drifted.json"
    external.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="canonical hash does not match"):
        validate_stage05_joint_warm_start(checkpoint, external)


def test_legacy_tabletop_ar_checkpoint_has_explicit_compatibility_error(tmp_path):
    checkpoint = _make_checkpoint(tmp_path)
    metadata = _metadata(checkpoint)
    metadata.pop(STAGE05_AR_JOINT_CONTRACT_KEY)
    _write_metadata(checkpoint, metadata, refresh_contract_hash=False)
    with pytest.raises(ValueError, match="legacy Tabletop AR checkpoints are not accepted"):
        validate_stage05_joint_warm_start(checkpoint, AUTHORITY_CONFIG)


def test_expert_weights_are_forbidden_in_stage05_ar_contract(tmp_path):
    checkpoint = _make_checkpoint(tmp_path)
    (checkpoint / "action_expert.safetensors").write_bytes(b"must not exist")
    with pytest.raises(ValueError, match="unexpectedly contains Expert weights"):
        validate_stage05_joint_warm_start(checkpoint, AUTHORITY_CONFIG)


def test_joint_resume_and_downstream_horizon_override_use_explicit_purposes(tmp_path):
    checkpoint = _make_joint_checkpoint(tmp_path)
    external_config = tmp_path / "external-action-config.json"
    external_config.write_bytes((checkpoint / "action_expert_config.json").read_bytes())
    resumed = validate_stage05_checkpoint_for_purpose(
        checkpoint,
        purpose=STAGE05_JOINT_RESUME,
        requested_action_horizon=32,
        resume_training=True,
    )
    assert resumed.config.action_horizon == 32
    downstream = validate_stage05_checkpoint_for_purpose(
        checkpoint,
        purpose=DOWNSTREAM_FINETUNE,
        external_config_path=external_config,
        requested_action_horizon=10,
    )
    assert downstream.config.action_horizon == 10

    with pytest.raises(ValueError, match="cannot be used as downstream resume"):
        validate_stage05_checkpoint_for_purpose(
            checkpoint,
            purpose=DOWNSTREAM_FINETUNE,
            requested_action_horizon=32,
            resume_training=True,
        )


def test_legacy_action_only_is_a_valid_fresh_downstream_source(tmp_path):
    checkpoint = _make_joint_checkpoint(tmp_path)
    metadata = _metadata(checkpoint)
    metadata.pop(STAGE05_AR_JOINT_CONTRACT_KEY)
    metadata["checkpoint_kind"] = "action_only"
    (checkpoint / "resolved_dataset_manifest.json").unlink()
    _write_metadata(checkpoint, metadata, refresh_contract_hash=False)

    resolved = validate_stage05_checkpoint_for_purpose(
        checkpoint,
        purpose=DOWNSTREAM_FINETUNE,
        requested_action_horizon=10,
    )
    assert resolved.source_action_horizon == 32
    assert resolved.config.action_horizon == 10


def test_stage05_manifest_identity_cannot_fall_back_when_metadata_is_missing(tmp_path):
    checkpoint = _make_joint_checkpoint(tmp_path)
    (checkpoint / "zr0_checkpoint_metadata.json").unlink()
    with pytest.raises(ValueError, match="Stage05 dataset identity.*missing"):
        validate_stage05_checkpoint_for_purpose(
            checkpoint,
            purpose=DOWNSTREAM_FINETUNE,
            requested_action_horizon=10,
        )


def test_inference_purpose_rejects_training_resume_flag():
    with pytest.raises(ValueError, match="inference.*resume_training"):
        validate_checkpoint_load_purpose_arguments(
            INFERENCE, resume_training=True
        )


def test_generic_contract_never_falls_through_legacy_loader(tmp_path):
    checkpoint = _make_joint_checkpoint(tmp_path)
    metadata = _metadata(checkpoint)
    metadata.pop(STAGE05_AR_JOINT_CONTRACT_KEY)
    (checkpoint / "resolved_dataset_manifest.json").unlink()
    payload = load_action_expert_config(checkpoint / "action_expert_config.json").payload
    from utils.action_expert_config import architecture_config_hash
    from utils.stage05_checkpoint_contract import validate_generic_action_expert_contract
    from safetensors import safe_open

    with safe_open(str(checkpoint / "action_expert.safetensors"), framework="pt", device="cpu") as source:
        shapes = {name: list(source.get_slice(name).get_shape()) for name in source.keys()}
    runtime = {
        "purpose": "legacy",
        "checkpoint_kind": "joint",
        "action_horizon": payload["action_horizon"],
        "source_action_horizon": payload["action_horizon"],
        "target_action_horizon": payload["action_horizon"],
    }
    generic = {
        "version": 1,
        "architecture_hash": architecture_config_hash(payload),
        "runtime_contract": runtime,
        "runtime_contract_hash": _hash_json(runtime),
        "config_file": "action_expert_config.json",
        "raw_file_sha256": hashlib.sha256((checkpoint / "action_expert_config.json").read_bytes()).hexdigest(),
        "canonical_sha256": _hash_json(payload),
        "state_dict_shapes": shapes,
    }
    generic["content_hash"] = _hash_json(generic)
    metadata["action_expert_contract"] = generic
    _write_metadata(checkpoint, metadata, refresh_contract_hash=False)
    with pytest.raises(ValueError, match="generic Action Expert contract"):
        from utils.stage05_checkpoint_contract import validate_legacy_action_checkpoint
        validate_legacy_action_checkpoint(checkpoint)
    resolved = validate_stage05_checkpoint_for_purpose(
        checkpoint,
        purpose=DOWNSTREAM_FINETUNE,
        requested_action_horizon=10,
    )
    assert resolved.config.action_horizon == 10
    assert validate_generic_action_expert_contract(checkpoint).config.action_horizon == 32

    metadata = _metadata(checkpoint)
    metadata["action_expert_contract"].pop("content_hash")
    _write_metadata(checkpoint, metadata, refresh_contract_hash=False)
    with pytest.raises(ValueError, match="contract is incomplete"):
        validate_generic_action_expert_contract(checkpoint)


def _write_resume_state(checkpoint: Path, *, global_step: int = 3) -> None:
    import torch

    torch.save(
        {"last_epoch": global_step, "_step_count": global_step + 1},
        checkpoint / "scheduler.pt",
    )
    torch.save(
        {"last_global_step": global_step, "global_steps": global_step, "dp_world_size": 1},
        checkpoint / "mp_rank_00_model_states.pt",
    )
    torch.save(
        {"optimizer_state_dict": {"state": {}, "param_groups": []}},
        checkpoint / "zero_pp_rank_0_mp_rank_00_optim_states.pt",
    )


def test_joint_resume_preflight_loads_weights_and_training_state(tmp_path):
    checkpoint = _make_joint_checkpoint(tmp_path)
    _write_resume_state(checkpoint)
    resolved = validate_stage05_resume_artifacts(
        checkpoint,
        requested_action_horizon=32,
    )
    assert resolved.config.action_horizon == 32


def test_joint_resume_preflight_rejects_corrupt_or_inconsistent_artifacts(tmp_path):
    checkpoint = _make_joint_checkpoint(tmp_path / "corrupt")
    _write_resume_state(checkpoint)
    (checkpoint / "action_expert.safetensors").write_bytes(b"joint fixture")
    with pytest.raises(ValueError, match="safetensors cannot be parsed"):
        validate_stage05_resume_artifacts(checkpoint)

    checkpoint = _make_joint_checkpoint(tmp_path / "step")
    _write_resume_state(checkpoint, global_step=3)
    import torch
    torch.save(
        {"last_epoch": 4, "_step_count": 5}, checkpoint / "scheduler.pt"
    )
    with pytest.raises(ValueError, match="scheduler/global step mismatch"):
        validate_stage05_resume_artifacts(checkpoint)


@pytest.mark.parametrize(
    ("purpose", "horizon"),
    [(STAGE05_AR_TO_JOINT, 10),
     (STAGE05_JOINT_RESUME, 10),
     (DOWNSTREAM_FINETUNE, 0)],
)
def test_checkpoint_purpose_rejects_horizon_crossing(tmp_path, purpose, horizon):
    checkpoint = _make_joint_checkpoint(tmp_path)
    with pytest.raises(ValueError):
        validate_stage05_checkpoint_for_purpose(
            checkpoint,
            purpose=purpose,
            external_config_path=AUTHORITY_CONFIG,
            requested_action_horizon=horizon,
        )
