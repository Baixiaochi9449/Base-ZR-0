import json
from pathlib import Path

import pytest

from utils.action_expert_config import load_action_expert_config


ROOT = Path(__file__).resolve().parents[1]
AUTHORITY = ROOT / "configs/stage05_four_dataset_action_expert.json"


def _payload():
    return json.loads(AUTHORITY.read_text(encoding="utf-8"))


def _write(tmp_path, payload, name="config.json"):
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


@pytest.mark.parametrize("case", ["missing", "empty", "damaged"])
def test_action_expert_config_missing_empty_or_damaged_fails_fast(tmp_path, case):
    path = tmp_path / "config.json"
    if case == "empty":
        path.write_text("{}", encoding="utf-8")
    elif case == "damaged":
        path.write_text("{broken", encoding="utf-8")
    with pytest.raises(ValueError, match="Action Expert config"):
        load_action_expert_config(path)


def test_action_expert_config_missing_structural_field_fails_fast(tmp_path):
    payload = _payload()
    del payload["diffusion_transformer_cfg"]["num_layers"]
    with pytest.raises(ValueError, match="num_layers"):
        load_action_expert_config(_write(tmp_path, payload))


@pytest.mark.parametrize(
    ("field", "expected", "message"),
    [
        ("action_dim", 63, "action_dim mismatch"),
        ("state_dim", 63, "state_dim mismatch"),
        ("action_horizon", 31, "action_horizon mismatch"),
        ("vlm_output_embedding_dim", 1024, "vlm_output_embedding_dim mismatch"),
    ],
)
def test_action_expert_config_contract_mismatch_fails_fast(tmp_path, field, expected, message):
    with pytest.raises(ValueError, match=message):
        load_action_expert_config(
            _write(tmp_path, _payload()),
            expected_action_dim=(expected if field == "action_dim" else 64),
            expected_state_dim=(expected if field == "state_dim" else 64),
            expected_action_horizon=(expected if field == "action_horizon" else 32),
            expected_vlm_hidden_size=(
                expected if field == "vlm_output_embedding_dim" else 2048
            ),
        )


def test_authoritative_action_expert_config_is_complete_and_not_class_defaults():
    resolved = load_action_expert_config(
        AUTHORITY,
        expected_action_dim=64,
        expected_state_dim=64,
        expected_action_horizon=32,
        expected_vlm_hidden_size=2048,
    )
    assert resolved.config.action_horizon == 32  # class default is 20
    assert resolved.config.mlp_hidden_size == 512
    assert resolved.config.diffusion_transformer_cfg["num_layers"] == 9
    assert resolved.config.diffusion_transformer_cfg["num_attention_heads"] == 32
    assert resolved.config.max_seq_len == 256
    assert resolved.source_sha256 and resolved.parsed_sha256
    assert resolved.source_action_horizon == 32
    assert resolved.action_horizon_overridden is False


def test_action_horizon_override_preserves_source_identity_and_resolves_runtime(tmp_path):
    path = _write(tmp_path, _payload())
    resolved = load_action_expert_config(
        path,
        expected_action_dim=64,
        expected_state_dim=64,
        action_horizon_override=10,
        expected_vlm_hidden_size=2048,
    )

    assert resolved.source_action_horizon == 32
    assert resolved.config.action_horizon == 10
    assert resolved.payload["action_horizon"] == 10
    assert resolved.action_horizon_overridden is True
    assert resolved.source_sha256 != resolved.parsed_sha256


@pytest.mark.parametrize("override", [0, -1, True, 256])
def test_action_horizon_override_must_be_positive_and_fit_capacity(tmp_path, override):
    with pytest.raises(ValueError, match="action_horizon override"):
        load_action_expert_config(
            _write(tmp_path, _payload()), action_horizon_override=override
        )


def test_action_horizon_override_and_strict_expectation_are_mutually_exclusive(tmp_path):
    with pytest.raises(ValueError, match="mutually exclusive"):
        load_action_expert_config(
            _write(tmp_path, _payload()),
            expected_action_horizon=32,
            action_horizon_override=10,
        )


def test_action_expert_constructor_values_come_from_json(tmp_path):
    payload = _payload()
    payload["mlp_hidden_size"] = 640
    payload["diffusion_transformer_cfg"]["num_layers"] = 7
    payload["diffusion_transformer_cfg"]["dropout"] = 0.125
    resolved = load_action_expert_config(_write(tmp_path, payload))
    assert resolved.config.mlp_hidden_size == 640
    assert resolved.config.diffusion_transformer_cfg["num_layers"] == 7
    assert resolved.config.diffusion_transformer_cfg["dropout"] == 0.125
