import json
from pathlib import Path

import numpy as np
import pytest
import torch


REAL_V3_ROOT = Path(
    "/opt/data/private/lq/datasets/molmoact_dataset_tabletop-v3_stage05"
)


def _v3_entry(root=REAL_V3_ROOT):
    return {
        "dataset_path": str(root),
        "dataset_type": "vla",
        "dataset_adapter": "lerobot_v3_future_difference",
        "target_text_field": "train_data",
        "camera_keys": ["first_view", "second_view", "wrist_image"],
        "state_field": "state",
        "action_field": "actions",
        "stats_path": "meta/stats_gr00t.json",
        "use_quantile": True,
        "sample_ratio": 1.0,
    }


def test_real_v3_spec_matches_training_schema_and_shared_stats():
    if not REAL_V3_ROOT.is_dir():
        pytest.skip("real stage05 dataset is unavailable")
    from utils.dataset_spec import resolve_dataset_spec
    from utils.dataset_adapters import validate_v3_quantile_stats

    spec = resolve_dataset_spec(
        "molmoact_tabletop_v3_stage05",
        _v3_entry(),
        action_horizon=32,
        require_action=True,
    )
    raw_stats = json.loads((REAL_V3_ROOT / "meta/stats_gr00t.json").read_text())
    expected = validate_v3_quantile_stats(raw_stats)

    assert spec.adapter == "lerobot_v3_future_difference"
    assert spec.camera_keys == ("first_view", "second_view", "wrist_image")
    assert spec.state_key == "state"
    assert spec.action_key == "actions"
    assert spec.state_dim == spec.action_dim == 7
    for key in ("observation.state", "action"):
        np.testing.assert_array_equal(spec.normalization_stats[key]["q01"], expected[key]["q01"])
        np.testing.assert_array_equal(spec.normalization_stats[key]["q99"], expected[key]["q99"])


def test_v3_release_metadata_overrides_false_yaml_eligibility_claim():
    if not REAL_V3_ROOT.is_dir():
        pytest.skip("real stage05 dataset is unavailable")
    from utils.dataset_spec import resolve_dataset_spec

    entry = _v3_entry()
    entry["training_eligibility_exists"] = True
    spec = resolve_dataset_spec(
        "molmoact_tabletop_v3_stage05",
        entry,
        action_horizon=32,
        require_action=False,
    )

    assert spec.training_eligibility_exists is False
    assert spec.training_eligibility_used is False
    assert spec.training_eligibility_source == "unavailable_in_release"


def test_v3_action_denormalization_uses_shared_quantiles():
    if not REAL_V3_ROOT.is_dir():
        pytest.skip("real stage05 dataset is unavailable")
    from utils.dataset_spec import denormalize_actions, resolve_dataset_spec
    from utils.normalization import min_max_denorm

    spec = resolve_dataset_spec(
        "molmoact_tabletop_v3_stage05",
        _v3_entry(),
        action_horizon=32,
        require_action=True,
    )
    normalized = torch.tensor([[-1.0, -0.5, 0.0, 0.25, 0.5, 0.75, 1.0, 99.0]])
    actual = denormalize_actions(normalized, spec)
    expected = min_max_denorm(
        normalized[:, :7], spec.normalization_stats["action"], True
    )
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda entry: entry.update(dataset_adapter="unknown"), "unknown"),
        (lambda entry: entry.update(camera_keys=["first_view"]), "camera_keys"),
        (lambda entry: entry.update(action_field="missing"), "missing"),
    ],
)
def test_v3_spec_fails_with_entry_and_field(mutation, match):
    if not REAL_V3_ROOT.is_dir():
        pytest.skip("real stage05 dataset is unavailable")
    from utils.dataset_spec import resolve_dataset_spec

    entry = _v3_entry()
    mutation(entry)
    with pytest.raises(ValueError, match=rf"molmoact_tabletop_v3_stage05.*{match}"):
        resolve_dataset_spec(
            "molmoact_tabletop_v3_stage05",
            entry,
            action_horizon=32,
            require_action=True,
        )


def test_v2_spec_preserves_metadata_fields_and_stats():
    from utils.dataset_spec import resolve_dataset_spec

    stats = {
        "observation.state": {"min": np.zeros(3), "max": np.ones(3)},
        "action": {"min": np.zeros(2), "max": np.ones(2)},
    }
    metadata = type(
        "Metadata",
        (),
        {
            "camera_keys": ["cam_a", "cam_b"],
            "grounding_camera_keys": ["cam_a"],
            "stats": stats,
            "features": {
                "observation.state": {"shape": (3,)},
                "action": {"shape": (2,)},
            },
        },
    )()
    spec = resolve_dataset_spec(
        "legacy",
        {
            "dataset_path": "/tmp/legacy",
            "dataset_type": "vla",
            "use_quantile": False,
            "sample_ratio": 1.0,
        },
        action_horizon=16,
        require_action=True,
        v2_metadata=metadata,
    )
    assert spec.adapter == "lerobot_v2"
    assert spec.camera_keys == ("cam_a", "cam_b")
    assert spec.grounding_camera_keys == ("cam_a",)
    assert spec.state_key == "observation.state"
    assert spec.action_key == "action"
    assert spec.state_dim == 3 and spec.action_dim == 2
    assert spec.normalization_stats is stats


def test_real_v3_policy_constructs_with_matching_checkpoint_manifest(
    monkeypatch, tmp_path
):
    if not REAL_V3_ROOT.is_dir():
        pytest.skip("real stage05 dataset is unavailable")
    pytest.importorskip("transformers")
    import policies.reasoning_vla_policy as policy_module
    from utils.dataset_manifest import (
        build_resolved_dataset_manifest,
        write_resolved_dataset_manifest,
    )
    from utils.dataset_spec import resolve_dataset_spec

    class FakeModel:
        action_expert_config = type(
            "Config", (), {"action_horizon": 32, "action_dim": 64, "state_dim": 64}
        )()

        def to(self, *args, **kwargs):
            return self

        def eval(self):
            return self

    monkeypatch.setitem(
        policy_module.DATASET2FEATURE,
        "molmoact_tabletop_v3_stage05",
        _v3_entry(),
    )
    monkeypatch.setattr(policy_module.AutoProcessor, "from_pretrained", lambda *_: object())
    monkeypatch.setattr(policy_module.ZR0Model, "from_pretrained", lambda *_, **__: FakeModel())
    monkeypatch.setattr(torch, "compile", lambda model, **_: model)
    monkeypatch.setattr(
        policy_module,
        "LeRobotDatasetMetadata",
        lambda *_, **__: pytest.fail("v3 policy must not construct v2 metadata"),
    )
    spec = resolve_dataset_spec(
        "molmoact_tabletop_v3_stage05",
        _v3_entry(),
        action_horizon=32,
        require_action=True,
    )
    write_resolved_dataset_manifest(
        tmp_path,
        build_resolved_dataset_manifest([spec], "vlm_and_action"),
    )

    policy = policy_module.ZR0Policy(
        "molmoact_tabletop_v3_stage05",
        str(tmp_path),
        "direct_action",
        window_size=1,
        device="cpu",
    )
    assert policy.camera_keys == ["first_view", "second_view", "wrist_image"]
    assert policy.state_dim == policy.action_dim == 7
    assert policy.dataset_spec.action_key == "actions"
    for key in ("observation.state", "action"):
        np.testing.assert_array_equal(
            policy.normalization_stats[key]["q01"],
            spec.normalization_stats[key]["q01"],
        )
        np.testing.assert_array_equal(
            policy.normalization_stats[key]["q99"],
            spec.normalization_stats[key]["q99"],
        )


def test_v2_policy_keeps_legacy_metadata_contract(monkeypatch):
    pytest.importorskip("transformers")
    import policies.reasoning_vla_policy as policy_module

    stats = {
        "observation.state": {"q01": np.zeros(3), "q99": np.ones(3)},
        "action": {"q01": np.zeros(2), "q99": np.ones(2)},
    }
    metadata = type(
        "Metadata",
        (),
        {
            "camera_keys": ["front"],
            "grounding_camera_keys": [],
            "stats": stats,
            "features": {
                "observation.state": {"shape": [3]},
                "action": {"shape": [2]},
            },
        },
    )()

    class FakeModel:
        action_expert_config = type(
            "Config", (), {"action_horizon": 10, "action_dim": 64, "state_dim": 64}
        )()

        def to(self, *args, **kwargs):
            return self

        def eval(self):
            return self

    monkeypatch.setitem(
        policy_module.DATASET2FEATURE,
        "legacy",
        {
            "dataset_path": "/tmp/legacy",
            "dataset_type": "vla",
            "use_quantile": True,
            "sample_ratio": 1.0,
        },
    )
    monkeypatch.setattr(policy_module.AutoProcessor, "from_pretrained", lambda *_: object())
    monkeypatch.setattr(policy_module.ZR0Model, "from_pretrained", lambda *_, **__: FakeModel())
    monkeypatch.setattr(policy_module, "LeRobotDatasetMetadata", lambda **_: metadata)
    monkeypatch.setattr(torch, "compile", lambda model, **_: model)

    policy = policy_module.ZR0Policy(
        "legacy",
        "/tmp/checkpoint",
        "direct_action",
        window_size=1,
        device="cpu",
        allow_legacy_checkpoint_without_manifest=True,
    )
    assert policy.dataset_meta is metadata
    assert policy.camera_keys == ["front"]
    assert policy.state_dim == 3 and policy.action_dim == 2
    assert policy.normalization_stats is stats
