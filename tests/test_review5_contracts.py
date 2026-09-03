import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


def _v2_metadata(window_root=None):
    return SimpleNamespace(
        camera_keys=["front", "wrist"],
        grounding_camera_keys=["front"],
        stats={
            "observation.state": {"q01": np.zeros(3), "q99": np.ones(3)},
            "action": {"q01": np.zeros(2), "q99": np.ones(2)},
        },
        features={
            "observation.state": {"shape": [3]},
            "action": {"shape": [2]},
            "target": {"dtype": "string"},
        },
        root=window_root,
        codebase_version="v2-test",
    )


def _v2_entry():
    return {
        "dataset_path": "/tmp/v2",
        "dataset_type": "vla",
        "target_text_field": "target",
        "use_quantile": True,
        "sample_ratio": 1.0,
    }


def _vqa_spec():
    from utils.dataset_spec import ResolvedDatasetSpec

    return ResolvedDatasetSpec(
        dataset_entry="qa",
        dataset_path="/tmp/qa",
        dataset_type="vlm",
        adapter="vqa_parquet",
        target_text_field="json",
        camera_keys=(),
        grounding_camera_keys=(),
        state_key="",
        action_key="",
        state_dim=0,
        action_dim=0,
        action_horizon=32,
        stats_path=None,
        stats_key=None,
        normalization="none",
        normalization_stats=None,
        sample_ratio=1.0,
        training_eligibility_exists=False,
        training_eligibility_used=False,
        data_version="qa-v1",
    )


def test_v2_observation_contract_records_window_and_rejects_resume_mismatch(tmp_path):
    from utils.dataset_manifest import (
        build_resolved_dataset_manifest,
        validate_resume_manifest,
        write_resolved_dataset_manifest,
    )
    from utils.dataset_spec import resolve_dataset_spec, resolve_objective_requirements

    requirements = resolve_objective_requirements(
        "vlm_and_action",
        adapter="lerobot_v2",
        target_text_field="target",
        dataset_type="vla",
    )
    one = resolve_dataset_spec(
        "legacy",
        _v2_entry(),
        action_horizon=32,
        window_size=1,
        requirements=requirements,
        v2_metadata=_v2_metadata(),
    )
    three = resolve_dataset_spec(
        "legacy",
        _v2_entry(),
        action_horizon=32,
        window_size=3,
        requirements=requirements,
        v2_metadata=_v2_metadata(),
    )
    one_manifest = build_resolved_dataset_manifest([one], "vlm_and_action")
    three_manifest = build_resolved_dataset_manifest([three], "vlm_and_action")
    assert one_manifest["entries"][0]["observation_contract"] == {
        "version": 1,
        "window_size": 1,
        "history_order": "frame_major_oldest_to_current",
        "history_stride": "previous_policy_execution_horizon",
    }
    assert three_manifest["entries"][0]["observation_contract"]["window_size"] == 3
    write_resolved_dataset_manifest(tmp_path, one_manifest)
    with pytest.raises(ValueError, match="observation_contract.*window_size"):
        validate_resume_manifest(three_manifest, tmp_path)


def test_policy_manifest_requires_explicit_override_for_legacy_missing_window(tmp_path):
    from utils.dataset_manifest import (
        build_resolved_dataset_manifest,
        validate_policy_dataset_manifest,
        write_resolved_dataset_manifest,
    )
    from utils.dataset_spec import resolve_dataset_spec

    spec = resolve_dataset_spec(
        "legacy",
        _v2_entry(),
        action_horizon=32,
        window_size=1,
        require_action=True,
        v2_metadata=_v2_metadata(),
    )
    manifest = build_resolved_dataset_manifest([spec], "vlm_and_action")
    del manifest["entries"][0]["observation_contract"]
    write_resolved_dataset_manifest(tmp_path, manifest)
    with pytest.raises(ValueError, match="legacy.*observation contract.*override"):
        validate_policy_dataset_manifest(spec, tmp_path)
    with pytest.warns(RuntimeWarning, match="observation contract"):
        validate_policy_dataset_manifest(
            spec,
            tmp_path,
            allow_legacy_missing_observation_contract=True,
        )


def test_action_only_rejects_vqa_before_processor_or_sampling(monkeypatch):
    import utils.load_training_dataset as loader

    entry = {
        "dataset_path": "/does/not/matter",
        "dataset_type": "vlm",
        "sample_ratio": 1.0,
    }
    monkeypatch.setattr(loader, "DATASET2FEATURE", {"qa": entry})
    processor_calls = []
    monkeypatch.setattr(
        loader.AutoProcessor,
        "from_pretrained",
        lambda *args, **kwargs: processor_calls.append((args, kwargs)),
    )
    with pytest.raises(
        ValueError, match=r"dataset entry 'qa'.*dataset_type='vlm'.*loss_type='action'"
    ):
        loader.build_concat_streaming_dataset(
            ["qa"],
            "model",
            "fast",
            window_size=1,
            action_horizon=32,
            accelerator=None,
            loss_type="action",
        )
    assert processor_calls == []


def test_joint_vqa_and_vla_manifest_requirements_are_entry_scoped():
    from utils.dataset_manifest import build_resolved_dataset_manifest
    from utils.dataset_spec import resolve_dataset_spec, resolve_objective_requirements

    vla_requirements = resolve_objective_requirements(
        "vlm_and_action",
        adapter="lerobot_v2",
        target_text_field="target",
        dataset_type="vla",
    )
    vla = resolve_dataset_spec(
        "robot",
        _v2_entry(),
        action_horizon=32,
        window_size=1,
        requirements=vla_requirements,
        v2_metadata=_v2_metadata(),
    )
    manifest = build_resolved_dataset_manifest([vla, _vqa_spec()], "vlm_and_action")
    by_name = {entry["dataset_entry"]: entry for entry in manifest["entries"]}
    assert by_name["qa"]["loss_requirements"] == ["images", "task", "target_text"]
    assert by_name["robot"]["loss_requirements"] == [
        "images",
        "task",
        "target_text",
        "state",
        "actions",
        "stats",
    ]


@pytest.mark.parametrize("use_difference_query", [False, True])
def test_v3_subtask_then_action_fails_before_processor_or_model(
    monkeypatch, use_difference_query
):
    import policies.reasoning_vla_policy as policy_module

    entry = {
        "dataset_path": "/tmp/v3",
        "dataset_type": "vla",
        "dataset_adapter": "lerobot_v3_future_difference",
    }
    monkeypatch.setitem(policy_module.DATASET2FEATURE, "future", entry)
    monkeypatch.setattr(
        policy_module.AutoProcessor,
        "from_pretrained",
        lambda *_: pytest.fail("processor must not load for unsupported inference mode"),
    )
    monkeypatch.setattr(
        policy_module.ZR0Model,
        "from_pretrained",
        lambda *_args, **_kwargs: pytest.fail("model must not load for unsupported inference mode"),
    )
    with pytest.raises(
        ValueError, match="future.*lerobot_v3_future_difference.*subtask_then_action.*direct_action"
    ):
        policy_module.ZR0Policy(
            "future",
            "/tmp/checkpoint",
            "subtask_then_action",
            window_size=1,
            device="cpu",
            use_difference_query=use_difference_query,
        )


def test_v3_policy_rejects_non_single_frame_window_before_loading(monkeypatch):
    import policies.reasoning_vla_policy as policy_module

    monkeypatch.setitem(
        policy_module.DATASET2FEATURE,
        "future",
        {
            "dataset_path": "/tmp/v3",
            "dataset_type": "vla",
            "dataset_adapter": "lerobot_v3_future_difference",
        },
    )
    monkeypatch.setattr(
        policy_module.AutoProcessor,
        "from_pretrained",
        lambda *_: pytest.fail("processor must not load for invalid window_size"),
    )
    with pytest.raises(ValueError, match="future.*window_size=1"):
        policy_module.ZR0Policy(
            "future",
            "/tmp/checkpoint",
            "direct_action",
            window_size=2,
            device="cpu",
        )


def test_v3_observation_contract_is_fixed_to_one(tmp_path):
    from utils.dataset_spec import resolve_dataset_spec

    meta = tmp_path / "meta"
    meta.mkdir()
    (meta / "info.json").write_text(
        json.dumps(
            {
                "features": {
                    "a": {"dtype": "image"},
                    "b": {"dtype": "image"},
                    "c": {"dtype": "image"},
                    "target": {"dtype": "string"},
                }
            }
        ),
        encoding="utf-8",
    )
    entry = {
        "dataset_path": str(tmp_path),
        "dataset_type": "vla",
        "dataset_adapter": "lerobot_v3_future_difference",
        "camera_keys": ["a", "b", "c"],
        "target_text_field": "target",
        "sample_ratio": 1.0,
    }
    spec = resolve_dataset_spec(
        "future", entry, action_horizon=32, window_size=1, require_action=False
    )
    assert spec.observation_contract.window_size == 1
    with pytest.raises(ValueError, match="future.*window_size=1"):
        resolve_dataset_spec(
            "future", entry, action_horizon=32, window_size=2, require_action=False
        )


def test_v3_builder_rejects_non_single_frame_window_before_processor(monkeypatch):
    import utils.load_training_dataset as loader

    monkeypatch.setattr(
        loader,
        "DATASET2FEATURE",
        {
            "future": {
                "dataset_path": "/tmp/v3",
                "dataset_type": "vla",
                "dataset_adapter": "lerobot_v3_future_difference",
                "target_text_field": "target",
                "sample_ratio": 1.0,
            }
        },
    )
    monkeypatch.setattr(
        loader.AutoProcessor,
        "from_pretrained",
        lambda *_args, **_kwargs: pytest.fail(
            "processor must not load for an invalid observation contract"
        ),
    )
    with pytest.raises(ValueError, match="future.*window_size=1"):
        loader.build_concat_streaming_dataset(
            ["future"],
            "model",
            "fast",
            window_size=2,
            action_horizon=32,
            accelerator=None,
            loss_type="vlm",
        )


def test_v2_policy_enforces_checkpoint_observation_window(monkeypatch, tmp_path):
    import policies.reasoning_vla_policy as policy_module
    from utils.dataset_manifest import (
        build_resolved_dataset_manifest,
        write_resolved_dataset_manifest,
    )
    from utils.dataset_spec import resolve_dataset_spec

    metadata = _v2_metadata()
    spec = resolve_dataset_spec(
        "legacy",
        _v2_entry(),
        action_horizon=32,
        window_size=3,
        require_action=True,
        v2_metadata=metadata,
    )
    write_resolved_dataset_manifest(
        tmp_path, build_resolved_dataset_manifest([spec], "vlm_and_action")
    )

    class FakeModel:
        action_expert_config = SimpleNamespace(
            action_horizon=32,
            action_dim=64,
            state_dim=64,
        )

        def to(self, *args, **kwargs):
            return self

        def eval(self):
            return self

    monkeypatch.setitem(policy_module.DATASET2FEATURE, "legacy", _v2_entry())
    monkeypatch.setattr(
        policy_module.AutoProcessor, "from_pretrained", lambda *_: object()
    )
    monkeypatch.setattr(
        policy_module.ZR0Model, "from_pretrained", lambda *_, **__: FakeModel()
    )
    monkeypatch.setattr(
        policy_module, "LeRobotDatasetMetadata", lambda **_: metadata
    )
    monkeypatch.setattr(torch, "compile", lambda model, **_: model)

    policy = policy_module.ZR0Policy(
        "legacy", str(tmp_path), "direct_action", window_size=3, device="cpu"
    )
    assert policy.window_size == 3
    assert policy.dataset_spec.observation_contract.window_size == 3

    with pytest.raises(ValueError, match="observation_contract.window_size"):
        policy_module.ZR0Policy(
            "legacy", str(tmp_path), "direct_action", window_size=1, device="cpu"
        )
