from collections import OrderedDict
from types import MethodType, SimpleNamespace

import numpy as np
import pytest
import torch

from utils.stage05_dataset import Stage05MixedPretrainingDataset
from utils.stage05_canonical import canonical_droid_arrays
from utils.stage3_horizon_runtime import extend_fm_supervision


def dataset_fixture(*, valid_at=30, invalid_state=False, length=60):
    states = np.zeros((length, 6))
    targets = np.zeros((length, 6))
    targets[:valid_at, 0] = 1000
    if invalid_state:
        states[0, 0] = np.nan
    rows = [{"observation.state.cartesian_position": states[i],
             "observation.state.gripper_position": 0.5,
             "action.cartesian_position": targets[i], "action.gripper_position": 0.5}
            for i in range(length)]
    arrays = canonical_droid_arrays(states, np.full(length, 0.5), targets, np.full(length, 0.5))
    flags = np.zeros((length, 5), dtype=bool)
    flags[:, 3] = arrays[3]
    ds = SimpleNamespace(action_horizon=50, max_pad_length=64, kind="droid",
        _frozen_validity=np.packbits(flags, axis=0), _canonical_cache=OrderedDict(),
        _episode_for_global=lambda i: (dict(episode_index=0, dataset_to_index=length), i),
        _episode_rows=lambda e: rows,
        stats={name: dict(q01=np.full(7, -1, dtype=np.float32), q99=np.ones(7, dtype=np.float32))
               for name in ("observation.state", "action")})
    for name in ("_canonical_chunk", "_action_inputs"):
        setattr(ds, name, MethodType(getattr(Stage05MixedPretrainingDataset, name), ds))
    return ds


def empty():
    return dict(fm_eligible=torch.tensor(False), action_supervision_available=torch.tensor(False))


def test_supervision_after_frame_ten_is_not_lost():
    ds = dataset_fixture()
    result = extend_fm_supervision(ds, 0, empty())
    assert bool(result["fm_eligible"])
    assert result["action"].shape == (50, 64)
    assert not result["action_mask"][:30].any()
    assert int(result["action_mask"].sum()) == 20 * 7
    assert not result["action"][:, 7:].any()


def test_padding_never_crosses_episode():
    ds = dataset_fixture(valid_at=0, length=60)
    result = extend_fm_supervision(ds, 55, empty())
    assert int(result["action_mask"].sum()) == 5 * 7
    assert not result["action"][5:].any()


@pytest.mark.parametrize("kwargs", [dict(valid_at=60), dict(invalid_state=True)])
def test_absent_actions_and_invalid_states_stay_masked(kwargs):
    result = extend_fm_supervision(dataset_fixture(**kwargs), 0, empty())
    assert not bool(result["fm_eligible"])


def test_previously_eligible_sample_is_unchanged():
    result = dict(fm_eligible=True, sentinel=object())
    assert extend_fm_supervision(None, 0, result) is result


def test_unexpected_canonical_failure_is_not_hidden():
    ds = dataset_fixture()
    def fail(*args):
        raise ValueError("corrupt numeric input")
    ds._canonical_chunk = fail
    with pytest.raises(ValueError, match="corrupt numeric"):
        extend_fm_supervision(ds, 0, empty())


def test_horizon_loader_checks_other_fields_and_restores(tmp_path, monkeypatch):
    from utils import preparation_audit_cache as cache
    from utils.stage3_horizon_runtime import install_horizon_adaptation
    calls = []
    def original(path, *, audit_cache, expected_generation):
        calls.append(expected_generation)
        if expected_generation.get("kind") != "droid":
            raise ValueError("wrong kind")
        return {"generation": {"horizon": 10}}
    monkeypatch.setattr(cache, "load_cached_sidecar", original)
    restore = install_horizon_adaptation({"droid": dict(joint_sidecar_path=str(tmp_path))})
    try:
        result = cache.load_cached_sidecar(tmp_path, audit_cache=None,
            expected_generation=dict(horizon=50, kind="droid"))
        assert result["generation"]["horizon"] == 10
        assert calls[-1] == dict(horizon=10, kind="droid")
        with pytest.raises(ValueError, match="wrong kind"):
            cache.load_cached_sidecar(tmp_path, audit_cache=None,
                expected_generation=dict(horizon=50, kind="molmo"))
    finally:
        restore()
    assert cache.load_cached_sidecar is original
