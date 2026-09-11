"""Explicit H50 runtime adaptation over immutable, audited H10 sidecars."""

from dataclasses import replace
from functools import wraps
from pathlib import Path

import numpy as np


def extend_fm_supervision(dataset, global_index, result):
    if bool(result["fm_eligible"]):
        return result
    episode_meta, base = dataset._episode_for_global(global_index)
    stop = min(global_index + dataset.action_horizon, int(episode_meta["dataset_to_index"]))
    indices = np.arange(global_index, stop, dtype=np.int64)
    valid = (dataset._frozen_validity[indices // 8, 3] >> (7 - indices % 8)) & 1
    if not valid.any():
        return result
    episode = int(episode_meta["episode_index"])
    rows = dataset._episode_rows(episode)
    try:
        chunk = dataset._canonical_chunk(episode, rows, base)
    except ValueError:
        # The original routine caches canonical validity before rejecting a state.
        cached = dataset._canonical_cache.get(episode)
        if cached is None or bool(cached[2][base]):
            raise
        return result
    if chunk.temporal_mask.any():
        result.update(dataset._action_inputs(episode, rows, base))
        result["fm_eligible"] = result["action_supervision_available"]
    return result


def install_horizon_adaptation(routes):
    """Only the explicit H50 entrypoint installs these scoped adapters."""
    from utils import preparation_audit_cache as cache
    from utils.stage05_dataset import Stage05MixedPretrainingDataset as Dataset

    allowed = {str(Path(entry["joint_sidecar_path"]).resolve()) for entry in routes.values()}
    original_loader, original_init, original_get = cache.load_cached_sidecar, Dataset.__init__, Dataset._get_item_once

    @wraps(original_loader)
    def load(path, *, audit_cache, expected_generation=None):
        expected = dict(expected_generation or {})
        if str(Path(path).resolve()) not in allowed or expected.get("horizon") != 50:
            return original_loader(path, audit_cache=audit_cache, expected_generation=expected_generation)
        expected["horizon"] = 10
        return original_loader(path, audit_cache=audit_cache, expected_generation=expected)

    @wraps(original_init)
    def initialize(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        if self.action_horizon != 50:
            return
        if (str(self.sidecar_root) not in allowed or not self.frozen_index
                or self.loss_type != "vlm_and_action" or not self.entry.get("preparation_audit_cache")):
            raise ValueError("H50 adaptation requires the approved frozen Joint data routes")
        auxiliary = dict(self.spec.auxiliary_contract or {})
        auxiliary["action_horizon_adaptation"] = dict(version=1, source_horizon=10, runtime_horizon=50,
            fm_eligibility="canonical_state_and_any_valid_action_in_runtime_horizon",
            statistics="unchanged_independent_frozen_training_statistics")
        self.spec = replace(self.spec, auxiliary_contract=auxiliary)

    @wraps(original_get)
    def get(self, global_index):
        result = original_get(self, global_index)
        if self.action_horizon == 50:
            result = extend_fm_supervision(self, global_index, result)
        return result

    cache.load_cached_sidecar, Dataset.__init__, Dataset._get_item_once = load, initialize, get

    def restore():
        cache.load_cached_sidecar, Dataset.__init__, Dataset._get_item_once = original_loader, original_init, original_get

    return restore
