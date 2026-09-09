"""Strict Stage06 manifest join with process-local bounded HDF5 handles."""

from collections import OrderedDict
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from utils.training_tokenization import DatasetIntegrityError


class OpticalFlowReader:
    def __init__(self, root, manifest, *, delta_frames=10, camera_key=None, max_handles=4, contract=None):
        import h5py

        self.root = Path(root).resolve()
        self.manifest = Path(manifest)
        if not self.manifest.is_absolute():
            self.manifest = self.root / self.manifest
        self.delta_frames = delta_frames
        self.contract = contract
        self.expected_fps = contract["fps"] if contract else 10.0
        self.camera_key = camera_key
        if max_handles < 1:
            raise ValueError("max_handles must be positive")
        self.max_handles = max_handles
        self._handles = OrderedDict()
        self._pid = os.getpid()
        self.rows = {}
        self.episodes = {}
        payload = self.manifest.read_bytes()
        self.manifest_sha256 = hashlib.sha256(payload).hexdigest()
        try:
            entries = [json.loads(line) for line in payload.splitlines() if line.strip()]
            if not entries:
                raise ValueError("empty flow manifest")
            if camera_key is None:
                camera_key = entries[0].get("camera_key")
                if not isinstance(camera_key, str) or not camera_key:
                    raise ValueError("manifest has no camera key")
            self.camera_key = camera_key
            for entry in entries:
                source_episode = entry["merged_episode_index"]
                remapping = contract.get("flow_to_dataset_episode") if contract else None
                if remapping is not None and str(source_episode) not in remapping:
                    raise ValueError("Flow manifest record is absent from verified remapping")
                episode = remapping[str(source_episode)] if remapping is not None else source_episode
                if not isinstance(episode, int) or episode < 0 or episode in self.episodes:
                    raise ValueError(f"duplicate/invalid episode {episode}")
                if entry["camera_key"] != camera_key:
                    raise ValueError("flow camera mapping mismatch")
                if contract is not None:
                    if (entry.get("schema_version") != "stage06_flow_manifest_v2"
                            or entry.get("dataset_id") != contract["dataset_id"]
                            or str(episode) not in contract["mapping"]
                            or entry["source_episode_index"] != contract["mapping"][str(episode)]["old_episode_index"]):
                        raise ValueError("flow dataset_id/schema/episode mapping mismatch")
                path = (self.root / entry["hdf5_path"]).resolve()
                if not path.is_relative_to(self.root):
                    raise ValueError("HDF5 path escapes root")
                self.episodes[episode] = (path, entry)
        except Exception as error:
            raise DatasetIntegrityError(f"flow manifest {self.manifest}: {error}") from error

    def _validate_structure(self, handle, entry):
        n = entry["frame_count"]
        for key, shape, kinds in (
            ("flow", (n, 2, 224, 224), "f"), ("valid_mask", (n, 1, 224, 224), "bu"),
            ("frame_index", (n,), "iu"), ("target_frame_index", (n,), "iu"),
            ("actual_delta_frames", (n,), "iu"), ("label_source", (n,), "iu"),
        ):
            value = handle[key]
            if value.shape != shape or value.dtype.kind not in kinds:
                raise ValueError(f"invalid flow field {key}: {value.shape}, {value.dtype}")
        expected = {"camera_key": self.camera_key, "merged_episode_index": entry["merged_episode_index"],
                    "source_episode_index": entry["source_episode_index"],
                    "flow_units": "normalized_source_image_extent", "flow_direction": "forward_only",
                    "tail_policy": "clamp",
                    "validity_semantics": "finite_and_forward_destination_in_bounds_not_occlusion"}
        for key, value in expected.items():
            if handle.attrs.get(key) != value:
                raise ValueError(f"flow metadata mismatch: {key}")
        if "dataset_id" in entry and handle.attrs.get("dataset_id") != entry["dataset_id"]:
            raise ValueError("flow metadata mismatch: dataset_id")
        if "timestamp" in handle:
            raise ValueError("unsupported Flow timestamp alias; require source_timestamp_s/target_timestamp_s/actual_delta_s")
        nominal = int(handle.attrs["nominal_delta_frames"])
        if nominal != self.delta_frames:
            raise ValueError(f"flow metadata mismatch: nominal_delta_frames={nominal}")
        fps = float(handle.attrs["fps"])
        if not np.isfinite(fps) or fps != self.expected_fps:
            raise ValueError("invalid flow fps")
        frames = handle["frame_index"][:]
        targets = handle["target_frame_index"][:]
        delta = handle["actual_delta_frames"][:]
        source = handle["label_source"][:]
        if n < 1 or len(np.unique(frames)) != n or (frames < 0).any():
            raise ValueError("duplicate/invalid frame_index")
        if not np.isin(targets, frames).all() or not np.array_equal(targets - frames, delta):
            raise ValueError("target frame crosses episode or delta mismatch")
        if (delta < 0).any() or (delta > self.delta_frames).any() or not np.isin(source, [1, 2]).all():
            raise ValueError("invalid delta or label_source")
        if ((source == 2) != (delta == 0)).any():
            raise ValueError("identity label_source must have delta zero")
        if self.contract is not None:
            from utils.aux_data_contract import validate_flow_file
            validate_flow_file(handle, entry, self.contract, frames, targets)
        return frames

    def close(self):
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()
        self.rows.clear()

    def eligible_frames(self, min_valid_fraction):
        """Optional Slot+Flow sampling uses the same fixed-horizon pooled mask rule."""
        eligible = set()
        for episode in self.episodes:
            handle = self._handle(episode)
            for start in range(0, len(handle["frame_index"]), 32):
                stop = min(start + 32, len(handle["frame_index"]))
                masks = handle["valid_mask"][start:stop]
                if not np.isin(masks, [0, 1]).all():
                    raise ValueError(f"flow episode={episode}: invalid mask during eligibility scan")
                pooled = masks.reshape(-1, 1, 56, 4, 56, 4).mean(axis=(3, 5))
                valid = (pooled >= min_valid_fraction).reshape(stop - start, -1).any(-1)
                valid &= handle["actual_delta_frames"][start:stop] == self.delta_frames
                valid &= handle["label_source"][start:stop] == 1
                if self.contract and self.contract.get("excluded_frames"):
                    valid &= ~np.isin(handle["frame_index"][start:stop], self.contract["excluded_frames"].get(str(episode), []))
                eligible.update((episode, int(frame)) for frame in handle["frame_index"][start:stop][valid])
        return eligible

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_handles"] = OrderedDict()
        state["rows"] = {}
        state["_pid"] = None
        return state

    def _handle(self, episode):
        import h5py

        audit_cache = self.contract.get("audit_cache") if self.contract else None
        if audit_cache is not None:
            path, entry = self.episodes[episode]
            audit_cache.check(path, entry["sha256"])
        if self._pid != os.getpid():
            self.close()
            self._pid = os.getpid()
        handle = self._handles.pop(episode, None)
        if handle is None:
            path, entry = self.episodes[episode]
            handle = h5py.File(path, "r")
            try:
                frames = handle["frame_index"][:] if audit_cache is not None else self._validate_structure(handle, entry)
                mapping = {(episode, int(frame)): row for row, frame in enumerate(frames)}
            except Exception:
                handle.close()
                raise
            self.rows.update(mapping)
        self._handles[episode] = handle
        while len(self._handles) > self.max_handles:
            old_episode, old_handle = self._handles.popitem(last=False)
            old_handle.close()
            self.rows = {key: row for key, row in self.rows.items() if key[0] != old_episode}
        return handle

    def read(self, episode, frame):
        result = {"flow_supervision_available": torch.tensor(False),
                  "flow_episode_id": torch.tensor(episode), "flow_frame_index": torch.tensor(frame),
                  "flow_actual_delta_frames": torch.tensor(-1), "flow_label_source": torch.tensor(-1),
                  "flow_target": None, "flow_valid_mask": None}
        result["flow_nominal_delta_frames"] = torch.tensor(self.delta_frames)
        result["flow_fps"] = torch.tensor(self.expected_fps)
        result["flow_exclusion_reason"] = torch.tensor(1)
        if episode not in self.episodes:
            return result
        try:
            handle = self._handle(episode)
            row = self.rows.get((episode, frame))
            if row is None:
                return result
            if int(handle["frame_index"][row]) != frame:
                raise ValueError("flow frame_index changed after manifest validation")
            delta, source = int(handle["actual_delta_frames"][row]), int(handle["label_source"][row])
            result.update(flow_actual_delta_frames=torch.tensor(delta), flow_label_source=torch.tensor(source))
            result["flow_exclusion_reason"] = torch.tensor(2)
            excluded = self.contract.get("excluded_frames", {}).get(str(episode), []) if self.contract else []
            if int(frame) in excluded:
                result["flow_exclusion_reason"] = torch.tensor(5)
                return result
            if delta != self.delta_frames or source != 1:
                return result
            flow, mask = handle["flow"][row].astype(np.float32), handle["valid_mask"][row]
            if not np.isin(mask, [0, 1]).all() or not np.isfinite(flow).all():
                raise ValueError("nonfinite flow or nonbinary mask")
            result["flow_exclusion_reason"] = torch.tensor(0 if mask.any() else 3)
            result.update(flow_supervision_available=torch.tensor(True),
                          flow_target=torch.from_numpy(flow), flow_valid_mask=torch.from_numpy(mask.astype(bool)))
            return result
        except Exception as error:
            failed = self._handles.pop(episode, None)
            if failed is not None:
                failed.close()
            self.rows = {key: row for key, row in self.rows.items() if key[0] != episode}
            raise DatasetIntegrityError(f"flow episode={episode} frame={frame}: {error}") from error
