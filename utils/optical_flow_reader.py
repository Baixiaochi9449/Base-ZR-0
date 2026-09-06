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
    def __init__(self, root, manifest, *, delta_frames=10, camera_key="observation.images.image", max_handles=4):
        import h5py

        self.root = Path(root).resolve()
        self.manifest = Path(manifest)
        if not self.manifest.is_absolute():
            self.manifest = self.root / self.manifest
        self.delta_frames = delta_frames
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
            for entry in entries:
                episode = entry["merged_episode_index"]
                if not isinstance(episode, int) or episode < 0 or episode in self.episodes:
                    raise ValueError(f"duplicate/invalid episode {episode}")
                if entry["source_episode_index"] != episode or entry["camera_key"] != camera_key:
                    raise ValueError("flow episode/camera mapping mismatch")
                path = (self.root / entry["hdf5_path"]).resolve()
                if not path.is_relative_to(self.root):
                    raise ValueError("HDF5 path escapes root")
                with h5py.File(path, "r") as handle:
                    frames = self._validate_structure(handle, entry)
                self.episodes[episode] = (path, entry)
                for row, frame in enumerate(frames):
                    self.rows[(episode, int(frame))] = row
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
                    "source_episode_index": entry["source_episode_index"], "nominal_delta_frames": self.delta_frames,
                    "flow_units": "normalized_source_image_extent", "flow_direction": "forward_only",
                    "tail_policy": "clamp", "fps": 10.0,
                    "validity_semantics": "finite_and_forward_destination_in_bounds_not_occlusion"}
        for key, value in expected.items():
            if handle.attrs.get(key) != value:
                raise ValueError(f"flow metadata mismatch: {key}")
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
        return frames

    def close(self):
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_handles"] = OrderedDict()
        state["_pid"] = None
        return state

    def _handle(self, episode):
        import h5py

        if self._pid != os.getpid():
            self.close()
            self._pid = os.getpid()
        handle = self._handles.pop(episode, None)
        if handle is None:
            path, entry = self.episodes[episode]
            handle = h5py.File(path, "r")
            try:
                self._validate_structure(handle, entry)
            except Exception:
                handle.close()
                raise
        self._handles[episode] = handle
        while len(self._handles) > self.max_handles:
            self._handles.popitem(last=False)[1].close()
        return handle

    def read(self, episode, frame):
        result = {"flow_supervision_available": torch.tensor(False),
                  "flow_episode_id": torch.tensor(episode), "flow_frame_index": torch.tensor(frame),
                  "flow_actual_delta_frames": torch.tensor(-1), "flow_label_source": torch.tensor(-1),
                  "flow_target": None, "flow_valid_mask": None}
        row = self.rows.get((episode, frame))
        if row is None:
            return result
        try:
            handle = self._handle(episode)
            if int(handle["frame_index"][row]) != frame:
                raise ValueError("flow frame_index changed after manifest validation")
            delta, source = int(handle["actual_delta_frames"][row]), int(handle["label_source"][row])
            result.update(flow_actual_delta_frames=torch.tensor(delta), flow_label_source=torch.tensor(source))
            if delta != self.delta_frames or source != 1:
                return result
            flow, mask = handle["flow"][row].astype(np.float32), handle["valid_mask"][row]
            if not np.isin(mask, [0, 1]).all() or not np.isfinite(flow).all():
                raise ValueError("nonfinite flow or nonbinary mask")
            result.update(flow_supervision_available=torch.tensor(True),
                          flow_target=torch.from_numpy(flow), flow_valid_mask=torch.from_numpy(mask.astype(bool)))
            return result
        except Exception as error:
            raise DatasetIntegrityError(f"flow episode={episode} frame={frame}: {error}") from error
