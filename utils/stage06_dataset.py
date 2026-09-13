"""LIBERO v3 video/parquet observations joined to optional Stage06 labels."""

from collections import OrderedDict
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from utils.dataset_spec import resolve_dataset_spec, resolve_objective_requirements
from utils.stage05_dataset import Stage05MixedPretrainingDataset
from utils.training_tokenization import run_with_same_sample_retries


class Stage06LiberoDataset(Stage05MixedPretrainingDataset):
    # Reuse the existing v3 parquet LRU, video offset lookup and decoder.
    def __init__(self, *, entry, processor, loss_type, max_length, action_horizon=32,
                 max_pad_state_and_action_length=64, dataset_id=0):
        from utils.optical_flow_reader import OpticalFlowReader
        from utils.aux_data_contract import flow_contract

        self.entry = dict(entry)
        self.root = Path(entry["dataset_path"]).resolve()
        self.processor, self.loss_type, self.max_length = processor, loss_type, max_length
        self.action_horizon, self.max_pad_length, self.dataset_id = action_horizon, max_pad_state_and_action_length, dataset_id
        self.camera_keys = list(entry["camera_keys"])
        if entry.get("geometric_augmentation", False):
            raise ValueError("OF forbids unsynchronized geometric augmentation")
        flow_contract_value = (flow_contract(entry) if entry.get("aux_dataset_identity") else None)
        self.flow_reader = OpticalFlowReader(entry["optical_flow_data_root"], entry["optical_flow_manifest"],
                                             delta_frames=entry["flow_delta_frames"],
                                             label_source=int(entry.get("flow_label_source", 1)),
                                             contract=flow_contract_value,
                                             schema_contract=entry.get("flow_schema_contract"))
        if self.flow_reader.camera_key not in self.camera_keys:
            raise ValueError("supervised flow camera must be present in observations")
        info = json.loads((self.root / "meta/info.json").read_text())
        self.manifest = {"generation": {"native_fps": info["fps"]}, "counts": {"source_frames": info["total_frames"]}}
        self.video_backend = "pyav"
        self.source_episode_metadata = {}
        for path in sorted((self.root / "meta/episodes").glob("**/*.parquet")):
            for row in pq.read_table(path).to_pylist():
                episode = int(row["episode_index"])
                if episode in self.source_episode_metadata:
                    raise ValueError("duplicate LIBERO episode metadata")
                self.source_episode_metadata[episode] = row
        self.episode_by_id = self.source_episode_metadata
        self._parquet_cache, self._episode_rows_cache = OrderedDict(), OrderedDict()
        self.steps = []
        for episode in sorted(self.source_episode_metadata):
            rows = self._episode_rows(episode)
            frames = [int(row["frame_index"]) for row in rows]
            if len(set(frames)) != len(frames):
                raise ValueError("duplicate LIBERO frame_index")
            self.steps.extend((episode, frame) for frame in frames)
        self.subset_indices = np.arange(len(self.steps))
        if not set(self.flow_reader.rows).issubset(set(self.steps)):
            raise ValueError("flow manifest declares frames absent from LIBERO parquet")
        ratio = float(entry.get("sample_ratio", 1))
        self.subset_indices = self.subset_indices[:max(1, int(len(self.steps) * ratio))]
        task_table = pq.read_table(self.root / "meta/tasks.parquet").to_pylist()
        self.tasks = {int(row["task_index"]): row.get("task", row.get("__index_level_0__")) for row in task_table}
        self.requirements = resolve_objective_requirements(loss_type, adapter="stage06_libero_flow", target_text_field=None)
        self.spec = resolve_dataset_spec(entry["dataset_entry"], entry, action_horizon=action_horizon,
                                         window_size=1, requirements=self.requirements)
        self.stats = self.spec.normalization_stats
        if self.spec.sidecar_sha256 != self.flow_reader.manifest_sha256:
            raise ValueError("Stage06 flow manifest changed during dataset initialization")

    def _columns(self):
        columns = ["episode_index", "frame_index", "index", "task_index", "timestamp"]
        if self.loss_type in {"action", "vlm_and_action"}:
            columns += ["observation.state", "action"]
        if self.entry.get("target_text_field"):
            columns.append(self.entry["target_text_field"])
        return columns

    def __len__(self):
        return len(self.subset_indices)

    def set_epoch(self, epoch):
        self._epoch = epoch

    def sampling_groups(self):
        groups = OrderedDict()
        for index, sampled in enumerate(self.subset_indices):
            groups.setdefault(self.steps[int(sampled)][0], []).append(index)
        return list(groups.values())

    def sampling_group_ranges(self):
        return [(group[0], group[-1] + 1) for group in self.sampling_groups()]

    def __getitem__(self, index):
        episode, frame = self.steps[int(self.subset_indices[index])]
        return run_with_same_sample_retries(lambda: self._sample(episode, frame),
            dataset_entry=self.entry["dataset_entry"], sample_id=f"episode={episode} frame={frame}", max_transient_retries=2)

    def _sample(self, episode, frame):
        from torchvision.transforms.functional import pil_to_tensor
        from utils.load_training_dataset import prepare_qwen_vl_inputs_cpu, prepare_action_expert_inputs_cpu

        rows = self._episode_rows(episode)
        by_frame = {int(row["frame_index"]): row for row in rows}
        row = by_frame[frame]
        data = {key: pil_to_tensor(self._decode_video(episode, key, float(row["timestamp"]))) for key in self.camera_keys}
        data["task"] = self.tasks[int(row["task_index"])]
        result = {}
        if self.requirements.requires_action:
            chunk = [by_frame.get(frame + offset) for offset in range(self.action_horizon)]
            data["observation.state"] = torch.tensor([row["observation.state"]], dtype=torch.float32)
            data["action"] = torch.tensor([item["action"] if item else row["action"] for item in chunk], dtype=torch.float32)
            data["action_is_pad"] = torch.tensor([item is None for item in chunk])
            result.update(prepare_action_expert_inputs_cpu(data, self.stats, self.max_pad_length, True,
                          dataset_entry=self.entry["dataset_entry"], sample_id=f"episode={episode} frame={frame}"))
        target_field = self.entry.get("target_text_field")
        has_target = bool(target_field and row.get(target_field)) and self.loss_type != "aux"
        if has_target:
            data[target_field] = row[target_field]
        requirements = replace(self.requirements, requires_target=has_target)
        result.update(prepare_qwen_vl_inputs_cpu(data, self.camera_keys, [], self.processor, "train", "", None,
                      max_length=self.max_length, requirements=requirements, target_text_field=target_field))
        if self.loss_type == "vlm_and_action" and "labels" not in result:
            result["labels"] = torch.full_like(result["input_ids"], -100)
        result["sub_task_flag"] = torch.tensor(0)
        result.update(dataset_id=torch.tensor(self.dataset_id), sample_global_index=torch.tensor(row["index"]),
                      ar_eligible=torch.tensor(has_target), fm_eligible=torch.tensor(self.requirements.requires_action))
        result.update(self.flow_reader.read(episode, frame))
        return result
