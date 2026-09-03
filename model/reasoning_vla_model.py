from transformers.feature_extraction_utils import BatchFeature
from .qwen_vl_backbone import QwenVLBackbone
from torch import nn
from typing import Tuple
from .flow_matching_action_head import FlowmatchingActionHead, FlowmatchingActionHeadConfig
from .difference_query import (
    resolve_difference_query_config,
    save_difference_query_artifacts,
)

import torch
import json
import os
from safetensors.torch import save_file, load_file
import math
from pathlib import Path
from utils.dataset_manifest import write_resolved_dataset_manifest


_LOSS_TYPES = {"vlm", "action", "vlm_and_action"}
CHECKPOINT_METADATA_NAME = "zr0_checkpoint_metadata.json"
CHECKPOINT_METADATA_VERSION = 1


def _checkpoint_kind_for_loss_type(loss_type: str) -> str:
    return {
        "vlm": "ar_only",
        "action": "action_only",
        "vlm_and_action": "joint",
    }[loss_type]


def _read_checkpoint_kind(directory) -> str | None:
    if not directory:
        return None
    path = Path(directory) / CHECKPOINT_METADATA_NAME
    if not path.is_file():
        return None
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:
        raise ValueError(f"failed to read checkpoint metadata {path}: {error}") from error
    kind = metadata.get("checkpoint_kind")
    if metadata.get("version") != CHECKPOINT_METADATA_VERSION or kind not in {
        "ar_only",
        "joint",
        "action_only",
    }:
        raise ValueError(f"invalid checkpoint metadata in {path}")
    return kind

class ZR0Model(nn.Module):
    def __init__(
            self,
            vlm_name_or_path: str,
            action_expert_name_or_path: str,
            action_expert_config: FlowmatchingActionHeadConfig,
            tune_vlm=True,
            tune_action_expert=True,
            detach_vlm_outputs_for_action_expert=False,
            lora_args=None,
            use_difference_query=None,
            num_difference_queries=None,
            vlm_attention_backend=None,
            loss_type="vlm_and_action",
        ):
        super().__init__()

        if loss_type not in _LOSS_TYPES:
            raise ValueError(f"loss_type must be one of {sorted(_LOSS_TYPES)}")
        if loss_type == "vlm":
            if not tune_vlm:
                raise ValueError("loss_type=vlm requires tune_vlm=True")
            if tune_action_expert:
                raise ValueError("loss_type=vlm requires tune_action_expert=False")
            if action_expert_name_or_path:
                raise ValueError(
                    "loss_type=vlm does not accept action_expert_name_or_path"
                )
        self.loss_type = loss_type
        self.resolved_dataset_manifest = None
        source_kind = _read_checkpoint_kind(vlm_name_or_path)
        if loss_type == "vlm" and source_kind not in (None, "ar_only"):
            raise ValueError(
                f"checkpoint kind {source_kind!r} cannot be loaded for loss_type=vlm"
            )
        if loss_type == "vlm_and_action" and source_kind == "action_only":
            raise ValueError(
                "checkpoint kind 'action_only' cannot provide the VLM for joint training"
            )
        if loss_type == "vlm_and_action" and source_kind == "ar_only" and action_expert_name_or_path:
            raise ValueError(
                "joint warm start from an ar_only checkpoint must randomly initialize "
                "the Action Expert and forbids action_expert_name_or_path"
            )
        if loss_type == "vlm_and_action" and source_kind == "joint":
            same_checkpoint = action_expert_name_or_path and (
                Path(action_expert_name_or_path).resolve()
                == Path(vlm_name_or_path).resolve()
            )
            if not same_checkpoint:
                raise ValueError(
                    "joint checkpoint resume requires the same checkpoint as the "
                    "Action Expert weight source"
                )
        action_kind = _read_checkpoint_kind(action_expert_name_or_path)
        if action_kind == "ar_only":
            raise ValueError("checkpoint kind 'ar_only' cannot provide Action Expert weights")

        self.action_expert_config = action_expert_config
        action_horizon = action_expert_config.action_horizon
        max_seq_len = action_expert_config.max_seq_len
        invalid_horizon = (
            isinstance(action_horizon, bool)
            or not isinstance(action_horizon, int)
            or action_horizon <= 0
        )
        invalid_max_seq_len = (
            isinstance(max_seq_len, bool)
            or not isinstance(max_seq_len, int)
            or max_seq_len <= 0
        )
        if invalid_horizon or invalid_max_seq_len or action_horizon > max_seq_len:
            raise ValueError(
                "Action Expert action_horizon must be a positive integer no greater "
                f"than max_seq_len; got action_horizon={action_horizon}, "
                f"max_seq_len={max_seq_len}"
            )

        difference_query_config = resolve_difference_query_config(
            vlm_name_or_path,
            action_expert_name_or_path,
            use_difference_query=use_difference_query,
            num_difference_queries=num_difference_queries,
            vlm_attention_backend=vlm_attention_backend,
        )
        self.use_difference_query = difference_query_config.enabled
        self.num_difference_queries = (
            difference_query_config.num_difference_queries
        )
        self.vlm_attention_backend = difference_query_config.attention_backend

        self.backbone = QwenVLBackbone(
            vlm_name_or_path,
            tune_vlm,
            lora_args,
            resolved_difference_query_config=difference_query_config,
        )
        print("the size of VLM's last layer hidden state:", self.backbone.model.config.text_config.hidden_size)
        self.action_expert_config.vlm_output_embedding_dim = self.backbone.model.config.text_config.hidden_size
        # self.action_expert_config.vlm_output_embedding_dim = self.backbone.model.config.hidden_size
        self.action_expert = None
        if loss_type != "vlm":
            self.action_expert = FlowmatchingActionHead(
                self.action_expert_config, tune_action_expert
            )
        self.detach_vlm_outputs_for_action_expert = detach_vlm_outputs_for_action_expert
        
        if action_expert_name_or_path:
            print(f"load pre-trained weights from {action_expert_name_or_path} to initialize the action expert")
            self.action_expert.load_state_dict(
                load_file(os.path.join(action_expert_name_or_path, "action_expert.safetensors"))
            )

    def _require_action_expert(self) -> FlowmatchingActionHead:
        if self.action_expert is None:
            raise ValueError("this ar_only model has no Action Expert")
        return self.action_expert

    def _validate_action_conditioning(
        self, backbone_outputs: BatchFeature, expected_batch_size: int
    ) -> None:
        if not self.use_difference_query:
            return

        embeddings = backbone_outputs.get("backbone_embeddings")
        attention_mask = backbone_outputs.get("action_expert_cross_attn_mask")
        if not isinstance(embeddings, torch.Tensor) or embeddings.ndim != 3:
            raise ValueError("Difference Query action conditioning must have shape [B, Nq, H]")
        if embeddings.shape[0] != expected_batch_size:
            raise ValueError(
                "Difference Query action conditioning batch mismatch: "
                f"{embeddings.shape[0]} != {expected_batch_size}"
            )
        if embeddings.shape[1] != self.num_difference_queries:
            raise ValueError(
                "Difference Query action conditioning Nq mismatch: "
                f"{embeddings.shape[1]} != {self.num_difference_queries}"
            )
        expected_hidden_size = self.action_expert_config.vlm_output_embedding_dim
        if embeddings.shape[2] != expected_hidden_size:
            raise ValueError(
                "Difference Query action conditioning hidden size mismatch: "
                f"{embeddings.shape[2]} != {expected_hidden_size}"
            )
        expected_mask_shape = embeddings.shape[:2]
        if not isinstance(attention_mask, torch.Tensor) or attention_mask.shape != expected_mask_shape:
            actual_shape = getattr(attention_mask, "shape", None)
            raise ValueError(
                "Difference Query action conditioning mask shape mismatch: "
                f"{actual_shape} != {expected_mask_shape}"
            )
        if attention_mask.dtype != torch.bool:
            raise ValueError("Difference Query action conditioning mask must have bool dtype")
        if attention_mask.device != embeddings.device:
            raise ValueError(
                "Difference Query action conditioning mask and hidden states must share a device"
            )
        if not attention_mask.all():
            raise ValueError("Difference Query action conditioning mask must be all True")
        if not torch.isfinite(embeddings).all():
            raise ValueError("Difference Query action conditioning hidden states must be finite")

    def prepare_vla_input(self, batch_inputs) -> Tuple[BatchFeature, BatchFeature]:
        backbone_inputs = self.backbone.prepare_inputs(batch_inputs)
        action_expert_inputs = self._require_action_expert().prepare_inputs(batch_inputs)
        # print(f"prepare a batched samples spends {time.time()-start_time}s")
        return backbone_inputs, action_expert_inputs

    @torch.inference_mode()
    def get_action_direct(self, batch_inputs: BatchFeature, num_denoised_steps: int = 5) -> BatchFeature:
        action_expert = self._require_action_expert()
        with torch.no_grad():
            backbone_inputs = self.backbone.prepare_inputs(batch_inputs)
            backbone_outputs = self.backbone(
                backbone_inputs, compute_vlm_loss=False
            )
            self._validate_action_conditioning(
                backbone_outputs, expected_batch_size=batch_inputs["input_ids"].shape[0]
            )

            action_expert_inputs = action_expert.prepare_inputs(batch_inputs)
            action_expert_outputs = action_expert.get_action(
                backbone_outputs, action_expert_inputs, num_denoised_steps
            )
        # return action_pred
        return BatchFeature(data={"action_pred": action_expert_outputs["action_pred"]})
    
    @torch.inference_mode()
    def get_n_actions_direct(self, batch_inputs: BatchFeature, num_denoised_steps: int = 5, n: int = 8) -> BatchFeature:
        # sampling `n` different action chunks
        action_expert = self._require_action_expert()
        with torch.no_grad():
            backbone_inputs = self.backbone.prepare_inputs(batch_inputs)
            backbone_outputs = self.backbone(
                backbone_inputs, compute_vlm_loss=False
            )
            self._validate_action_conditioning(
                backbone_outputs, expected_batch_size=batch_inputs["input_ids"].shape[0]
            )

            action_expert_inputs = action_expert.prepare_inputs(batch_inputs)
            action_preds_list = []
            for i in range(n):
                action_expert_outputs = action_expert.get_action(
                    backbone_outputs, action_expert_inputs, num_denoised_steps
                )
                action_preds_list.append(action_expert_outputs["action_pred"][0])
        n_action_preds = torch.stack(action_preds_list, dim=0)

        # return action_pred
        return BatchFeature({"n_action_preds": n_action_preds})
  
    @torch.inference_mode()
    def get_action_subtask(self, batch_inputs: BatchFeature, num_denoised_steps: int = 5) -> BatchFeature:
        action_expert = self._require_action_expert()
        if self.use_difference_query:
            raise NotImplementedError(
                "subtask/autoregressive generation is not implemented with Difference Query"
            )
        with torch.no_grad():
            prompt_len = len(batch_inputs["input_ids"][0])
            
            # 1. generate subtask
            new_input_ids = self.backbone.generate(
                input_ids = batch_inputs["input_ids"],
                attention_mask = batch_inputs["attention_mask"],
                pixel_values = batch_inputs["pixel_values"],
                image_grid_thw = batch_inputs["image_grid_thw"],
                do_sample=False,
                eos_token_id=[153718]
            )
            new_attention_mask = torch.ones_like(new_input_ids, dtype=torch.long)

            print("output subtask:", self.backbone.processor.tokenizer.decode(new_input_ids[0][prompt_len:]))

            # 2. perpare new VLM inputs (prompt + subtask)
            backbone_inputs = {
                "input_ids": new_input_ids,
                "attention_mask": new_attention_mask,
                "pixel_values": batch_inputs["pixel_values"],
                "image_grid_thw": batch_inputs["image_grid_thw"],
                "sub_task_flag": batch_inputs["sub_task_flag"]
            }
            # 3. VLM forward
            backbone_outputs = self.backbone(
                backbone_inputs, compute_vlm_loss=False
            )
            
            # 4. action expert diffusion
            action_expert_inputs = action_expert.prepare_inputs(batch_inputs)
            action_expert_outputs = action_expert.get_action(
                backbone_outputs, action_expert_inputs, num_denoised_steps
            )
        # return action_pred
        return BatchFeature(data={"action_pred": action_expert_outputs["action_pred"]})

    def _resolve_training_loss_type(self, dynamic_loss_type, loss_type) -> str:
        if dynamic_loss_type is not None and loss_type is not None and dynamic_loss_type != loss_type:
            raise ValueError(
                "dynamic_loss_type and loss_type must match when both are provided"
            )
        requested_loss_type = loss_type if loss_type is not None else dynamic_loss_type
        if requested_loss_type is None:
            return self.loss_type
        if requested_loss_type not in _LOSS_TYPES:
            raise ValueError(
                f"Unrecognized loss_type: {requested_loss_type}. "
                "Expected one of [vlm, action, vlm_and_action]"
            )
        if requested_loss_type != self.loss_type:
            raise ValueError(
                f"forward loss mode mismatch: constructed mode={self.loss_type!r}, "
                f"requested mode={requested_loss_type!r}"
            )
        return self.loss_type

    @staticmethod
    def _validate_loss_weight(value, name: str, *, required: bool) -> float:
        try:
            numeric_value = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{name} must be a finite non-negative number") from error
        if not math.isfinite(numeric_value) or numeric_value < 0:
            raise ValueError(f"{name} must be finite and non-negative")
        if required and numeric_value == 0:
            raise ValueError(f"{name} must be greater than zero for this loss_type")
        return numeric_value

    @staticmethod
    def _validate_training_inputs(batch_inputs, loss_type: str) -> None:
        input_ids = batch_inputs.get("input_ids")
        if not isinstance(input_ids, torch.Tensor) or input_ids.ndim < 1:
            raise ValueError("input_ids must be a non-empty Tensor with a batch dimension")
        batch_size = input_ids.shape[0]
        if loss_type in ("vlm", "vlm_and_action"):
            labels = batch_inputs.get("labels")
            if not isinstance(labels, torch.Tensor):
                raise ValueError("labels are required for vlm and vlm_and_action training")
            if labels.shape != input_ids.shape:
                raise ValueError("labels shape must match input_ids")
            if not (labels != -100).any(dim=1).all():
                raise ValueError("labels must contain supervision for each sample, not only -100")
        if loss_type in ("action", "vlm_and_action"):
            required_keys = (
                "observation.state",
                "state_mask",
                "action",
                "action_mask",
            )
            for key in required_keys:
                if key not in batch_inputs:
                    raise ValueError(f"{key} is required for action training")
            tensors = {}
            for key in required_keys:
                value = batch_inputs[key]
                if not isinstance(value, torch.Tensor):
                    raise ValueError(f"{key} must be a non-empty Tensor")
                if value.numel() == 0 or value.ndim < 1:
                    raise ValueError(f"{key} must be a non-empty Tensor")
                if value.shape[0] != batch_size:
                    raise ValueError(f"{key} batch dimension must match input_ids")
                tensors[key] = value
            if tensors["state_mask"].shape != tensors["observation.state"].shape:
                raise ValueError("state_mask shape must match observation.state")
            if tensors["action_mask"].shape != tensors["action"].shape:
                raise ValueError("action_mask shape must match action")
            action_supervision = batch_inputs.get("action_supervision_available")
            if action_supervision is None:
                action_supervision = torch.ones(
                    batch_size, dtype=torch.bool, device=input_ids.device
                )
            elif (
                not isinstance(action_supervision, torch.Tensor)
                or action_supervision.shape != (batch_size,)
            ):
                raise ValueError(
                    "action_supervision_available must have shape [batch_size]"
                )
            action_supervision = action_supervision.to(
                device=input_ids.device, dtype=torch.bool
            )
            if loss_type == "action" and not action_supervision.all():
                raise ValueError(
                    "action-only training does not accept samples without action supervision"
                )
            per_sample_state_mask = tensors["state_mask"].to(dtype=torch.bool).reshape(batch_size, -1)
            per_sample_action_mask = tensors["action_mask"].to(dtype=torch.bool).reshape(batch_size, -1)
            state_valid = per_sample_state_mask.any(dim=1)
            action_valid = per_sample_action_mask.any(dim=1)
            if (action_supervision & ~state_valid).any():
                raise ValueError(
                    "VLA state_mask must contain a valid state for each sample with action supervision"
                )
            if (action_supervision & ~action_valid).any():
                raise ValueError(
                    "VLA action_mask must contain a valid action for each sample with action supervision"
                )
            if (~action_supervision & action_valid).any():
                raise ValueError(
                    "samples without action supervision must have an all-zero action_mask"
                )

    @staticmethod
    def _loss_outputs(
        *,
        ar_loss,
        ar_loss_sum,
        ar_loss_count,
        flow_matching_loss,
        flow_matching_loss_sum,
        flow_matching_loss_count,
        vlm_loss_weight: float,
        action_expert_loss_weight: float,
    ) -> BatchFeature:
        data = {
            "vlm_loss_weight": vlm_loss_weight,
            "action_expert_loss_weight": action_expert_loss_weight,
            "vlm_loss weight": vlm_loss_weight,
            "action_expert_loss weight": action_expert_loss_weight,
        }
        total_loss = None
        if ar_loss is not None:
            weighted_ar_loss = vlm_loss_weight * ar_loss
            data.update(
                {
                    "vlm_loss": ar_loss,
                    "ar_loss": ar_loss,
                    "weighted_vlm_loss": weighted_ar_loss,
                    "weighted_ar_loss": weighted_ar_loss,
                    "ar_loss_sum": ar_loss_sum,
                    "ar_loss_count": ar_loss_count,
                }
            )
            total_loss = weighted_ar_loss
        if flow_matching_loss is not None:
            weighted_flow_matching_loss = action_expert_loss_weight * flow_matching_loss
            data.update(
                {
                    "action_expert_loss": flow_matching_loss,
                    "flow_matching_loss": flow_matching_loss,
                    "weighted_action_expert_loss": weighted_flow_matching_loss,
                    "weighted_flow_matching_loss": weighted_flow_matching_loss,
                    "flow_matching_loss_sum": flow_matching_loss_sum,
                    "flow_matching_loss_count": flow_matching_loss_count,
                }
            )
            total_loss = (
                weighted_flow_matching_loss
                if total_loss is None
                else total_loss + weighted_flow_matching_loss
            )
        data["total_loss"] = total_loss
        data["loss"] = total_loss
        return BatchFeature(data=data)

    def forward(
        self,
        batch_inputs,
        training_progress: float,
        dynamic_loss_type=None,
        vlm_loss_weight=1.0,
        action_expert_loss_weight=1.0,
        *,
        loss_type=None,
    ) -> BatchFeature:
        resolved_loss_type = self._resolve_training_loss_type(
            dynamic_loss_type, loss_type
        )
        if self.action_expert is None and resolved_loss_type != "vlm":
            raise ValueError("ar_only model only supports loss_type=vlm")
        vlm_loss_weight = self._validate_loss_weight(
            vlm_loss_weight,
            "vlm_loss_weight",
            required=resolved_loss_type in ("vlm", "vlm_and_action"),
        )
        action_expert_loss_weight = self._validate_loss_weight(
            action_expert_loss_weight,
            "action_expert_loss_weight",
            required=resolved_loss_type in ("action", "vlm_and_action"),
        )
        if (
            self.use_difference_query
            and self.detach_vlm_outputs_for_action_expert
            and resolved_loss_type in ("action", "vlm_and_action")
        ):
            raise ValueError(
                "detach_vlm_outputs_for_action_expert=True is incompatible with "
                "Difference Query action training"
            )
        self._validate_training_inputs(batch_inputs, resolved_loss_type)
        if resolved_loss_type in ("action", "vlm_and_action"):
            actual_horizon = batch_inputs["action"].shape[-2]
            expected_horizon = self.action_expert_config.action_horizon
            if actual_horizon != expected_horizon:
                raise ValueError(
                    f"action horizon mismatch: model={expected_horizon}, batch={actual_horizon}"
                )
        backbone_inputs = self.backbone.prepare_inputs(batch_inputs)
        backbone_outputs = self.backbone(
            backbone_inputs,
            compute_vlm_loss=resolved_loss_type in ("vlm", "vlm_and_action"),
        )
        vlm_loss = backbone_outputs.get("vlm_loss")
        ar_loss_count = None
        ar_loss_sum = None
        if isinstance(vlm_loss, torch.Tensor):
            ar_loss_count = (
                batch_inputs["labels"][..., 1:].ne(-100).sum().detach().to(torch.float32)
            )
            ar_loss_sum = vlm_loss.float() * ar_loss_count

        if resolved_loss_type == "vlm":
            if not isinstance(vlm_loss, torch.Tensor):
                raise ValueError("backbone did not return vlm_loss for vlm training")
            return self._loss_outputs(
                ar_loss=vlm_loss,
                ar_loss_sum=ar_loss_sum,
                ar_loss_count=ar_loss_count,
                flow_matching_loss=None,
                flow_matching_loss_sum=None,
                flow_matching_loss_count=None,
                vlm_loss_weight=vlm_loss_weight,
                action_expert_loss_weight=action_expert_loss_weight,
            )

        if self.detach_vlm_outputs_for_action_expert:
            backbone_outputs["backbone_embeddings"] = backbone_outputs["backbone_embeddings"].detach()

        self._validate_action_conditioning(
            backbone_outputs, expected_batch_size=batch_inputs["input_ids"].shape[0]
        )
        
        action_expert = self._require_action_expert()
        action_expert_inputs = action_expert.prepare_inputs(batch_inputs)
        action_expert_outputs = action_expert(
            backbone_outputs, action_expert_inputs, training_progress
        )
        action_expert_loss = action_expert_outputs["action_expert_loss"]
        flow_matching_loss_count = action_expert_outputs.get(
            "flow_matching_loss_count"
        )
        flow_matching_loss_sum = action_expert_outputs.get("flow_matching_loss_sum")
        if flow_matching_loss_count is None or flow_matching_loss_sum is None:
            flow_matching_loss_count = (
                batch_inputs["action_mask"].sum().detach().to(torch.float32)
            )
            flow_matching_loss_sum = (
                action_expert_loss.float() * flow_matching_loss_count
            )

        if resolved_loss_type == "vlm_and_action" and not isinstance(vlm_loss, torch.Tensor):
            raise ValueError("backbone did not return vlm_loss for vlm_and_action training")
        return self._loss_outputs(
            ar_loss=vlm_loss if resolved_loss_type == "vlm_and_action" else None,
            ar_loss_sum=ar_loss_sum if resolved_loss_type == "vlm_and_action" else None,
            ar_loss_count=ar_loss_count if resolved_loss_type == "vlm_and_action" else None,
            flow_matching_loss=action_expert_loss,
            flow_matching_loss_sum=flow_matching_loss_sum,
            flow_matching_loss_count=flow_matching_loss_count,
            vlm_loss_weight=vlm_loss_weight,
            action_expert_loss_weight=action_expert_loss_weight,
        )
    
    @property
    def device(self):
        return next(iter(self.parameters())).device

    def save_pretrained(self, save_directory):
        os.makedirs(save_directory, exist_ok=True)
        # save backbone model parameter and model config (Qwen-VL)
        self.backbone.model.save_pretrained(save_directory)
        # save backnone's processor
        self.backbone.processor.save_pretrained(save_directory)
        action_weight_path = os.path.join(save_directory, "action_expert.safetensors")
        if self.action_expert is not None:
            save_file(self.action_expert.state_dict(), action_weight_path)
        elif os.path.exists(action_weight_path):
            os.unlink(action_weight_path)
        # save action expert's config
        with open(os.path.join(save_directory, "action_expert_config.json"), "w") as json_file:
            json.dump(self.action_expert_config.to_dict(), json_file, indent=4)
        save_difference_query_artifacts(
            save_directory,
            enabled=self.use_difference_query,
            hidden_size=self.action_expert_config.vlm_output_embedding_dim,
            difference_query=(
                self.backbone.difference_query.weight
                if self.use_difference_query
                else None
            ),
        )
        with open(
            os.path.join(save_directory, CHECKPOINT_METADATA_NAME),
            "w",
            encoding="utf-8",
        ) as metadata_file:
            json.dump(
                {
                    "version": CHECKPOINT_METADATA_VERSION,
                    "checkpoint_kind": _checkpoint_kind_for_loss_type(self.loss_type),
                },
                metadata_file,
                indent=2,
                sort_keys=True,
            )
            metadata_file.write("\n")
        if self.resolved_dataset_manifest is not None:
            write_resolved_dataset_manifest(
                save_directory, self.resolved_dataset_manifest
            )
    
    @classmethod
    def from_pretrained(
        cls,
        save_directory,
        tune_vlm=False,
        tune_action_expert=False,
        detach_vlm_outputs_for_action_expert=False,
        use_difference_query=None,
        num_difference_queries=None,
        vlm_attention_backend=None,
        loss_type="vlm_and_action",
        allow_ar_warm_start=False,
        for_action_inference=False,
    ):
        checkpoint_kind = _read_checkpoint_kind(save_directory)
        action_weights = Path(save_directory) / "action_expert.safetensors"
        if checkpoint_kind is None:
            checkpoint_kind = "legacy_full" if action_weights.is_file() else "legacy_ar"
            print(
                f"legacy checkpoint without {CHECKPOINT_METADATA_NAME}: "
                f"using conservative {checkpoint_kind} compatibility path"
            )
        if for_action_inference:
            if checkpoint_kind in {"ar_only", "legacy_ar"}:
                raise ValueError(
                    f"checkpoint kind {checkpoint_kind!r} cannot be used for action inference"
                )
            if checkpoint_kind not in {"action_only", "joint", "legacy_full"}:
                raise ValueError(
                    f"checkpoint kind {checkpoint_kind!r} is not action-inference compatible"
                )
            loss_type = "action"
        if loss_type == "vlm" and checkpoint_kind not in {"ar_only", "legacy_ar", "legacy_full"}:
            raise ValueError(
                f"checkpoint kind {checkpoint_kind!r} cannot be loaded for loss_type=vlm"
            )
        if loss_type == "vlm_and_action" and checkpoint_kind == "action_only":
            raise ValueError(
                "checkpoint kind 'action_only' cannot initialize loss_type=vlm_and_action"
            )
        if (
            loss_type == "vlm_and_action"
            and checkpoint_kind in {"ar_only", "legacy_ar"}
            and not allow_ar_warm_start
        ):
            raise ValueError(
                "AR checkpoint to joint initialization requires "
                "allow_ar_warm_start=True"
            )
        if (
            loss_type == "action"
            and checkpoint_kind not in {"action_only", "legacy_full"}
            and not (for_action_inference and checkpoint_kind == "joint")
        ):
            raise ValueError(
                f"checkpoint kind {checkpoint_kind!r} cannot be loaded for loss_type=action"
            )
        with open(
            os.path.join(save_directory, "action_expert_config.json"), encoding="utf-8"
        ) as config_file:
            action_expert_config_json = json.load(config_file)
        action_expert_config = FlowmatchingActionHeadConfig(**action_expert_config_json)

        load_action_weights = (
            loss_type != "vlm"
            and checkpoint_kind in {"joint", "action_only", "legacy_full"}
        )
        return cls(
            vlm_name_or_path = save_directory,
            action_expert_name_or_path=(save_directory if load_action_weights else None),
            action_expert_config = action_expert_config,
            tune_vlm = tune_vlm,
            tune_action_expert = tune_action_expert,
            detach_vlm_outputs_for_action_expert = detach_vlm_outputs_for_action_expert,
            use_difference_query = use_difference_query,
            num_difference_queries = num_difference_queries,
            vlm_attention_backend = vlm_attention_backend,
            loss_type=loss_type,
        )
