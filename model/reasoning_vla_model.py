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
        ):
        super().__init__()

        self.action_expert_config = action_expert_config

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
        self.action_expert = FlowmatchingActionHead(self.action_expert_config, tune_action_expert)
        self.detach_vlm_outputs_for_action_expert = detach_vlm_outputs_for_action_expert
        
        if action_expert_name_or_path:
            print(f"load pre-trained weights from {action_expert_name_or_path} to initialize the action expert")
            self.action_expert.load_state_dict(
                load_file(os.path.join(action_expert_name_or_path, "action_expert.safetensors"))
            )

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
        action_expert_inputs = self.action_expert.prepare_inputs(batch_inputs)
        # print(f"prepare a batched samples spends {time.time()-start_time}s")
        return backbone_inputs, action_expert_inputs

    @torch.inference_mode()
    def get_action_direct(self, batch_inputs: BatchFeature, num_denoised_steps: int = 5) -> BatchFeature:
        with torch.no_grad():
            backbone_inputs = self.backbone.prepare_inputs(batch_inputs)
            backbone_outputs = self.backbone(
                backbone_inputs, compute_vlm_loss=False
            )
            self._validate_action_conditioning(
                backbone_outputs, expected_batch_size=batch_inputs["input_ids"].shape[0]
            )

            action_expert_inputs = self.action_expert.prepare_inputs(batch_inputs)
            action_expert_outputs = self.action_expert.get_action(
                backbone_outputs, action_expert_inputs, num_denoised_steps
            )
        # return action_pred
        return BatchFeature(data={"action_pred": action_expert_outputs["action_pred"]})
    
    @torch.inference_mode()
    def get_n_actions_direct(self, batch_inputs: BatchFeature, num_denoised_steps: int = 5, n: int = 8) -> BatchFeature:
        # sampling `n` different action chunks
        with torch.no_grad():
            backbone_inputs = self.backbone.prepare_inputs(batch_inputs)
            backbone_outputs = self.backbone(
                backbone_inputs, compute_vlm_loss=False
            )
            self._validate_action_conditioning(
                backbone_outputs, expected_batch_size=batch_inputs["input_ids"].shape[0]
            )

            action_expert_inputs = self.action_expert.prepare_inputs(batch_inputs)
            action_preds_list = []
            for i in range(n):
                action_expert_outputs = self.action_expert.get_action(
                    backbone_outputs, action_expert_inputs, num_denoised_steps
                )
                action_preds_list.append(action_expert_outputs["action_pred"][0])
        n_action_preds = torch.stack(action_preds_list, dim=0)

        # return action_pred
        return BatchFeature({"n_action_preds": n_action_preds})
  
    @torch.inference_mode()
    def get_action_subtask(self, batch_inputs: BatchFeature, num_denoised_steps: int = 5) -> BatchFeature:
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
            action_expert_inputs = self.action_expert.prepare_inputs(batch_inputs)
            action_expert_outputs = self.action_expert.get_action(
                backbone_outputs, action_expert_inputs, num_denoised_steps
            )
        # return action_pred
        return BatchFeature(data={"action_pred": action_expert_outputs["action_pred"]})

    def forward(self, batch_inputs, training_progress: float, dynamic_loss_type = "vlm_and_action", 
                vlm_loss_weight = 1.0, action_expert_loss_weight = 1.0) -> BatchFeature:
        if (
            self.use_difference_query
            and self.detach_vlm_outputs_for_action_expert
            and dynamic_loss_type == "action"
        ):
            raise ValueError(
                "detach_vlm_outputs_for_action_expert=True is incompatible with "
                "Difference Query action-only training"
            )
        # backbone_inputs, action_expert_inputs = self.prepare_vla_input(batch_inputs)
        backbone_inputs = self.backbone.prepare_inputs(batch_inputs)
        backbone_outputs = self.backbone(
            backbone_inputs,
            compute_vlm_loss=dynamic_loss_type in ("vlm", "vlm_and_action"),
        )
        vlm_loss = backbone_outputs["vlm_loss"]

        if dynamic_loss_type == "vlm":
            return BatchFeature(data={"loss": vlm_loss, "vlm_loss": vlm_loss})

        if self.detach_vlm_outputs_for_action_expert:
            backbone_outputs["backbone_embeddings"] = backbone_outputs["backbone_embeddings"].detach()

        self._validate_action_conditioning(
            backbone_outputs, expected_batch_size=batch_inputs["input_ids"].shape[0]
        )
        
        action_expert_inputs = self.action_expert.prepare_inputs(batch_inputs)
        action_expert_outputs = self.action_expert(backbone_outputs, action_expert_inputs, training_progress)
        action_expert_loss = action_expert_outputs["action_expert_loss"]

        if dynamic_loss_type == "action":
            return BatchFeature(data={"loss": action_expert_loss, "action_expert_loss": action_expert_loss})
        elif dynamic_loss_type == "vlm_and_action":
            return BatchFeature(
                data={
                      "loss": vlm_loss_weight * vlm_loss + action_expert_loss_weight * action_expert_loss,
                      "vlm_loss": vlm_loss,
                      "action_expert_loss": action_expert_loss,
                      "weighted_vlm_loss": vlm_loss_weight * vlm_loss,
                      "weighted_action_expert_loss": action_expert_loss_weight * action_expert_loss,
                      "vlm_loss weight": vlm_loss_weight,
                      "action_expert_loss weight": action_expert_loss_weight
                    }
            )
        else:
            raise ValueError(f"Unrecognized dynamic_loss_type: {dynamic_loss_type}. It should be [vlm, action, vlm_and_action]")
    
    @property
    def device(self):
        return next(iter(self.parameters())).device

    def save_pretrained(self, save_directory):
        os.makedirs(save_directory, exist_ok=True)
        # save backbone model parameter and model config (Qwen-VL)
        self.backbone.model.save_pretrained(save_directory)
        # save backnone's processor
        self.backbone.processor.save_pretrained(save_directory)
        # save action expert's model parameter
        save_file(self.action_expert.state_dict(), os.path.join(save_directory, "action_expert.safetensors"))
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
    ):
        # `save_directory` should contain complete VLA weights (a VLM and an action expert)
        with open(
            os.path.join(save_directory, "action_expert_config.json"), encoding="utf-8"
        ) as config_file:
            action_expert_config_json = json.load(config_file)
        action_expert_config = FlowmatchingActionHeadConfig(**action_expert_config_json)

        return cls(
            vlm_name_or_path = save_directory,
            action_expert_name_or_path = save_directory,
            action_expert_config = action_expert_config,
            tune_vlm = tune_vlm,
            tune_action_expert = tune_action_expert,
            detach_vlm_outputs_for_action_expert = detach_vlm_outputs_for_action_expert,
            use_difference_query = use_difference_query,
            num_difference_queries = num_difference_queries,
            vlm_attention_backend = vlm_attention_backend,
        )
