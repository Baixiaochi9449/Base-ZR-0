import copy

import pytest
import torch
from transformers.feature_extraction_utils import BatchFeature

from model.flow_matching_action_head import (
    FlowmatchingActionHead,
    FlowmatchingActionHeadConfig,
)
from utils.action_expert_config import architecture_config_hash


def _config(horizon):
    return FlowmatchingActionHeadConfig(
        add_pos_embed=True,
        vlm_output_embedding_dim=8,
        action_or_state_token_embedding_dim=8,
        mlp_hidden_size=8,
        max_seq_len=256,
        action_dim=64,
        state_dim=64,
        action_horizon=horizon,
        noise_beta_alpha=1.5,
        noise_beta_beta=1.0,
        noise_s=0.999,
        num_timestep_buckets=32,
        diffusion_transformer_cfg={
            "num_attention_heads": 2,
            "attention_head_dim": 4,
            "output_dim": 8,
            "num_layers": 1,
            "dropout": 0.0,
            "attention_bias": True,
            "activation_fn": "gelu-approximate",
            "upcast_attention": False,
            "norm_type": "ada_norm",
            "norm_elementwise_affine": False,
            "norm_eps": 1e-5,
            "max_num_positional_embeddings": 128,
            "positional_embeddings": None,
            "final_dropout": False,
            "interleave_self_attention": True,
            "causal_mask_in_self_attn": False,
        },
    )


def test_h32_to_h10_action_expert_state_dict_is_strict_and_shape_stable():
    torch.manual_seed(42)
    source = FlowmatchingActionHead(_config(32), tune_action_expert=True)
    target = FlowmatchingActionHead(_config(10), tune_action_expert=True)
    assert list(source.state_dict()) == list(target.state_dict())
    for name, value in source.state_dict().items():
        assert value.shape == target.state_dict()[name].shape, name
    target.load_state_dict(copy.deepcopy(source.state_dict()), strict=True)
    assert architecture_config_hash(_config(32).to_dict()) == architecture_config_hash(
        _config(10).to_dict()
    )


@pytest.mark.parametrize("source_horizon,target_horizon", [(16, 8), (8, 16), (16, 16)])
def test_horizon_override_is_not_limited_to_32_or_only_shortening(
    source_horizon, target_horizon
):
    source = FlowmatchingActionHead(_config(source_horizon), tune_action_expert=True)
    target = FlowmatchingActionHead(_config(target_horizon), tune_action_expert=True)
    assert list(source.state_dict()) == list(target.state_dict())
    assert {
        name: tuple(value.shape) for name, value in source.state_dict().items()
    } == {
        name: tuple(value.shape) for name, value in target.state_dict().items()
    }
    target.load_state_dict(copy.deepcopy(source.state_dict()), strict=True)


def test_h10_flow_matching_forward_backward_and_direct_action_shapes():
    torch.manual_seed(42)
    head = FlowmatchingActionHead(_config(10), tune_action_expert=True)
    batch_size = 2
    backbone = BatchFeature(
        data={
            "backbone_embeddings": torch.randn(batch_size, 32, 8),
            "action_expert_cross_attn_mask": torch.ones(
                batch_size, 32, dtype=torch.bool
            ),
        }
    )
    inputs = BatchFeature(
        data={
            "observation.state": torch.randn(batch_size, 1, 64),
            "state_mask": torch.ones(batch_size, 1, 64, dtype=torch.bool),
            "action": torch.randn(batch_size, 10, 64),
            "action_mask": torch.ones(batch_size, 10, 64, dtype=torch.bool),
        }
    )
    output = head(backbone, inputs, training_progress=0.0)
    loss = output["action_expert_loss"]
    assert torch.isfinite(loss)
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3)
    parameter_before = next(head.parameters()).detach().clone()
    loss.backward()
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in head.parameters()
        if parameter.requires_grad
    )
    optimizer.step()
    assert not torch.equal(next(head.parameters()).detach(), parameter_before)

    inference = head.get_action(
        backbone,
        BatchFeature(
            data={
                "observation.state": inputs["observation.state"],
                "state_mask": inputs["state_mask"],
                "infer_action_mask": torch.ones(
                    batch_size, 10, 64, dtype=torch.bool
                ),
            }
        ),
        num_denoised_steps=2,
    )
    assert inference["action_pred"].shape == (batch_size, 10, 64)


@pytest.mark.parametrize("horizon", [0, -1, 1.5, True])
def test_downstream_horizon_override_rejects_invalid_or_unsupported_values(horizon):
    from utils.action_expert_config import load_action_expert_config
    from pathlib import Path
    source = Path(__file__).resolve().parents[1] / "configs/stage05_four_dataset_action_expert.json"
    with pytest.raises(ValueError):
        load_action_expert_config(source, action_horizon_override=horizon)
