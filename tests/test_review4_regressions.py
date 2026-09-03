import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel


def _v2_stats(state_dim=2, action_dim=2):
    return {
        "observation.state": {
            "q01": np.zeros(state_dim, dtype=np.float32),
            "q99": np.ones(state_dim, dtype=np.float32),
        },
        "action": {
            "q01": np.zeros(action_dim, dtype=np.float32),
            "q99": np.ones(action_dim, dtype=np.float32),
        },
    }


def test_v2_action_padding_masks_temporal_tail_and_fast_excludes_it(monkeypatch):
    from utils.dataset_spec import resolve_objective_requirements
    from utils import load_training_dataset as loader

    data = {
        "observation.state": torch.tensor([[0.25, 0.75]]),
        "action": torch.tensor(
            [[0.1, 0.2], [0.3, 0.4], [0.3, 0.4], [0.3, 0.4]]
        ),
        "action_is_pad": torch.tensor([False, False, True, True]),
        "front": torch.zeros(3, 4, 4),
        "task": "move",
        "embodied_cot": json.dumps({"Finished": "No"}),
    }
    action_inputs = loader.prepare_action_expert_inputs_cpu(
        data, _v2_stats(), max_pad_length=4, use_quantile=True
    )
    assert action_inputs["action_mask"].dtype == torch.bool
    assert action_inputs["action_mask"].tolist() == [
        [True, True, False, False],
        [True, True, False, False],
        [False, False, False, False],
        [False, False, False, False],
    ]
    assert action_inputs["action_supervision_available"].item()

    seen = {}

    class Fast:
        def __call__(self, actions):
            seen["fast_actions"] = actions.clone()
            return [[10 + index for index in range(actions.shape[0])]]

    def fake_tokenize(_msg, _mode, _processor, **kwargs):
        seen["target"] = kwargs["target"]
        return {"input_ids": torch.tensor([1]), "attention_mask": torch.tensor([1])}

    monkeypatch.setattr(loader, "tokenize_vision_language_inputs", fake_tokenize)
    requirements = resolve_objective_requirements(
        "vlm_and_action", adapter="lerobot_v2", target_text_field=None
    )
    loader.prepare_qwen_vl_inputs_cpu(
        data=data,
        camera_keys=["front"],
        grounding_camera_keys=[],
        processor=object(),
        process_mode="train",
        prompt_suffix="",
        fast_tokenizer=Fast(),
        requirements=requirements,
        dataset_entry="tail-v2",
        sample_id="episode=0 frame=1 sample=1",
    )
    assert seen["fast_actions"].shape == (2, 2)
    assert "<robot_action_10><robot_action_11>" in seen["target"]
    assert "<robot_action_12>" not in seen["target"]


def test_v2_streaming_tail_keeps_action_is_pad_without_cross_episode(monkeypatch):
    from utils.dataset_spec import resolve_objective_requirements
    from utils import load_training_dataset as loader

    class Table:
        features = {"action": object()}
        column_names = ["action"]

    class Meta:
        fps = 30
        camera_keys = ["front"]
        grounding_camera_keys = []
        stats = _v2_stats()

    class Dataset:
        meta = Meta()
        hf_dataset = Table()

        def __len__(self):
            return 1

        def getitem_with_delta_timestamps(self, index, delta_timestamps):
            assert index == 0
            assert delta_timestamps["action"] == [0.0, 1 / 30, 2 / 30, 3 / 30]
            return {
                "episode_index": torch.tensor(0),
                "index": torch.tensor(2),
                "task": "move",
                "front": torch.zeros(3, 4, 4),
                "observation.state": torch.tensor([[0.2, 0.4]]),
                "action": torch.tensor(
                    [[0.1, 0.2], [0.3, 0.4], [0.3, 0.4], [0.3, 0.4]]
                ),
                "action_is_pad": torch.tensor([False, False, True, True]),
            }

    monkeypatch.setattr(
        loader,
        "prepare_qwen_vl_inputs_cpu",
        lambda **_: {"input_ids": torch.tensor([1]), "attention_mask": torch.tensor([1])},
    )
    dataset = loader.StreamingLeRobotSampleDataset(
        lerobot_dataset=Dataset(),
        use_quantile=True,
        sample_ratio=1.0,
        processor=object(),
        fast_tokenizer=None,
        window_size=1,
        action_horizon=4,
        process_mode="train",
        dataset_id=0,
        max_pad_state_and_action_length=4,
        dataset_entry="tail-v2",
        requirements=resolve_objective_requirements(
            "action", adapter="lerobot_v2", target_text_field=None
        ),
    )
    sample = dataset[0]
    assert sample["action_mask"][:2, :2].all()
    assert not sample["action_mask"][2:].any()
    assert sample["action_supervision_available"].item()


class _DistributedMaskedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, values, mask):
        from model.flow_matching_action_head import distributed_masked_mean

        return distributed_masked_mean((self.weight * values).square(), mask)


def _distributed_masked_mean_worker(rank, init_file, zero_rank, output_dir):
    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=2
    )
    try:
        model = DistributedDataParallel(_DistributedMaskedModel())
        values = torch.tensor([1.0, 3.0]) if rank == 0 else torch.tensor([5.0, 7.0])
        mask = (
            torch.tensor([True, True])
            if rank == 0
            else torch.tensor([False, False] if zero_rank else [True, False])
        )
        loss = model(values, mask)
        loss.backward()
        torch.save(
            {"loss": float(loss.detach()), "grad": float(model.module.weight.grad)},
            Path(output_dir) / f"rank-{rank}.pt",
        )
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize(
    ("zero_rank", "expected_loss", "expected_grad"),
    [(False, 35.0 / 3.0, 70.0 / 3.0), (True, 5.0, 10.0)],
)
def test_distributed_masked_mean_uses_global_valid_elements(
    tmp_path, zero_rank, expected_loss, expected_grad
):
    init_file = tmp_path / "gloo-init"
    mp.spawn(
        _distributed_masked_mean_worker,
        args=(str(init_file), zero_rank, str(tmp_path)),
        nprocs=2,
        join=True,
    )
    for rank in range(2):
        result = torch.load(tmp_path / f"rank-{rank}.pt", weights_only=True)
        assert result["loss"] == pytest.approx(expected_loss)
        assert result["grad"] == pytest.approx(expected_grad)


def _joint_batch(action_available, action_mask):
    batch_size = len(action_available)
    return {
        "input_ids": torch.ones(batch_size, 3, dtype=torch.long),
        "labels": torch.tensor([[-100, 1, -100]] * batch_size),
        "observation.state": torch.ones(batch_size, 2),
        "state_mask": torch.tensor(
            [[True, True] if available else [False, False] for available in action_available]
        ),
        "action": torch.ones(batch_size, 2, 2),
        "action_mask": torch.as_tensor(action_mask, dtype=torch.bool),
        "action_supervision_available": torch.tensor(action_available),
    }


def test_joint_validation_allows_vqa_only_and_mixed_but_checks_vla():
    from model.reasoning_vla_model import ZR0Model

    zero = [[[False, False], [False, False]]]
    ZR0Model._validate_training_inputs(_joint_batch([False], zero), "vlm_and_action")
    mixed_mask = [
        [[True, True], [True, True]],
        [[False, False], [False, False]],
    ]
    ZR0Model._validate_training_inputs(
        _joint_batch([True, False], mixed_mask), "vlm_and_action"
    )
    with pytest.raises(ValueError, match="VLA.*valid action|action supervision"):
        ZR0Model._validate_training_inputs(
            _joint_batch([True], zero), "vlm_and_action"
        )
    with pytest.raises(ValueError, match="action-only.*VQA|action supervision"):
        ZR0Model._validate_training_inputs(_joint_batch([False], zero), "action")


def test_vqa_dummy_contract_is_explicitly_unsupervised_and_collates_with_vla():
    from utils.load_training_dataset import VQADataset, custom_collate_fn

    dataset = VQADataset.__new__(VQADataset)
    dataset.max_pad_state_and_action_length = 4
    dataset.action_horizon = 2
    vqa = {
        **dataset.generate_dummy_action_expert_inputs(),
        "input_ids": torch.tensor([1, 2, 0, 0]),
        "attention_mask": torch.tensor([1, 1, 0, 0]),
        "labels": torch.tensor([-100, 2, -100, -100]),
    }
    vla = {
        "observation.state": torch.ones(1, 4),
        "state_mask": torch.ones(1, 4, dtype=torch.bool),
        "action": torch.ones(2, 4),
        "action_mask": torch.ones(2, 4, dtype=torch.bool),
        "action_supervision_available": torch.tensor(True),
        "input_ids": torch.tensor([1, 2, 3, 0]),
        "attention_mask": torch.tensor([1, 1, 1, 0]),
        "labels": torch.tensor([-100, 2, 3, -100]),
    }
    assert not vqa["action_supervision_available"].item()
    assert not vqa["state_mask"].any()
    assert not vqa["action_mask"].any()
    batch = custom_collate_fn([vla, vqa])
    assert batch["action_supervision_available"].tolist() == [True, False]
    assert batch["action_mask"][0].any()
    assert not batch["action_mask"][1].any()


def test_masked_mean_has_zero_value_and_zero_sample_gradient_for_vqa():
    from model.flow_matching_action_head import distributed_masked_mean

    element_loss = torch.tensor([[[1.0]], [[7.0]]], requires_grad=True)
    mask = torch.tensor([[[True]], [[False]]])
    loss = distributed_masked_mean(element_loss, mask)
    loss.backward()
    assert loss.item() == 1.0
    assert element_loss.grad[0].item() == 1.0
    assert element_loss.grad[1].item() == 0.0

    all_vqa = torch.tensor([3.0], requires_grad=True)
    zero_loss = distributed_masked_mean(all_vqa, torch.tensor([False]))
    zero_loss.backward()
    assert zero_loss.item() == 0.0
    assert all_vqa.grad.item() == 0.0


def test_v2_missing_grounding_camera_metadata_normalizes_to_empty_tuple():
    from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from utils.dataset_spec import resolve_dataset_spec

    metadata = LeRobotDatasetMetadata.__new__(LeRobotDatasetMetadata)
    metadata.info = {
        "codebase_version": "v2.0",
        "features": {
            "front": {"dtype": "image", "shape": [3, 4, 4]},
            "observation.state": {"dtype": "float32", "shape": [2]},
            "action": {"dtype": "float32", "shape": [2]},
        },
    }
    metadata.stats = _v2_stats()
    assert metadata.grounding_camera_keys is None
    spec = resolve_dataset_spec(
        "old-v2",
        {
            "dataset_path": "/tmp/old-v2",
            "dataset_type": "vla",
            "use_quantile": True,
            "sample_ratio": 1.0,
        },
        action_horizon=2,
        require_action=True,
        v2_metadata=metadata,
    )
    assert spec.grounding_camera_keys == ()


def test_collate_recomputes_padding_after_sequence_crop():
    from utils.load_training_dataset import custom_collate_fn

    samples = []
    for valid_length in (2, 4):
        attention = torch.tensor([1] * valid_length + [0] * (8 - valid_length))
        samples.append(
            {
                "input_ids": torch.arange(8),
                "attention_mask": attention,
                "labels": torch.full((8,), -100),
                "padding_token_count": torch.tensor(8 - valid_length),
                "padding_token_count_valid": torch.tensor(True),
            }
        )
    batch = custom_collate_fn(samples)
    assert batch["attention_mask"].shape == (2, 5)
    assert batch["padding_token_count"].tolist() == [3, 1]
    assert batch["padding_token_count_valid"].tolist() == [True, True]

    without_precomputed_metric = [
        {
            "input_ids": torch.arange(8),
            "attention_mask": torch.tensor([1, 1, 1, 0, 0, 0, 0, 0]),
            "labels": torch.full((8,), -100),
        }
    ]
    recomputed = custom_collate_fn(without_precomputed_metric)
    assert recomputed["padding_token_count"].tolist() == [1]
    assert recomputed["padding_token_count_valid"].tolist() == [True]


def test_length_audit_marks_only_context_overflow_as_input_truncation():
    from utils.future_difference_audit import FutureDifferenceTokenMeasurer

    class Tokenizer:
        pad_token_id = 0

        @staticmethod
        def encode(text, add_special_tokens=False):
            del add_special_tokens
            return [ord(char) + 1 for char in text]

    class Processor:
        tokenizer = Tokenizer()

        @staticmethod
        def apply_chat_template(messages, tokenize=False, add_generation_prompt=False):
            del tokenize, add_generation_prompt
            pieces = []
            for message in messages:
                pieces.append(f"<{message['role']}>")
                content = message["content"]
                if isinstance(content, str):
                    pieces.append(content)
                else:
                    pieces.extend(
                        "<image>" if item["type"] == "image" else item["text"]
                        for item in content
                    )
            if messages[-1]["role"] == "assistant":
                pieces.append("</assistant>")
            return "".join(pieces)

        def __call__(self, *, text, images, videos, **kwargs):
            del images, videos, kwargs
            ids = torch.tensor([self.tokenizer.encode(text[0])], dtype=torch.long)
            return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    measurer = FutureDifferenceTokenMeasurer(
        Processor(),
        camera_shapes={
            "first_view": (4, 4, 3),
            "second_view": (4, 4, 3),
            "wrist_image": (4, 4, 3),
        },
    )
    full = measurer("move", "target", 10_000, "sample=0")
    context = full["context_tokens"]
    assert measurer("move", "target", context - 1, "sample=0")["input_truncated"]
    equal = measurer("move", "target", context, "sample=0")
    assert not equal["input_truncated"]
    assert equal["target_truncated"]
    target_only = measurer(
        "move",
        "target",
        context + full["original_target_tokens"],
        "sample=0",
    )
    assert not target_only["input_truncated"]
    assert target_only["target_truncated"]
    assert not full["input_truncated"]
    assert not full["target_truncated"]
