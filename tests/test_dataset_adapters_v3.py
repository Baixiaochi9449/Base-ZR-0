import io
import json
import pickle
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from PIL import Image


CANONICAL_KEYS = (
    "Task_temporal",
    "Spatial_motion",
    "Contact_interaction",
    "Object_constraints",
)


class FakeTokenizer:
    pad_token_id = 0
    padding_side = "right"

    @staticmethod
    def encode(text, add_special_tokens=False):
        del add_special_tokens
        return [ord(char) + 1 for char in text]

    @staticmethod
    def decode(ids, skip_special_tokens=False):
        del skip_special_tokens
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return "".join(chr(token - 1) for token in ids if token > 0)


class FakeProcessor:
    def __init__(self):
        self.tokenizer = FakeTokenizer()
        self.messages = []

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        assert not tokenize
        self.messages.append(messages)
        pieces = []
        for message in messages:
            pieces.append(f"<{message['role']}>")
            content = message["content"]
            if isinstance(content, str):
                pieces.append(content)
            else:
                for item in content:
                    pieces.append("<image>" if item["type"] == "image" else item["text"])
        if add_generation_prompt:
            pieces.append("<assistant>")
        elif messages and messages[-1]["role"] == "assistant":
            pieces.append("</assistant>")
        return "".join(pieces)

    def __call__(self, *, text, images, videos, **kwargs):
        del videos, kwargs
        ids = torch.tensor([self.tokenizer.encode(text[0])], dtype=torch.long)
        return {
            "input_ids": ids,
            "attention_mask": torch.ones_like(ids),
            "pixel_values": torch.ones((len(images or []), 3), dtype=torch.float32),
            "image_grid_thw": torch.ones((len(images or []), 3), dtype=torch.long),
        }


def _image_bytes(color):
    image = Image.new("RGB", (4, 3), color)
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def _target(prefix):
    return json.dumps(
        {key: f"{prefix}-{key}" for key in reversed(CANONICAL_KEYS)},
        ensure_ascii=False,
    )


def _write_episode(root: Path, episode: int, task_index: int, length: int):
    relative = Path(f"data/chunk-000/file-{episode:03d}.parquet")
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for frame in range(length):
        rows.append(
            {
                "first_view": {"bytes": _image_bytes("red"), "path": None},
                "second_view": {"bytes": _image_bytes("green"), "path": None},
                "wrist_image": {"bytes": _image_bytes("blue"), "path": None},
                "state": [float(episode * 10 + frame + i) for i in range(7)],
                "actions": [float(episode * 100 + frame + i) for i in range(7)],
                "timestamp": frame / 30.0,
                "frame_index": frame,
                "episode_index": episode,
                "index": episode * 1000 + frame,
                "task_index": task_index,
                "train_data": _target(f"e{episode}f{frame}"),
                "slot_data": "MUST_NOT_LEAK",
            }
        )
    pq.write_table(pa.Table.from_pylist(rows), path)
    return relative.as_posix()


@pytest.fixture
def v3_root(tmp_path):
    root = tmp_path / "v3"
    meta = root / "meta"
    meta.mkdir(parents=True)
    source0 = _write_episode(root, 0, 0, 3)
    source1 = _write_episode(root, 1, 1, 2)
    with (meta / "steps_data_index.pkl").open("wb") as output:
        pickle.dump(
            {
                "steps": [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)],
                "total_steps": 5,
                "num_trajectories": 2,
            },
            output,
        )
    with (meta / "stage05_episode_mapping.jsonl").open("w", encoding="utf-8") as output:
        output.write(json.dumps({"new_episode_index": 0, "old_episode_index": 0, "source_data_uri": source0}) + "\n")
        output.write(json.dumps({"new_episode_index": 1, "old_episode_index": 1, "source_data_uri": source1}) + "\n")
    pq.write_table(
        pa.Table.from_pylist(
            [{"task_index": 0, "task": "task zero"}, {"task_index": 1, "task": "task one"}]
        ),
        meta / "tasks.parquet",
    )
    stats = {
        "statistics": {
            "state": {"q01": [-100.0] * 7, "q99": [100.0] * 7},
            "actions": {"q01": [-1000.0] * 7, "q99": [1000.0] * 7},
        }
    }
    (meta / "stats_gr00t.json").write_text(json.dumps(stats), encoding="utf-8")
    (meta / "info.json").write_text(
        json.dumps(
            {
                "codebase_version": "v3.0-test",
                "total_episodes": 2,
                "total_frames": 5,
                "fps": 30,
                "features": {
                    "first_view": {"dtype": "image", "shape": [3, 4, 3]},
                    "second_view": {"dtype": "image", "shape": [3, 4, 3]},
                    "wrist_image": {"dtype": "image", "shape": [3, 4, 3]},
                    "state": {"dtype": "float32", "shape": [7]},
                    "actions": {"dtype": "float32", "shape": [7]},
                    "train_data": {"dtype": "string", "shape": [1]},
                },
            }
        ),
        encoding="utf-8",
    )
    return root


def _entry(root, ratio=1.0):
    return {
        "dataset_path": str(root),
        "dataset_type": "vla",
        "dataset_adapter": "lerobot_v3_future_difference",
        "target_text_field": "train_data",
        "camera_keys": ["first_view", "second_view", "wrist_image"],
        "state_field": "state",
        "action_field": "actions",
        "stats_path": "meta/stats_gr00t.json",
        "use_quantile": True,
        "sample_ratio": ratio,
    }


def test_registry_default_and_unknown_error_lists_choices():
    from utils.dataset_adapters import DATASET_ADAPTERS, resolve_dataset_adapter_name

    assert set(DATASET_ADAPTERS) == {
        "lerobot_v2", "lerobot_v3_future_difference", "stage05_mixed_pretraining"
    }
    assert resolve_dataset_adapter_name({"dataset_type": "vla"}) == "lerobot_v2"
    with pytest.raises(ValueError, match="lerobot_v2.*lerobot_v3_future_difference"):
        resolve_dataset_adapter_name({"dataset_type": "vla", "dataset_adapter": "bad"})


def test_public_target_token_encoder_matches_tokenizer_boundary():
    from utils.dataset_adapters import encode_future_difference_target_tokens

    target = '{"Task_temporal":"x"}'
    assert encode_future_difference_target_tokens(FakeTokenizer(), target) == [
        ord(char) + 1 for char in target
    ]


def test_train_and_direct_context_use_one_prompt_builder(v3_root):
    from utils.dataset_adapters import (
        LeRobotV3FutureDifferenceDataset,
        prepare_future_difference_direct_inputs,
    )

    processor = FakeProcessor()
    dataset = LeRobotV3FutureDifferenceDataset(
        entry=_entry(v3_root), processor=processor, loss_type="vlm", max_length=2000
    )
    training = dataset[0]
    table = dataset._cache.read(dataset.episode_sources[0], dataset._required_columns())
    row_index = dataset._row_index(table, 0, 0, "sample")
    row = table.slice(row_index, 1).to_pylist()[0]
    images = [
        (key, dataset._decode_image(row[key], "sample", key))
        for key in dataset.camera_keys
    ]

    direct = prepare_future_difference_direct_inputs(
        task="  task zero  ",
        images=images,
        processor=processor,
    )

    context_length = training["context_token_count"].item()
    torch.testing.assert_close(
        training["input_ids"][:context_length], direct["input_ids"]
    )
    assert context_length == direct["context_token_count"].item()
    assert processor.tokenizer.decode(direct["input_ids"]).endswith("<assistant>")
    latest_user = processor.messages[-1][0]["content"]
    assert [item["text"] for item in latest_user if item["type"] == "text"] == [
        "first_view",
        "second_view",
        "wrist_image",
        "<TASK> task zero </TASK>",
    ]
    assert "<\\TASK>" not in repr(latest_user)


def test_canonical_target_is_compact_ordered_and_reports_sample_id():
    from utils.dataset_adapters import canonicalize_future_difference_target

    canonical = canonicalize_future_difference_target(_target("x"), "episode=3 frame=4 sample=9")
    assert list(json.loads(canonical)) == list(CANONICAL_KEYS)
    assert " " not in canonical
    with pytest.raises(ValueError, match="episode=3 frame=4 sample=9"):
        canonicalize_future_difference_target('{"Task_temporal":"x"}', "episode=3 frame=4 sample=9")
    with pytest.raises(ValueError, match="episode=3 frame=4 sample=9"):
        canonicalize_future_difference_target("[]", "episode=3 frame=4 sample=9")


def test_vlm_reads_only_required_columns_and_never_loads_stats(v3_root, monkeypatch):
    from utils import dataset_adapters

    columns = []
    original_read = dataset_adapters.pq.read_table

    def audited_read(path, *args, **kwargs):
        if str(path).endswith(".parquet") and "/data/" in str(path):
            columns.append(tuple(kwargs.get("columns", ())))
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(dataset_adapters.pq, "read_table", audited_read)
    (v3_root / "meta/stats_gr00t.json").unlink()
    dataset = dataset_adapters.LeRobotV3FutureDifferenceDataset(
        entry=_entry(v3_root), processor=FakeProcessor(), loss_type="vlm", max_length=2000
    )
    sample = dataset[0]
    assert "action" not in sample and "observation.state" not in sample
    assert columns and not ({"state", "actions"} & set(columns[-1]))
    message = next(
        message
        for message in reversed(dataset.processor.messages)
        if isinstance(message[0]["content"], list)
        and any(item.get("type") == "image" for item in message[0]["content"])
    )
    assert len(message) == 2
    assert "MUST_NOT_LEAK" not in repr(message[0])
    assert "train_data" not in repr(message[0])
    assert all(item.get("type") != "image" or isinstance(item["image"], Image.Image) for item in message[0]["content"])


def test_action_does_not_read_or_parse_train_data(v3_root, monkeypatch):
    from utils import dataset_adapters

    columns = []
    original_read = dataset_adapters.pq.read_table

    def audited_read(path, *args, **kwargs):
        if "/data/" in str(path):
            columns.append(tuple(kwargs.get("columns", ())))
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(dataset_adapters.pq, "read_table", audited_read)
    dataset = dataset_adapters.LeRobotV3FutureDifferenceDataset(
        entry=_entry(v3_root), processor=FakeProcessor(), loss_type="action", max_length=2000
    )
    sample = dataset[0]
    assert "labels" not in sample
    assert columns and "train_data" not in columns[-1]
    assert sample["observation.state"].shape == (1, 64)
    assert sample["action"].shape == (32, 64)


def test_v3_vlm_accepts_minimal_schema_without_state_action_or_stats(v3_root, monkeypatch):
    from utils import dataset_adapters

    info_path = v3_root / "meta/info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["features"].pop("state")
    info["features"].pop("actions")
    info_path.write_text(json.dumps(info), encoding="utf-8")
    (v3_root / "meta/stats_gr00t.json").unlink()
    selected_columns = []
    original_read = dataset_adapters.pq.read_table

    def audited_read(path, *args, **kwargs):
        if "/data/" in str(path):
            selected_columns.append(tuple(kwargs.get("columns", ())))
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(dataset_adapters.pq, "read_table", audited_read)
    dataset = dataset_adapters.LeRobotV3FutureDifferenceDataset(
        entry=_entry(v3_root),
        processor=FakeProcessor(),
        loss_type="vlm",
        max_length=2000,
    )
    sample = dataset[0]
    assert "labels" in sample
    assert "action" not in sample and "observation.state" not in sample
    assert not ({"state", "actions"} & set(selected_columns[-1]))


def test_v3_action_accepts_schema_without_target(v3_root):
    from utils.dataset_adapters import LeRobotV3FutureDifferenceDataset

    info_path = v3_root / "meta/info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["features"].pop("train_data")
    info_path.write_text(json.dumps(info), encoding="utf-8")
    dataset = LeRobotV3FutureDifferenceDataset(
        entry=_entry(v3_root),
        processor=FakeProcessor(),
        loss_type="action",
        max_length=2000,
    )
    sample = dataset[0]
    assert "labels" not in sample
    assert sample["action"].shape == (32, 64)


def test_action_tail_zero_padding_and_episode_isolation(v3_root):
    from utils.dataset_adapters import LeRobotV3FutureDifferenceDataset

    dataset = LeRobotV3FutureDifferenceDataset(
        entry=_entry(v3_root), processor=FakeProcessor(), loss_type="action", max_length=2000
    )
    sample = dataset[1]
    assert sample["action_mask"].dtype in (torch.bool, torch.long)
    assert sample["action_mask"][:2, :7].all()
    assert not sample["action_mask"][2:].any()
    assert not sample["action_mask"][:, 7:].any()
    assert torch.count_nonzero(sample["action"][2:]) == 0
    assert torch.count_nonzero(sample["action"][:, 7:]) == 0
    # The next published step is episode 1, but the horizon must stop at episode 0.
    assert sample["action"][1, 0] < 0.01


def test_action_chunk_does_not_materialize_whole_image_table(v3_root):
    from utils.dataset_adapters import LeRobotV3FutureDifferenceDataset

    dataset = LeRobotV3FutureDifferenceDataset(
        entry=_entry(v3_root), processor=FakeProcessor(), loss_type="action", max_length=2000
    )
    table = dataset._cache.read(dataset.episode_sources[0], dataset._required_columns())

    class NoWholeTableConversion:
        def __getitem__(self, key):
            return table[key]

        def to_pylist(self):
            raise AssertionError("action chunk must not convert the whole image table")

    result = dataset._action_inputs(
        NoWholeTableConversion(), row_index=0, episode=0, frame=0, sample_id="sample"
    )
    assert result["action_mask"][:3, :7].all()


def test_joint_single_sample_and_collate(v3_root):
    from utils.dataset_adapters import LeRobotV3FutureDifferenceDataset
    from utils.load_training_dataset import custom_collate_fn

    dataset = LeRobotV3FutureDifferenceDataset(
        entry=_entry(v3_root), processor=FakeProcessor(), loss_type="vlm_and_action", max_length=2000
    )
    sample = dataset[0]
    batch = custom_collate_fn([sample, sample])
    assert batch["action"].shape == (2, 32, 64)
    assert batch["labels"].shape[0] == 2
    assert batch["context_token_count"].shape == (2,)
    assert batch["supervised_token_count"].gt(0).all()


def test_collate_mixed_v2_v3_defaults_token_stats_without_breaking_v2():
    from utils.load_training_dataset import custom_collate_fn

    common = {
        "input_ids": torch.tensor([1, 2, 0]),
        "attention_mask": torch.tensor([1, 1, 0]),
        "pixel_values": torch.ones(1, 3),
        "image_grid_thw": torch.ones(1, 3, dtype=torch.long),
        "sub_task_flag": torch.tensor(0),
    }
    v2 = {
        **common,
        "labels": torch.tensor([-100, 2, -100]),
    }
    v3 = {
        **common,
        "context_token_count": torch.tensor(5),
        "supervised_token_count": torch.tensor(2),
        "input_truncated": torch.tensor(False),
    }
    batch = custom_collate_fn([v2, v3])
    torch.testing.assert_close(batch["context_token_count"], torch.tensor([0, 5]))
    torch.testing.assert_close(batch["supervised_token_count"], torch.tensor([0, 2]))
    for key in (
        "json_content_token_count",
        "chat_termination_token_count",
        "target_region_token_count",
    ):
        assert key in batch
        assert f"{key}_valid" in batch
        assert batch[f"{key}_valid"].tolist() == [False, False]
    assert batch["padding_token_count"].tolist() == [1, 1]
    assert batch["padding_token_count_valid"].tolist() == [True, True]
    assert batch["context_token_count_valid"].tolist() == [False, True]
    assert batch["supervised_token_count_valid"].tolist() == [False, True]
    assert batch["input_truncated"].dtype == torch.bool
    torch.testing.assert_close(batch["labels"][0], v2["labels"])
    torch.testing.assert_close(batch["labels"][1], torch.full((3,), -100))

    pure_v3_action = custom_collate_fn([common, common])
    assert "labels" not in pure_v3_action
    for key in (
        "json_content_token_count",
        "chat_termination_token_count",
        "target_region_token_count",
    ):
        assert not pure_v3_action[f"{key}_valid"].any()


@pytest.mark.parametrize(
    "bad_stats",
    [
        {"statistics": {"state": {"q01": [0] * 6, "q99": [1] * 6}, "actions": {"q01": [0] * 7, "q99": [1] * 7}}},
        {"statistics": {"state": {"q01": [0] * 7, "q99": [1] * 7}, "actions": {"q01": [0] * 7}}},
        {"statistics": {"state": {"q01": [0] * 7, "q99": [1] * 7}, "actions": {"q01": [0] * 7, "q99": [0] * 7}}},
        {"statistics": {"state": {"q01": [0] * 7, "q99": [1] * 7}, "actions": {"q01": [0] * 6 + [float("nan")], "q99": [1] * 7}}},
    ],
)
def test_stats_fail_fast(v3_root, bad_stats):
    from utils.dataset_adapters import LeRobotV3FutureDifferenceDataset

    (v3_root / "meta/stats_gr00t.json").write_text(json.dumps(bad_stats), encoding="utf-8")
    with pytest.raises(ValueError, match="stats"):
        LeRobotV3FutureDifferenceDataset(
            entry=_entry(v3_root), processor=FakeProcessor(), loss_type="action", max_length=2000
        )


def test_quantile_roundtrip_uses_exact_seven_dimensions():
    from utils.dataset_adapters import validate_v3_quantile_stats
    from utils.normalization import min_max_denorm, min_max_norm

    raw = torch.tensor([[0.25, -0.5, 1.5, 2.0, -3.0, 0.0, 1.0]])
    stats = {"q01": np.arange(7, dtype=np.float32) - 5, "q99": np.arange(7, dtype=np.float32) + 5}
    validated = validate_v3_quantile_stats({"statistics": {"state": stats, "actions": stats}})
    normalized = min_max_norm(raw, validated["observation.state"], True)
    restored = min_max_denorm(normalized, validated["observation.state"], True)
    torch.testing.assert_close(restored[:, :7], raw, atol=1e-6, rtol=1e-6)


def test_token_stats_and_supervised_decode_are_exact(v3_root):
    from utils.dataset_adapters import LeRobotV3FutureDifferenceDataset

    processor = FakeProcessor()
    dataset = LeRobotV3FutureDifferenceDataset(
        entry=_entry(v3_root), processor=processor, loss_type="vlm", max_length=2000
    )
    sample = dataset[0]
    supervised = sample["labels"][sample["labels"] != -100]
    expected = json.dumps(
        {key: f"e0f0-{key}" for key in CANONICAL_KEYS},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    content_count = sample["json_content_token_count"].item()
    termination_count = sample["chat_termination_token_count"].item()
    assert processor.tokenizer.decode(supervised[:content_count]) == expected
    assert processor.tokenizer.decode(supervised[content_count:]) == "</assistant>"
    assert termination_count == len(FakeTokenizer.encode("</assistant>"))
    assert sample["target_region_token_count"].item() == len(supervised)
    assert sample["original_target_token_count"].item() == content_count
    assert sample["kept_target_token_count"].item() == content_count
    assert sample["supervised_token_count"].item() == len(supervised)
    assert sample["padding_token_count"].item() == len(sample["input_ids"]) - (
        sample["context_token_count"].item() + len(supervised)
    )
    assert sample["truncated_token_count"].item() == 0
    assert not sample["input_truncated"].item()
    assert not sample["target_truncated"].item()


def test_target_truncation_and_all_masked_labels_fail_fast(v3_root, monkeypatch):
    from utils.dataset_adapters import LeRobotV3FutureDifferenceDataset

    with pytest.raises(ValueError, match=r"episode=0 frame=0 sample=0.*target.*max_length=20"):
        LeRobotV3FutureDifferenceDataset(
            entry=_entry(v3_root), processor=FakeProcessor(), loss_type="vlm", max_length=20
        )[0]

    dataset = LeRobotV3FutureDifferenceDataset(
        entry=_entry(v3_root), processor=FakeProcessor(), loss_type="vlm", max_length=2000
    )
    monkeypatch.setattr(dataset, "_target_token_ids", lambda target: [])
    with pytest.raises(ValueError, match=r"episode=0 frame=0 sample=0.*empty target"):
        dataset[0]


def test_assistant_termination_cannot_be_silently_trimmed(v3_root):
    from utils.dataset_adapters import LeRobotV3FutureDifferenceDataset

    full = LeRobotV3FutureDifferenceDataset(
        entry=_entry(v3_root), processor=FakeProcessor(), loss_type="vlm", max_length=2000
    )[0]
    assistant_end = (
        full["context_token_count"].item()
        + full["target_region_token_count"].item()
    )
    exact = LeRobotV3FutureDifferenceDataset(
        entry=_entry(v3_root), processor=FakeProcessor(), loss_type="vlm", max_length=assistant_end
    )[0]
    assert not exact["target_truncated"].item()
    assert exact["padding_token_count"].item() == 0
    with pytest.raises(
        ValueError,
        match=r"target_truncated.*original_target_region=.*termination=.*max_length=",
    ):
        LeRobotV3FutureDifferenceDataset(
            entry=_entry(v3_root), processor=FakeProcessor(), loss_type="vlm", max_length=assistant_end - 1
        )[0]


@pytest.mark.parametrize("action_horizon", [1, 10, 16, 32])
def test_action_horizon_is_external_and_tail_mask_is_exact(v3_root, action_horizon):
    from utils.dataset_adapters import LeRobotV3FutureDifferenceDataset

    dataset = LeRobotV3FutureDifferenceDataset(
        entry=_entry(v3_root),
        processor=FakeProcessor(),
        loss_type="action",
        max_length=2000,
        action_horizon=action_horizon,
    )
    first = dataset[0]
    tail = dataset[2]
    assert first["action"].shape == (action_horizon, 64)
    assert first["action_mask"].shape == (action_horizon, 64)
    assert first["action_mask"][:, :7].sum().item() == min(3, action_horizon) * 7
    assert tail["action_mask"][:, :7].sum().item() == 7
    assert not tail["action_mask"][1:].any()
    assert not tail["action"][:, 7:].any()


@pytest.mark.parametrize("action_horizon", [0, -1, 1.5, True])
def test_action_horizon_must_be_a_positive_integer(v3_root, action_horizon):
    from utils.dataset_adapters import LeRobotV3FutureDifferenceDataset

    with pytest.raises(ValueError, match="action_horizon"):
        LeRobotV3FutureDifferenceDataset(
            entry=_entry(v3_root),
            processor=FakeProcessor(),
            loss_type="action",
            max_length=2000,
            action_horizon=action_horizon,
        )


def test_sampling_is_deterministic_and_index_metadata_is_strict(v3_root):
    from utils.dataset_adapters import LeRobotV3FutureDifferenceDataset

    first = LeRobotV3FutureDifferenceDataset(
        entry=_entry(v3_root, 0.6), processor=FakeProcessor(), loss_type="vlm", max_length=2000
    )
    second = LeRobotV3FutureDifferenceDataset(
        entry=_entry(v3_root, 0.6), processor=FakeProcessor(), loss_type="vlm", max_length=2000
    )
    assert first.subset_indices == second.subset_indices
    assert len(first) == 3

    with (v3_root / "meta/steps_data_index.pkl").open("wb") as output:
        pickle.dump(
            {"steps": [(9, 0)], "total_steps": 1, "num_trajectories": 1},
            output,
        )
    with pytest.raises(ValueError, match="episode 9"):
        LeRobotV3FutureDifferenceDataset(
            entry=_entry(v3_root), processor=FakeProcessor(), loss_type="vlm", max_length=2000
        )


def test_v3_sampling_groups_keep_episode_frames_together_and_bound_cache(v3_root):
    from utils.dataset_adapters import LeRobotV3FutureDifferenceDataset

    dataset = LeRobotV3FutureDifferenceDataset(
        entry=_entry(v3_root), processor=FakeProcessor(), loss_type="vlm", max_length=2000
    )
    groups = dataset.sampling_groups()
    assert dataset._cache.max_files == 1
    assert [
        [dataset.steps[dataset.subset_indices[index]] for index in group]
        for group in groups
    ] == [
        [(0, 0), (0, 1), (0, 2)],
        [(1, 0), (1, 1)],
    ]


def test_steps_trajectory_count_and_mapping_schema_are_strict(v3_root):
    from utils.dataset_adapters import LeRobotV3FutureDifferenceDataset

    steps_path = v3_root / "meta/steps_data_index.pkl"
    with steps_path.open("rb") as source:
        metadata = pickle.load(source)
    metadata["num_trajectories"] = 99
    with steps_path.open("wb") as output:
        pickle.dump(metadata, output)
    with pytest.raises(ValueError, match="num_trajectories"):
        LeRobotV3FutureDifferenceDataset(
            entry=_entry(v3_root), processor=FakeProcessor(), loss_type="vlm", max_length=2000
        )

    metadata["num_trajectories"] = 2
    with steps_path.open("wb") as output:
        pickle.dump(metadata, output)
    mapping_path = v3_root / "meta/stage05_episode_mapping.jsonl"
    rows = [json.loads(line) for line in mapping_path.read_text().splitlines()]
    rows[0].pop("old_episode_index")
    mapping_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    with pytest.raises(ValueError, match="old_episode_index"):
        LeRobotV3FutureDifferenceDataset(
            entry=_entry(v3_root), processor=FakeProcessor(), loss_type="vlm", max_length=2000
        )


def test_builder_loads_fast_only_for_v2_and_supports_mixed(monkeypatch, v3_root):
    from utils import load_training_dataset as loader

    calls = []

    class Auto:
        @staticmethod
        def from_pretrained(path, **kwargs):
            calls.append((path, kwargs))
            return FakeProcessor()

    class FakeV2:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.spec = None

        def __len__(self):
            return 1

        def __getitem__(self, index):
            raise IndexError(index)

    lerobot_calls = []

    class FakeLeRobot:
        def __init__(self, **kwargs):
            lerobot_calls.append(kwargs)
            self.meta = type(
                "Meta",
                (),
                {
                    "camera_keys": ["front"],
                    "grounding_camera_keys": [],
                    "stats": {
                        "observation.state": {"q01": [0] * 7},
                        "action": {"q01": [0] * 7},
                    },
                    "features": {
                        "precomputed_target": {"shape": [1]},
                        "observation.state": {"shape": [7]},
                        "action": {"shape": [7]},
                    },
                    "codebase_version": "v2-test",
                },
            )()

    configs = {
        "v3": _entry(v3_root),
        "v2": {"dataset_path": "/tmp/v2", "dataset_type": "vla", "sample_ratio": 1.0, "use_quantile": True},
    }
    monkeypatch.setattr(loader, "AutoProcessor", Auto)
    monkeypatch.setattr(loader, "StreamingLeRobotSampleDataset", FakeV2)
    monkeypatch.setattr(loader, "LeRobotDataset", FakeLeRobot)
    monkeypatch.setattr(loader, "DATASET2FEATURE", configs)

    only_v3 = loader.build_concat_streaming_dataset(
        ["v3"], "vlm", "fast", 1, 32, None, loss_type="vlm", max_length=2000
    )
    assert len(only_v3.datasets) == 1
    assert [path for path, _ in calls] == ["vlm"]

    calls.clear()
    mixed = loader.build_concat_streaming_dataset(
        ["v2", "v3"], "vlm", "fast", 1, 32, None, loss_type="vlm_and_action", max_length=2000
    )
    assert len(mixed.datasets) == 2
    assert [path for path, _ in calls] == ["vlm", "fast"]
    assert [
        item["resolved_adapter"] for item in mixed.resolved_dataset_manifest["entries"]
    ] == ["lerobot_v2", "lerobot_v3_future_difference"]

    calls.clear()
    configs["v2"]["target_text_field"] = "precomputed_target"
    mixed_ar = loader.build_concat_streaming_dataset(
        ["v2", "v3"], "vlm", "fast", 1, 32, None,
        loss_type="vlm", max_length=2000,
    )
    assert len(mixed_ar.datasets) == 2
    assert [path for path, _ in calls] == ["vlm"]
    assert lerobot_calls[-1]["load_stats"] is False
    assert mixed_ar.datasets[0].kwargs["fast_tokenizer"] is None
    assert mixed_ar.datasets[0].kwargs["requirements"].requires_fast_tokenizer is False


def test_v2_vlm_without_action_independent_target_fails_before_fast(monkeypatch):
    from utils import load_training_dataset as loader

    calls = []

    class Auto:
        @staticmethod
        def from_pretrained(path, **kwargs):
            calls.append(path)
            return FakeProcessor()

    monkeypatch.setattr(loader, "AutoProcessor", Auto)
    monkeypatch.setattr(
        loader,
        "DATASET2FEATURE",
        {
            "v2": {
                "dataset_path": "/tmp/v2",
                "dataset_type": "vla",
                "sample_ratio": 1.0,
                "use_quantile": True,
            }
        },
    )
    with pytest.raises(ValueError, match="action-independent AR-only.*target_text_field"):
        loader.build_concat_streaming_dataset(
            ["v2"], "vlm", "fast", 1, 32, None,
            loss_type="vlm", max_length=2000,
        )
    assert calls == []


def test_objective_requirements_define_v2_action_and_joint_fast_dependencies():
    from utils.dataset_spec import resolve_objective_requirements

    action = resolve_objective_requirements(
        "action", adapter="lerobot_v2", target_text_field=None
    )
    joint = resolve_objective_requirements(
        "vlm_and_action", adapter="lerobot_v2", target_text_field=None
    )
    precomputed_joint = resolve_objective_requirements(
        "vlm_and_action", adapter="lerobot_v2", target_text_field="target"
    )

    assert action.requires_action and action.requires_state and action.requires_stats
    assert not action.requires_target and not action.requires_fast_tokenizer
    assert joint.requires_target and joint.requires_fast_tokenizer
    assert precomputed_joint.requires_target and not precomputed_joint.requires_fast_tokenizer


def test_v2_ar_only_spies_prove_stats_state_action_and_fast_are_not_accessed():
    from utils.dataset_spec import resolve_objective_requirements
    from utils.load_training_dataset import StreamingLeRobotSampleDataset

    selected_columns = []
    reads = []

    class Metadata:
        fps = 30
        camera_keys = ["front"]
        grounding_camera_keys = []
        video_keys = []

        @property
        def stats(self):
            raise AssertionError("AR-only must not open v2 stats")

    class Table:
        column_names = [
            "episode_index",
            "task_index",
            "timestamp",
            "index",
            "front",
            "precomputed_target",
            "observation.state",
            "action",
        ]

        def select_columns(self, columns):
            selected_columns.append(tuple(columns))
            projected = Table()
            projected.column_names = list(columns)
            return projected

    class Dataset:
        meta = Metadata()
        hf_dataset = Table()

        def __len__(self):
            return 1

        def getitem_with_delta_timestamps(self, index, delta_timestamps):
            reads.append((tuple(self.hf_dataset.column_names), dict(delta_timestamps)))
            return {
                "episode_index": 0,
                "task_index": 0,
                "timestamp": 0.0,
                "index": index,
                "task": "pick the object",
                "front": torch.zeros(3, 4, 4),
                "precomputed_target": "target",
            }

    requirements = resolve_objective_requirements(
        "vlm",
        adapter="lerobot_v2",
        target_text_field="precomputed_target",
    )
    dataset = StreamingLeRobotSampleDataset(
        lerobot_dataset=Dataset(),
        use_quantile=True,
        sample_ratio=1.0,
        processor=FakeProcessor(),
        fast_tokenizer=None,
        window_size=1,
        action_horizon=32,
        process_mode="train",
        dataset_id=0,
        max_pad_state_and_action_length=64,
        dataset_entry="legacy-precomputed",
        max_length=2000,
        requirements=requirements,
        target_text_field="precomputed_target",
    )

    sample = dataset[0]
    assert sample is not None and "labels" in sample
    assert selected_columns
    assert "observation.state" not in selected_columns[0]
    assert "action" not in selected_columns[0]
    assert reads == [(selected_columns[0], {})]


def test_v2_tokenizer_default_remains_1200_and_call_compatible():
    from utils.load_training_dataset import tokenize_vision_language_inputs

    class RecordingProcessor(FakeProcessor):
        def __init__(self):
            super().__init__()
            self.call_kwargs = []

        def __call__(self, **kwargs):
            self.call_kwargs.append(kwargs)
            return super().__call__(**kwargs)

    message = [
        {"role": "user", "content": [{"type": "text", "text": "legacy"}]},
        {"role": "assistant", "content": "complete target"},
    ]
    default_processor = RecordingProcessor()
    explicit_processor = RecordingProcessor()
    default = tokenize_vision_language_inputs(message, "train", default_processor)
    explicit = tokenize_vision_language_inputs(message, "train", explicit_processor, max_length=1200)
    torch.testing.assert_close(default["input_ids"], explicit["input_ids"])
    assert len(default["input_ids"]) == 1200
    assert default_processor.call_kwargs[-1]["truncation"] is False
    assert default_processor.call_kwargs[-1]["padding"] is False
    assert "max_length" not in default_processor.call_kwargs[-1]


def test_dataset_sample_ratios_cli_validation_and_builder_override(monkeypatch, v3_root):
    from utils.cli_options import parse_train_options

    options = parse_train_options(
        [
            "--dataset_entries", "one", "two",
            "--dataset_sample_ratios", "0.5", "1.0",
        ]
    )
    assert options.dataset_sample_ratios == [0.5, 1.0]
    with pytest.raises(SystemExit):
        parse_train_options(
            ["--dataset_entries", "one", "two", "--dataset_sample_ratios", "0.5"]
        )
    with pytest.raises(SystemExit):
        parse_train_options(
            ["--dataset_entries", "one", "--dataset_sample_ratios", "nan"]
        )
    with pytest.raises(SystemExit):
        parse_train_options(
            ["--dataset_entries", "one", "--dataset_sample_ratios", "1.1"]
        )
    assert parse_train_options(["--dataset_entries", "one"]).dataset_sample_ratios is None

    from utils import load_training_dataset as loader

    monkeypatch.setattr(loader.AutoProcessor, "from_pretrained", lambda *args, **kwargs: FakeProcessor())
    monkeypatch.setattr(loader, "DATASET2FEATURE", {"v3": _entry(v3_root)})
    dataset = loader.build_concat_streaming_dataset(
        ["v3"], "vlm", "fast", 1, 32, None,
        loss_type="vlm", max_length=2000, dataset_sample_ratios=[0.4],
    )
    assert len(dataset.datasets[0]) == 2


def test_real_v3_release_single_samples_are_read_only_and_model_free():
    from utils.dataset_adapters import LeRobotV3FutureDifferenceDataset
    from utils.load_training_dataset import custom_collate_fn

    root = Path("/opt/data/private/lq/datasets/molmoact_dataset_tabletop-v3_stage05")
    if not root.is_dir():
        pytest.skip("real stage05 dataset is unavailable")
    entry = _entry(root)
    joint_dataset = None
    for loss_type in ("vlm", "action", "vlm_and_action"):
        dataset = LeRobotV3FutureDifferenceDataset(
            entry=entry,
            processor=FakeProcessor(),
            loss_type=loss_type,
            max_length=10000,
        )
        assert len(dataset) == 310743
        sample = dataset[0]
        assert sample["image_grid_thw"].shape[0] == 3
        if loss_type == "action":
            assert "labels" not in sample
        else:
            assert sample["supervised_token_count"].item() > 0
        if loss_type == "vlm":
            assert "action" not in sample
        else:
            assert sample["action"].shape == (32, 64)
            assert sample["action_mask"].dtype == torch.bool
        if loss_type == "vlm_and_action":
            joint_dataset = dataset

    batch = custom_collate_fn([joint_dataset[0], joint_dataset[1]])
    assert batch["input_ids"].shape[0] == 2
    assert batch["labels"].shape[0] == 2
    assert batch["action"].shape == (2, 32, 64)
    assert batch["supervised_token_count"].gt(0).all()


def test_real_qwen_two_samples_supervise_json_im_end_and_newline():
    transformers = pytest.importorskip("transformers")
    root = Path("/opt/data/private/lq/datasets/molmoact_dataset_tabletop-v3_stage05")
    processor_path = Path("/opt/data/private/lq/models/Qwen3-VL-2B-Instruct")
    if not root.is_dir() or not processor_path.is_dir():
        pytest.skip("real stage05 dataset or Qwen processor is unavailable")
    from utils.dataset_adapters import (
        LeRobotV3FutureDifferenceDataset,
        prepare_future_difference_direct_inputs,
    )
    from model.difference_query import build_difference_query_sequence
    from policies.reasoning_vla_policy import ZR0Policy
    from torchvision.transforms import ToTensor

    processor = transformers.AutoProcessor.from_pretrained(processor_path)
    entry = _entry(root)
    entry["dataset_entry"] = "molmoact_tabletop_v3_stage05"
    dataset = LeRobotV3FutureDifferenceDataset(
        entry=entry,
        processor=processor,
        loss_type="vlm",
        max_length=1200,
    )
    for index in (0, 1):
        sample = dataset[index]
        context = sample["context_token_count"].item()
        json_length = sample["json_content_token_count"].item()
        termination = sample["chat_termination_token_count"].item()
        target_length = sample["target_region_token_count"].item()
        assert termination == 2
        assert target_length == json_length + termination
        assert sample["input_ids"][
            context + json_length : context + target_length
        ].tolist() == [151645, 198]
        assert sample["labels"][context : context + target_length].tolist() == sample[
            "input_ids"
        ][context : context + target_length].tolist()
        assert (sample["labels"][:context] == -100).all()
        assert (sample["labels"][context + target_length :] == -100).all()
        json_text = processor.tokenizer.decode(
            sample["input_ids"][context : context + json_length],
            skip_special_tokens=False,
        )
        assert list(json.loads(json_text)) == list(CANONICAL_KEYS)
        assert processor.tokenizer.decode(
            [151645, 198], skip_special_tokens=False
        ) == "<|im_end|>\n"
        sequence = build_difference_query_sequence(
            sample["input_ids"].unsqueeze(0),
            sample["attention_mask"].unsqueeze(0),
            labels=sample["labels"].unsqueeze(0),
            num_queries=32,
            placeholder_token_id=processor.tokenizer.pad_token_id,
        )
        assert sequence.target_lengths.item() == target_length
        first_target = torch.where(sequence.labels[0] != -100)[0][0]
        assert first_target.item() - 1 == sequence.query_positions[0, -1].item()
        query_rows = sequence.attention_mask[
            0, 0, sequence.query_positions[0]
        ]
        assert not query_rows[:, first_target:].any()

        sample_index = dataset.subset_indices[index]
        episode, frame = dataset.steps[sample_index]
        table = dataset._cache.read(
            dataset.episode_sources[episode], dataset._required_columns()
        )
        row_index = dataset._row_index(
            table, episode, frame, f"episode={episode} frame={frame} sample={sample_index}"
        )
        row = table.slice(row_index, 1).to_pylist()[0]
        images = [
            (
                key,
                dataset._decode_image(
                    row[key],
                    f"episode={episode} frame={frame} sample={sample_index}",
                    key,
                ),
            )
            for key in dataset.camera_keys
        ]
        direct = prepare_future_difference_direct_inputs(
            task=dataset.tasks[row["task_index"]],
            images=images,
            processor=processor,
        )
        policy = ZR0Policy.__new__(ZR0Policy)
        policy.dataset_adapter = "lerobot_v3_future_difference"
        policy.dataset_entry = "molmoact_tabletop_v3_stage05"
        policy.camera_keys = list(dataset.camera_keys)
        policy.processor = processor
        policy_direct = policy._prepare_vl_inputs(
            {
                "task": dataset.tasks[row["task_index"]],
                **{key: ToTensor()(image) for key, image in images},
            }
        )
        torch.testing.assert_close(
            sample["input_ids"][:context], direct["input_ids"], rtol=0, atol=0
        )
        torch.testing.assert_close(
            sample["input_ids"][:context],
            policy_direct["input_ids"],
            rtol=0,
            atol=0,
        )
        assert context == 177
        assert sample["input_ids"][169].item() == direct["input_ids"][169].item()
        assert context == direct["context_token_count"].item()
        assert context == policy_direct["context_token_count"].item()


def test_real_v3_processor_resolves_exact_vision_input_contract():
    from transformers import AutoProcessor

    from utils.dataset_adapters import LeRobotV3FutureDifferenceDataset

    root = Path("/opt/data/private/lq/datasets/molmoact_dataset_tabletop-v3_stage05")
    processor_path = Path("/opt/data/private/lq/models/Qwen3-VL-2B-Instruct")
    if not root.is_dir() or not processor_path.is_dir():
        pytest.skip("real stage05 dataset or Qwen processor is unavailable")
    processor = AutoProcessor.from_pretrained(processor_path)
    dataset = LeRobotV3FutureDifferenceDataset(
        entry=_entry(root), processor=processor, loss_type="vlm", max_length=1024
    )

    contract = dataset.spec.vision_input_contract
    assert contract["camera_order"] == [
        "first_view",
        "second_view",
        "wrist_image",
    ]
    assert contract["image_grid_thw_by_camera"] == {
        "first_view": [1, 14, 14],
        "second_view": [1, 14, 14],
        "wrist_image": [1, 14, 14],
    }
    assert contract["visual_tokens_by_camera"] == {
        "first_view": 49,
        "second_view": 49,
        "wrist_image": 49,
    }
    assert contract["total_visual_tokens"] == 147
    assert contract["pixel_values_shape_per_sample"] == [588, 1536]
