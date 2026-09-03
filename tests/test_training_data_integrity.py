import errno
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch


class TextTokenizer:
    pad_token_id = 0
    padding_side = "right"

    @staticmethod
    def encode(text, add_special_tokens=False):
        del add_special_tokens
        return [ord(character) + 1 for character in text]


class TextProcessor:
    def __init__(self):
        self.tokenizer = TextTokenizer()

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        assert not tokenize
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
        if add_generation_prompt:
            pieces.append("<assistant>")
        elif messages and messages[-1]["role"] == "assistant":
            pieces.append("</assistant>\n")
        return "".join(pieces)

    def __call__(
        self,
        *,
        text,
        images,
        videos,
        padding=False,
        max_length=None,
        truncation=False,
        **kwargs,
    ):
        del images, videos, kwargs
        ids = self.tokenizer.encode(text[0])
        if truncation and max_length is not None:
            ids = ids[:max_length]
        attention = [1] * len(ids)
        if padding == "max_length" and max_length is not None:
            padding_length = max_length - len(ids)
            ids.extend([self.tokenizer.pad_token_id] * padding_length)
            attention.extend([0] * padding_length)
        return {
            "input_ids": torch.tensor([ids], dtype=torch.long),
            "attention_mask": torch.tensor([attention], dtype=torch.long),
            "pixel_values": torch.empty((0, 3)),
            "image_grid_thw": torch.empty((0, 3), dtype=torch.long),
        }


def _messages(context, target):
    messages = [{"role": "user", "content": context}]
    if target is not None:
        messages.append({"role": "assistant", "content": target})
    return messages


def _rendered_length(processor, messages):
    rendered = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    return len(processor.tokenizer.encode(rendered))


def test_v2_complete_target_and_termination_fit_exactly():
    from utils.load_training_dataset import tokenize_vision_language_inputs

    processor = TextProcessor()
    target = '{"answer":"complete"}'
    messages = _messages("context", target)
    max_length = _rendered_length(processor, messages)

    result = tokenize_vision_language_inputs(
        messages,
        "train",
        processor,
        max_length=max_length,
        dataset_entry="legacy",
        sample_id="episode=4 frame=9 sample=22",
        target=target,
    )

    supervised = result["input_ids"][result["labels"].ne(-100)]
    expected = processor.tokenizer.encode(target + "</assistant>\n")
    assert supervised.tolist() == expected
    assert result["context_token_count"].item() + len(expected) == max_length
    assert result["chat_termination_token_count"].item() == len("</assistant>\n")
    assert result["padding_token_count"].item() == 0
    for key in (
        "json_content_token_count",
        "chat_termination_token_count",
        "target_region_token_count",
        "padding_token_count",
    ):
        assert result[f"{key}_valid"].item()


@pytest.mark.parametrize(
    ("context", "target", "max_length"),
    [
        ("C" * 300, "short", 100),
        ("short", "word " * 100, 100),
        ("short", "ordinary-word-boundary", 35),
        ("short", '{"nested":{"value":"unfinished"}}', 42),
    ],
)
def test_v2_never_trains_on_a_partial_target(context, target, max_length):
    from utils.load_training_dataset import tokenize_vision_language_inputs

    with pytest.raises(
        ValueError,
        match=(
            r"legacy.*episode=4 frame=9 sample=22.*context_tokens=.*"
            r"original_target_region=.*termination_tokens=.*total_tokens=.*"
            r"max_length=.*kept_target_region="
        ),
    ):
        tokenize_vision_language_inputs(
            _messages(context, target),
            "train",
            TextProcessor(),
            max_length=max_length,
            dataset_entry="legacy",
            sample_id="episode=4 frame=9 sample=22",
            target=target,
        )


def test_v2_rejects_when_only_assistant_termination_would_be_cut():
    from utils.load_training_dataset import tokenize_vision_language_inputs

    processor = TextProcessor()
    target = "complete-content"
    messages = _messages("context", target)
    full_length = _rendered_length(processor, messages)

    with pytest.raises(ValueError, match=r"termination_tokens=13.*kept_target_region="):
        tokenize_vision_language_inputs(
            messages,
            "train",
            processor,
            max_length=full_length - 1,
            dataset_entry="legacy",
            sample_id="episode=4 frame=9 sample=22",
            target=target,
        )


def test_v2_audit_reproduction_1152_to_180_fails_before_model():
    from utils.load_training_dataset import tokenize_vision_language_inputs

    processor = TextProcessor()
    empty_target_length = _rendered_length(processor, _messages("context", ""))
    target = ("multi-token target " * 100)[: 1152 - empty_target_length]
    messages = _messages("context", target)
    assert _rendered_length(processor, messages) == 1152

    with pytest.raises(
        ValueError,
        match=r"total_tokens=1152.*max_length=180.*kept_target_region=",
    ):
        tokenize_vision_language_inputs(
            messages,
            "train",
            processor,
            max_length=180,
            dataset_entry="legacy-audit",
            sample_id="episode=0 frame=0 sample=0",
            target=target,
        )


def test_v2_and_v3_share_target_completeness_semantics():
    from utils.dataset_adapters import tokenize_future_difference_message
    from utils.load_training_dataset import tokenize_vision_language_inputs

    target = "shared-target"
    messages = _messages("shared-context", target)
    full_length = _rendered_length(TextProcessor(), messages)
    calls = (
        lambda: tokenize_vision_language_inputs(
            messages,
            "train",
            TextProcessor(),
            max_length=full_length - 1,
            dataset_entry="legacy",
            sample_id="episode=1 frame=2 sample=3",
            target=target,
        ),
        lambda: tokenize_future_difference_message(
            messages,
            TextProcessor(),
            full_length - 1,
            "episode=1 frame=2 sample=3",
            target,
            dataset_entry="future",
        ),
    )
    for call in calls:
        with pytest.raises(
            ValueError,
            match=r"target_truncated.*termination_tokens=.*kept_target_region=",
        ):
            call()


def test_action_only_marks_target_metrics_invalid():
    from utils.load_training_dataset import tokenize_vision_language_inputs

    result = tokenize_vision_language_inputs(
        _messages("context", None),
        "train",
        TextProcessor(),
        max_length=128,
        has_target=False,
        dataset_entry="legacy",
        sample_id="sample=5",
    )
    for key in (
        "json_content_token_count",
        "chat_termination_token_count",
        "target_region_token_count",
        "supervised_token_count",
    ):
        assert result[key].item() == 0
        assert not result[f"{key}_valid"].item()
    assert result["padding_token_count_valid"].item()


def test_missing_assistant_target_reports_dataset_and_sample_identity():
    from utils.load_training_dataset import tokenize_vision_language_inputs

    with pytest.raises(
        ValueError,
        match=r"dataset_entry=vqa-entry sample=12.*must end with an assistant",
    ):
        tokenize_vision_language_inputs(
            _messages("context", None),
            "train",
            TextProcessor(),
            dataset_entry="vqa-entry",
            sample_id="sample=12",
        )


class _PermanentFailureSource:
    def __init__(self, error):
        self.error = error
        self.calls = []

    def getitem_with_delta_timestamps(self, index, delta_timestamps):
        self.calls.append((index, delta_timestamps))
        raise self.error


def _bare_v2_dataset(source, *, max_transient_retries=2):
    from utils.load_training_dataset import StreamingLeRobotSampleDataset

    dataset = StreamingLeRobotSampleDataset.__new__(StreamingLeRobotSampleDataset)
    dataset.subset_indices = [17, 29]
    dataset.action_horizon = 32
    dataset.window_size = 1
    dataset.dataset_id = 3
    dataset.dataset_entry = "legacy-entry"
    dataset._epoch = 0
    dataset.data_source = source
    dataset.requirements = SimpleNamespace(
        requires_state=False,
        requires_action=False,
        requires_target=True,
    )
    dataset.ecot_supported = False
    dataset.camera_keys = ["front"]
    dataset.grounding_camera_keys = []
    dataset.processor = TextProcessor()
    dataset.process_mode = "train"
    dataset.fast_tokenizer = None
    dataset.max_length = 256
    dataset.target_text_field = "target"
    dataset.max_transient_retries = max_transient_retries
    return dataset


def test_v2_permanent_error_is_not_retried_or_replaced():
    source = _PermanentFailureSource(ValueError("invalid target"))
    dataset = _bare_v2_dataset(source)

    with patch("utils.load_training_dataset.random.choice") as choose:
        with pytest.raises(ValueError, match=r"legacy-entry.*sample=17.*invalid target") as caught:
            dataset[0]

    assert len(source.calls) == 1
    choose.assert_not_called()
    assert isinstance(caught.value.__cause__, ValueError)


def test_v2_target_integrity_error_after_fetch_is_not_retried():
    class Source:
        def __init__(self):
            self.calls = 0

        def getitem_with_delta_timestamps(self, index, delta_timestamps):
            del delta_timestamps
            self.calls += 1
            return {
                "episode_index": torch.tensor(6),
                "frame_index": torch.tensor(8),
                "index": torch.tensor(index),
                "task": "task",
                "front": torch.zeros(3, 4, 4),
                "target": "",
            }

    source = Source()
    dataset = _bare_v2_dataset(source)
    with pytest.raises(
        ValueError,
        match=r"legacy-entry.*episode=6 frame=8 sample=17.*target_text_field",
    ):
        dataset[0]
    assert source.calls == 1


def test_nontransient_file_error_is_not_retried():
    source = _PermanentFailureSource(FileNotFoundError(errno.ENOENT, "missing"))
    dataset = _bare_v2_dataset(source)
    with pytest.raises(ValueError, match=r"legacy-entry.*sample=17.*missing") as caught:
        dataset[0]
    assert len(source.calls) == 1
    assert isinstance(caught.value.__cause__, FileNotFoundError)


def test_v2_transient_io_retries_the_same_index_then_succeeds(monkeypatch):
    source = _PermanentFailureSource(BlockingIOError(errno.EAGAIN, "temporary"))
    dataset = _bare_v2_dataset(source)
    calls = []

    def load_once(index):
        calls.append(index)
        if len(calls) == 1:
            raise BlockingIOError(errno.EAGAIN, "temporary")
        return {"index": index}

    monkeypatch.setattr(dataset, "_get_item_once", load_once, raising=False)
    with patch("utils.load_training_dataset.random.choice") as choose:
        assert dataset[0] == {"index": 17}
    assert calls == [17, 17]
    choose.assert_not_called()


def test_v2_transient_io_retry_exhaustion_preserves_cause(monkeypatch):
    source = _PermanentFailureSource(BlockingIOError(errno.ESTALE, "stale"))
    dataset = _bare_v2_dataset(source, max_transient_retries=2)
    calls = []

    def always_stale(index):
        calls.append(index)
        raise BlockingIOError(errno.ESTALE, "stale")

    monkeypatch.setattr(dataset, "_get_item_once", always_stale, raising=False)
    with pytest.raises(ValueError, match=r"legacy-entry.*sample=17.*after 3 attempts") as caught:
        dataset[0]
    assert calls == [17, 17, 17]
    assert isinstance(caught.value.__cause__, BlockingIOError)


def test_collate_rejects_none_and_empty_batches():
    from utils.load_training_dataset import custom_collate_fn

    sample = {"input_ids": torch.tensor([1]), "attention_mask": torch.tensor([1])}
    with pytest.raises(ValueError, match=r"None.*index 1"):
        custom_collate_fn([sample, None])
    with pytest.raises(ValueError, match="empty batch"):
        custom_collate_fn([])


def test_vqa_permanent_error_is_not_converted_to_none(monkeypatch):
    from utils.load_training_dataset import VQADataset

    dataset = VQADataset.__new__(VQADataset)
    dataset.subset_indices = [11]
    dataset.dataset_entry = "vqa-entry"
    dataset.max_transient_retries = 2
    calls = []

    def fail(index):
        calls.append(index)
        raise ValueError("corrupt json")

    monkeypatch.setattr(dataset, "_get_item_once", fail)
    with pytest.raises(ValueError, match=r"vqa-entry.*sample=11.*corrupt json"):
        dataset[0]
    assert calls == [11]
