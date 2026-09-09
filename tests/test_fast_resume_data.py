"""Resume must skip dataset work without changing rank windows or epoch tails."""

import multiprocessing

import pytest
import torch
from accelerate.data_loader import prepare_data_loader

from utils.load_training_dataset import (
    create_dataloader_for_concat,
    resume_dataloader_at_batch,
    set_dataloader_epoch,
)
from utils.validation_resume import fingerprint


class IndexedGroupedDataset(torch.utils.data.Dataset):
    natural_mix_block_size = 16

    def __init__(self, size, offset):
        self.size, self.offset = size, offset
        self.reads = multiprocessing.Value("q", 0)

    def __len__(self):
        return self.size

    def sampling_group_ranges(self):
        return [(start, min(start + 5, self.size)) for start in range(0, self.size, 5)]

    def __getitem__(self, index):
        with self.reads.get_lock():
            self.reads.value += 1
        index += self.offset
        return {"input_ids": torch.tensor([index, index + 1]),
                "attention_mask": torch.ones(2, dtype=torch.long),
                "labels": torch.tensor([-100, index + 1]),
                "pixel_values": torch.full((2, 3), float(index)),
                "image_grid_thw": torch.tensor([[1, 1, 2]])}


def make_loader(rank, workers):
    datasets = [IndexedGroupedDataset(size, offset) for size, offset in
                zip((63, 19, 27, 14), (0, 100, 200, 300))]
    concat = torch.utils.data.ConcatDataset(datasets)
    raw = create_dataloader_for_concat(concat, batch_size_per_device=3,
        num_processes=4, num_workers=workers, prefetch_factor=2, seed=42)
    loader = prepare_data_loader(raw, device=torch.device("cpu"), num_processes=4,
        process_index=rank, split_batches=False, even_batches=False, rng_types=[])
    return loader, raw.batch_sampler, datasets


@pytest.mark.parametrize("rank", range(4))
@pytest.mark.parametrize("workers", (0, 2))
@pytest.mark.parametrize("epoch", (0, 3))
def test_skipped_inputs_equal_reference_without_historical_reads(rank, workers, epoch):
    control, control_sampler, _ = make_loader(rank, workers)
    set_dataloader_epoch(control, control_sampler, epoch)
    torch.manual_seed(123)
    expected = list(enumerate(control))
    expected_rng = torch.get_rng_state()
    expected_remainder = control.remainder
    resumed, sampler, datasets = make_loader(rank, workers)
    full_length = len(resumed)
    start = 4
    tail = resume_dataloader_at_batch(resumed, sampler, epoch=epoch, batch_idx=start)
    assert len(resumed) == full_length
    assert len(tail) == full_length - start
    assert tail.total_batch_size == control.total_batch_size
    torch.manual_seed(123)
    actual = list(enumerate(tail, start=start))
    assert fingerprint(actual) == fingerprint(expected[start:])
    assert torch.equal(torch.get_rng_state(), expected_rng)
    assert tail.remainder == expected_remainder
    assert tail.end_of_dataloader
    assert sum(dataset.reads.value for dataset in datasets) == sum(
        batch["input_ids"].shape[0] for _, batch in expected[start:])
    assert actual[-1][0] + 1 == full_length
    # Reuse the original loader next epoch; the prefix must not remain skipped.
    set_dataloader_epoch(resumed, sampler, epoch + 1)
    following = list(enumerate(resumed))
    fresh, fresh_sampler, _ = make_loader(rank, workers)
    set_dataloader_epoch(fresh, fresh_sampler, epoch + 1)
    assert fingerprint(following) == fingerprint(list(enumerate(fresh)))


@pytest.mark.parametrize("rank", range(4))
@pytest.mark.parametrize("at_end", (False, True))
def test_partial_tail_and_exact_epoch_boundary(rank, at_end):
    control, control_sampler, _ = make_loader(rank, 0)
    set_dataloader_epoch(control, control_sampler, 2)
    expected = list(control)
    resumed, sampler, datasets = make_loader(rank, 0)
    start = len(resumed) - int(not at_end)
    tail = resume_dataloader_at_batch(resumed, sampler, epoch=2, batch_idx=start)
    actual = list(tail)
    assert fingerprint(actual) == fingerprint(expected[start:])
    assert sum(dataset.reads.value for dataset in datasets) == sum(
        batch["input_ids"].shape[0] for batch in expected[start:])


@pytest.mark.parametrize("cursor", (-1, 12, 1.5, True))
def test_invalid_saved_cursor_is_rejected(cursor):
    loader, sampler, _ = make_loader(0, 0)
    with pytest.raises(ValueError, match="cursor"):
        resume_dataloader_at_batch(loader, sampler, epoch=0, batch_idx=cursor)


def test_zero_cursor_reuses_original_loader():
    loader, sampler, datasets = make_loader(0, 0)
    assert resume_dataloader_at_batch(loader, sampler, epoch=3, batch_idx=0) is loader
    assert sampler.sampler.epoch == 3
    assert not any(dataset.reads.value for dataset in datasets)


def test_resume_window_keeps_absolute_gas_cursor():
    from train_vla import iter_optimizer_step_windows
    loader, sampler, _ = make_loader(2, 0)
    tail = resume_dataloader_at_batch(loader, sampler, epoch=1, batch_idx=4)
    windows = list(iter_optimizer_step_windows(enumerate(tail, start=4), 2))
    assert [[index for index, _ in window] for window in windows] == [[4, 5], [6, 7], [8, 9], [10]]


def test_fast_resume_is_explicit_and_requires_three_stage_contract():
    from utils.cli_options import build_train_parser
    from utils.bounded_validation import validate_update_limits
    parser = build_train_parser()
    assert not parser.get_default("fast_resume_data_skip")
    from types import SimpleNamespace
    with pytest.raises(ValueError, match="three-stage"):
        validate_update_limits(SimpleNamespace(fast_resume_data_skip=True))


@pytest.mark.parametrize("with_slots", (False, True))
def test_frozen_data_contract_accepts_only_reviewed_stage05_wrapper(with_slots):
    from types import SimpleNamespace
    from utils.load_training_dataset import validate_fast_resume_datasets
    from utils.stage05_dataset import Stage05MixedPretrainingDataset
    from utils.slot_supervision import SlotSupervisedDataset
    base = object.__new__(Stage05MixedPretrainingDataset)
    base.frozen_index = True
    dataset = base
    if with_slots:
        dataset = object.__new__(SlotSupervisedDataset)
        dataset.dataset = base
    validate_fast_resume_datasets(SimpleNamespace(datasets=[dataset]))
    base.frozen_index = False
    with pytest.raises(ValueError, match="deterministic frozen"):
        validate_fast_resume_datasets(SimpleNamespace(datasets=[dataset]))
    with pytest.raises(ValueError, match="deterministic frozen"):
        validate_fast_resume_datasets(SimpleNamespace(datasets=[SimpleNamespace(dataset=base)]))
