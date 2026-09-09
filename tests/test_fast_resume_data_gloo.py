"""Exercise fast data resume inside real CPU distributed processes."""

import os
from pathlib import Path
import subprocess
import sys

import torch


def worker(output):
    from accelerate import Accelerator, DataLoaderConfiguration
    from test_fast_resume_data import IndexedGroupedDataset
    from utils.load_training_dataset import create_dataloader_for_concat, resume_dataloader_at_batch, set_dataloader_epoch
    from utils.validation_resume import fingerprint
    accelerator = Accelerator(cpu=True, gradient_accumulation_steps=2,
        dataloader_config=DataLoaderConfiguration(even_batches=False))

    def create():
        datasets = [IndexedGroupedDataset(size, offset) for size, offset in
                    zip((63, 19, 27, 14), (0, 100, 200, 300))]
        raw = create_dataloader_for_concat(torch.utils.data.ConcatDataset(datasets),
            3, num_processes=accelerator.num_processes, num_workers=2, prefetch_factor=2, seed=42)
        return accelerator.prepare_data_loader(raw), raw.batch_sampler, datasets

    control, sampler, _ = create()
    set_dataloader_epoch(control, sampler, 3)
    expected = list(enumerate(control))
    resumed, sampler, datasets = create()
    tail = resume_dataloader_at_batch(resumed, sampler, epoch=3, batch_idx=4)
    actual = list(enumerate(tail, start=4))
    assert fingerprint(actual) == fingerprint(expected[4:])
    assert tail.remainder == control.remainder
    assert sum(d.reads.value for d in datasets) == sum(batch['input_ids'].shape[0] for _, batch in actual)
    assert sampler.sampler.epoch == 3
    assert not torch.cuda.is_initialized()
    torch.save(dict(rank=accelerator.process_index, world_size=accelerator.num_processes,
        input_tensors_exact=True, historical_data_reads=0, epoch=3,
        first_batch_idx=actual[0][0], final_batch_idx=actual[-1][0],
        cursor_after=actual[-1][0] + 1, full_epoch_batches=len(resumed),
        optimizer_updates=0), output / f'rank{accelerator.process_index}.pt')
    accelerator.wait_for_everyone()
    accelerator.end_training()


def test_fast_resume_with_two_real_ranks(tmp_path):
    root = Path(__file__).resolve().parents[1]
    environment = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'PYTHONNOUSERSITE': '1',
        'PYTHONPATH': f'{root}:{root / "lerobot"}:{root / "tests"}', 'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2'}
    result = subprocess.run([sys.executable, '-m', 'torch.distributed.run', '--standalone',
        '--nproc_per_node=2', str(Path(__file__).resolve()), str(tmp_path)],
        cwd=root, env=environment, capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
    for rank in range(2):
        state = torch.load(tmp_path / f'rank{rank}.pt', map_location='cpu', weights_only=True)
        assert state['rank'] == rank and state['world_size'] == 2
        assert state['input_tensors_exact'] and state['historical_data_reads'] == 0
        assert state['first_batch_idx'] == 4 and state['cursor_after'] == state['full_epoch_batches']
        assert state['optimizer_updates'] == 0


if __name__ == '__main__':
    worker(Path(sys.argv[1]))
