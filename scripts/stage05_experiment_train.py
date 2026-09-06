#!/usr/bin/env python3
"""Select an isolated registry, then invoke the unchanged production trainer."""

import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "lerobot"))

import yaml


class ProbeBatchSampler:
    """Fixed real sample fixture; production still owns every optimizer update."""

    def __init__(self, original, indices, gas):
        self.original = original
        self.indices = indices
        self.batch_size = original.batch_size
        self.drop_last = False
        self.gas = gas

    def __len__(self):
        return len(self.original)

    def set_epoch(self, epoch):
        self.original.set_epoch(epoch)

    def __iter__(self):
        for batch in range(3 * self.gas * self.original.num_processes):
            yield [self.indices[(batch * self.batch_size + offset) % len(self.indices)]
                   for offset in range(self.batch_size)]


def main():
    registry_path = Path(os.environ["ZR0_DATASET_REGISTRY"])
    import utils.constants
    utils.constants.DATASET2FEATURE = yaml.safe_load(registry_path.read_text())
    import train_vla
    options = train_vla.parse_option()
    fixture_path = os.environ.get("ZR0_PROBE_SAMPLES")
    if fixture_path:
        if options.max_train_steps not in (2, 3):
            raise ValueError("representative probe is limited to fresh step 2 or resume step 3")
        from torch.utils.data import DataLoader
        from utils.load_training_dataset import EpochGroupedDistributedBatchSampler, custom_collate_fn
        fixture = json.loads(Path(fixture_path).read_text())

        def create_probe_loader(concat_dataset, batch_size_per_device, num_processes=1,
                                num_workers=0, prefetch_factor=2, seed=42):
            import numpy as np
            indices = []
            offset = 0
            for dataset in concat_dataset.datasets:
                for selected in fixture:
                    if selected["dataset_entry"] != dataset.spec.dataset_entry:
                        continue
                    position = int(np.searchsorted(dataset.indices, selected["global_index"]))
                    if position >= len(dataset) or int(dataset.indices[position]) != selected["global_index"]:
                        raise ValueError(f"probe sample is no longer eligible: {selected}")
                    indices.append(offset + position)
                offset += len(dataset)
            if len(indices) != len(fixture):
                raise ValueError("probe sample entries do not match the resolved datasets")
            original = EpochGroupedDistributedBatchSampler(concat_dataset, batch_size_per_device, num_processes, seed)
            sampler = ProbeBatchSampler(original, indices, options.gradient_accumulation_steps)
            kwargs = {"num_workers": num_workers, "pin_memory": True, "collate_fn": custom_collate_fn}
            if num_workers:
                kwargs["prefetch_factor"] = prefetch_factor
            return DataLoader(concat_dataset, batch_sampler=sampler, **kwargs)

        train_vla.create_dataloader_for_concat = create_probe_loader
    train_vla.train(options)


if __name__ == "__main__":
    main()
