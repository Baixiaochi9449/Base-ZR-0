#!/usr/bin/env python3
"""Fresh-process continuation of real production sampling and bounded CPU updates."""

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]

import torch
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.utils import set_seed

from model.reasoning_vla_model import ZR0Model
from scripts.verify_auxiliary_production import MODEL, SLOTS, launch_options
from utils.slot_config import SlotConfig
from utils.optical_flow_config import OpticalFlowConfig
from utils.slot_routing import load_slot_supervision
from utils.load_training_dataset import build_concat_streaming_dataset, create_dataloader_for_concat, set_dataloader_epoch
from utils.dataset_manifest import validate_resume_manifest, write_resolved_dataset_manifest
from utils.dataset_seen_tracker import DatasetSeenTracker
from utils.training_checkpoint import (checkpoint_model_optimizer_scheduler, resume_model_optimizer_scheduler,
    restore_stage_runtime, restore_rng_state)
from train_vla import run_optimizer_step_window, advance_global_step, iter_optimizer_step_windows, should_skip_resumed_batch
from test_query_ar_joint_checkpoint import _TinyEngine


def fingerprint(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().contiguous()
        return {"shape": list(value.shape), "dtype": str(value.dtype),
            "sha256": hashlib.sha256(value.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()}
    if isinstance(value, dict):
        return {str(key): fingerprint(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (list, tuple)):
        return [fingerprint(item) for item in value]
    return value


def describe(batch):
    identities = list(zip(*(batch[key].tolist() for key in ("dataset_id", "episode_id", "frame_id"))))
    return {"identities": identities, "tensors": fingerprint(batch)}


def assert_equal(left, right):
    from torch.utils._pytree import tree_flatten
    a, sa = tree_flatten(left)
    b, sb = tree_flatten(right)
    assert sa == sb
    for x, y in zip(a, b):
        if isinstance(x, torch.Tensor):
            torch.testing.assert_close(x, y, rtol=0, atol=0)
        else:
            assert x == y, (x, y)


def worker(args):
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    set_seed(args.seed)
    accelerator = Accelerator(cpu=True, gradient_accumulation_steps=args.gas,
        dataloader_config=DataLoaderConfiguration(even_batches=False))
    checkpoint = args.output / "latest-model-optimizer-lr"
    source = checkpoint if args.worker == "resume" else args.source / (args.stage + "_" + args.mode) / "checkpoint"
    options = launch_options(args.stage, args.mode, source, args.output)
    slot = SlotConfig(**{key: getattr(options, key) for key in SlotConfig.__dataclass_fields__})
    flow = OpticalFlowConfig(**{key: getattr(options, key) for key in OpticalFlowConfig.__dataclass_fields__})
    reader = load_slot_supervision(SLOTS) if slot.enabled else None
    dataset = build_concat_streaming_dataset(options.dataset_entries, MODEL, None, options.window_size,
        options.action_horizon, None, loss_type=options.loss_type, max_length=1024, slot_config=slot,
        slot_reader=reader, optical_flow_config=flow, aux_dataset_config=options.aux_dataset_config)
    loader = create_dataloader_for_concat(dataset, batch_size_per_device=args.microbatch, num_workers=args.workers,
        prefetch_factor=args.prefetch, seed=args.seed)
    epoch_sampler = loader.batch_sampler
    model = ZR0Model.from_pretrained(source, training_stage=args.stage, resume_training=True,
        tune_vlm=args.stage == "stage3_joint", tune_action_expert=args.stage == "stage3_joint")
    model.resolved_dataset_manifest = dataset.resolved_dataset_manifest
    validate_resume_manifest(model.resolved_dataset_manifest, source)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.001)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
    model, optimizer, loader, scheduler = accelerator.prepare(model, optimizer, loader, scheduler)
    bare = accelerator.unwrap_model(model)
    # CPU uses the existing native checkpoint test adapter; sampler, runtime,
    # checkpoint orchestration, model export, loss and update are production APIs.
    engine = _TinyEngine(bare, optimizer)
    model.save_checkpoint, model.load_checkpoint = engine.save_checkpoint, engine.load_checkpoint
    bare.training_sampler_contract = {"seed": args.seed, "batch_size": args.microbatch,
        "gradient_accumulation_steps": args.gas, "dataloader_length": len(loader)}
    bare.training_data_cursor = {"epoch": args.epoch, "batch_idx": 0}
    tracker = DatasetSeenTracker(dataset, accelerator, flow_config=flow,
        resume_directory=checkpoint if args.worker == "resume" else None)
    step, resume_epoch, resume_batch = 0, args.epoch, 0
    if args.worker == "resume":
        step = resume_model_optimizer_scheduler(model, str(args.output), scheduler)
        resume_epoch, resume_batch = restore_stage_runtime(checkpoint, bare, accelerator, step)
        expected = torch.load(args.output / "expected.pt", map_location="cpu", weights_only=False)
        assert_equal(expected["saved_optimizer"], optimizer.state_dict())
        assert_equal(expected["saved_scheduler"], scheduler.state_dict())
    for source_dataset in dataset.datasets:
        source_dataset.set_epoch(args.epoch)
    set_dataloader_epoch(loader, epoch_sampler, args.epoch)
    model.train()

    def batches():
        for index, batch in enumerate(loader):
            if args.worker == "resume" and should_skip_resumed_batch(epoch=args.epoch, batch_idx=index,
                    resume_epoch=resume_epoch, resume_batch_idx=resume_batch):
                continue
            yield index, batch

    records, saved_optimizer, saved_scheduler = [], None, None
    iterator = iter(iter_optimizer_step_windows(batches(), args.gas))
    try:
        for _ in range(2 if args.worker == "resume" else 3):
            indexed = next(iterator)
            assert epoch_sampler.sampler.epoch == args.epoch, ("prepared loader changed sampler epoch", epoch_sampler.sampler.epoch, args.epoch)
            if getattr(bare, "pending_resume_rng", None) is not None:
                restore_rng_state(bare.pending_resume_rng)
                bare.pending_resume_rng = None
            values = [batch for _, batch in indexed]
            descriptions = [describe(batch) for batch in values]
            tracker.update(values)
            with torch.autocast("cpu", dtype=torch.bfloat16):
                metrics = run_optimizer_step_window(model=model, batches=values, accelerator=accelerator,
                    optimizer=optimizer, lr_scheduler=scheduler, training_progress=0., loss_type=bare.loss_type,
                    vlm_loss_weight=0. if bare.loss_type == "aux" else 1.,
                    action_expert_loss_weight=0. if bare.loss_type == "aux" else 1., next_global_step=step+1,
                    training_stage=args.stage, slot_config=slot, optical_flow_config=flow)
            step = advance_global_step(step, metrics, accelerator)
            assert metrics["optimizer_update_applied"], "selected production window lacks active supervision"
            bare.training_data_cursor = {"epoch": args.epoch, "batch_idx": indexed[-1][0] + 1}
            record = {"step": step, "cursor": dict(bare.training_data_cursor), "batches": descriptions,
                "loss": float(metrics["loss"]), "counts": {key: float(value) for key, value in metrics.items() if key.endswith("count")},
                "parameters": fingerprint(bare.state_dict()), "optimizer": fingerprint(optimizer.state_dict()),
                "scheduler": fingerprint(scheduler.state_dict())}
            records.append(record)
            print(json.dumps({key: record[key] for key in ("step", "cursor", "loss", "counts")}), flush=True)
            if args.worker == "control" and step == 1:
                checkpoint_model_optimizer_scheduler(model, str(args.output), step, scheduler, accelerator)
                tracker.save(checkpoint, epoch=args.epoch, global_step=step)
                saved_optimizer, saved_scheduler = copy.deepcopy(optimizer.state_dict()), copy.deepcopy(scheduler.state_dict())
            if args.worker == "resume":
                assert_equal(record, expected["records"][len(records)])
    finally:
        iterator.close()
    result = {"worker": args.worker, "pid": os.getpid(), "stage": args.stage, "mode": args.mode,
        "seed": args.seed, "epoch": args.epoch, "microbatch": args.microbatch, "gas": args.gas,
        "workers": args.workers, "prefetch": args.prefetch if args.workers else None,
        "persistent_workers": False, "dataloader_length": len(loader), "dataset_lengths": [len(ds) for ds in dataset.datasets],
        "sampler": type(epoch_sampler).__name__, "shuffle": "natural frame mixture and per-source grouped shuffle",
        "comparison_rtol": 0, "comparison_atol": 0, "scope": "CPU BF16 real dataloader; native AdamW checkpoint adapter, not ZeRO-2",
        "records": records, "seen": tracker.manifest(epoch=args.epoch, global_step=step)}
    if args.worker == "control":
        torch.save({"records": records, "saved_optimizer": saved_optimizer, "saved_scheduler": saved_scheduler,
            "seen": result["seen"]}, args.output / "expected.pt")
        write_resolved_dataset_manifest(args.output, dataset.resolved_dataset_manifest)
    else:
        assert_equal(result["seen"], expected["seen"])
        result["fresh_process_next_data_and_update_exact"] = True
    (args.output / (args.worker + ".json")).write_text(json.dumps(result, indent=2) + "\n")
    for ds in dataset.datasets:
        base = getattr(ds, "dataset", ds)
        if base.flow_reader is not None:
            base.flow_reader.close()
    accelerator.end_training()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--stage", choices=["stage2_aux", "stage3_joint"], default="stage3_joint")
    parser.add_argument("--mode", choices=["slot", "flow", "both", "none"], default="both")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--prefetch", type=int, default=3)
    parser.add_argument("--microbatch", type=int, default=4)
    parser.add_argument("--gas", type=int, default=2)
    parser.add_argument("--epoch", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--worker", choices=["control", "resume"])
    args = parser.parse_args()
    if args.worker:
        worker(args)
        return
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "experiment.md").write_text(
        "# Production Sampler Continuation\n\n" + json.dumps(vars(args), default=str, indent=2) + "\n\n"
        "Finite CPU BF16 diagnostic, no W&B/formal training. Full model/data/loss contract: "
        "docs/experiments/structured_slots_review4/experiment.md. Production sampler with no subset or fixed sample replay. "
        "One saved window and two subsequent updates; restart in a separate process. Exact tensor, optimizer, scheduler and seen-state comparison.\n\n"
        + "Command: " + __import__("shlex").join([sys.executable, *sys.argv]) + "\n")
    for phase in ("control", "resume"):
        with (args.output / (phase + ".log")).open("w") as log:
            subprocess.run([sys.executable, *sys.argv, "--worker", phase], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    with (args.output / "experiment.md").open("a") as report:
        report.write("\nBoth fresh processes succeeded. All next-batch tensors/masks, parameters, optimizer, scheduler and accounting matched exactly (rtol=atol=0).\n")
    print(json.dumps({"output": str(args.output), "fresh_process_resume_exact": True}), flush=True)


if __name__ == "__main__":
    main()
