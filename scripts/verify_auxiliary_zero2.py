#!/usr/bin/env python3
"""Two bounded real-mixture updates and a production ZeRO-2 runtime restore."""

import argparse
import copy
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from accelerate import Accelerator, DeepSpeedPlugin
from model.reasoning_vla_model import ZR0Model
from utils.load_training_dataset import custom_collate_fn
from utils.optical_flow_checkpoint import module_checksum
from utils.training_checkpoint import checkpoint_model_optimizer_scheduler, resume_model_optimizer_scheduler, restore_stage_runtime, restore_rng_state
from utils.dataset_manifest import load_resolved_dataset_manifest, validate_resume_manifest
from utils.dataset_seen_tracker import DatasetSeenTracker
from train_vla import run_optimizer_step_window, advance_global_step


def move(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: move(item, device) for key, item in value.items()}
    return value


class AccountingSource:
    def __init__(self, row, contract):
        self.spec = SimpleNamespace(dataset_entry=row["dataset_entry"], auxiliary_contract=contract)
        self.manifest = {"counts": {"source_frames": row["source_frames"]}}
        self.unfiltered_length, self.length = row["unfiltered_length"], row["filtered_length"]

    def __len__(self):
        return self.length


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=["stage2_aux", "stage3_joint"], required=True)
    parser.add_argument("--resume-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    world = int(os.environ.get("WORLD_SIZE", "1"))
    gas = 4 // world
    config = {"train_micro_batch_size_per_gpu": 1, "train_batch_size": 4,
              "gradient_accumulation_steps": gas, "zero_optimization": {"stage": 2},
              "bf16": {"enabled": True}, "gradient_clipping": 1., "zero_allow_untested_optimizer": True}
    accelerator = Accelerator(mixed_precision="bf16", gradient_accumulation_steps=gas,
                              deepspeed_plugin=DeepSpeedPlugin(hf_ds_config=config))
    torch.manual_seed(42)
    checkpoint = args.output / "latest-model-optimizer-lr" if args.resume_only else args.source / "checkpoint"
    model = ZR0Model.from_pretrained(checkpoint, training_stage=args.stage, resume_training=True,
        tune_vlm=args.stage == "stage3_joint", tune_action_expert=args.stage == "stage3_joint")
    manifest = load_resolved_dataset_manifest(args.source)
    model.resolved_dataset_manifest = manifest
    validate_resume_manifest(manifest, args.source)
    samples = torch.load(args.source / "samples.pt", weights_only=False)
    local = samples[accelerator.process_index::world]
    batches = [move(custom_collate_fn([sample]), accelerator.device) for sample in local]
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
    unwrapped = accelerator.unwrap_model(model)
    source_result = json.loads((args.source / "result.json").read_text())
    accounting = SimpleNamespace(datasets=[AccountingSource(row, entry["auxiliary_contract"])
        for row, entry in zip(source_result["seen"]["datasets"], manifest["entries"])])
    tracker = DatasetSeenTracker(accounting, accelerator, flow_config=unwrapped.optical_flow_config,
                                 resume_directory=checkpoint if args.resume_only else None)
    tracker.update(batches)
    if accelerator.is_main_process:
        expected_seen = 2 if args.resume_only else 1
        assert tracker.seen.tolist() == [expected_seen] * 4
        assert tracker.aux_counts[:, tracker.aux_names.index("flow_eligible")].tolist() == [expected_seen] * 4
    tracker.save(args.output, epoch=0, global_step=0)
    unwrapped.training_data_cursor = {"epoch": 0, "batch_idx": gas}
    unwrapped.training_sampler_contract = {"world_size": world, "gas": gas, "seed": 42, "manifest": manifest}
    before = module_checksum(unwrapped.backbone.model)
    query_before = module_checksum(unwrapped.backbone.difference_query)
    heads = {name: module for name, module in (("slot", unwrapped.slot_aux),
        ("flow", unwrapped.optical_flow_aux), ("expert", unwrapped.action_expert)) if module is not None}
    head_checksums = {name: module_checksum(module) for name, module in heads.items()}
    gradients = []
    adam = model.optimizer.optimizer
    adam.register_step_pre_hook(lambda opt, _a, _k: gradients.append(
        [parameter.grad.detach().cpu().clone() for group in opt.param_groups for parameter in group["params"]]))
    def run(step, values=batches):
        return run_optimizer_step_window(model=model, batches=values, accelerator=accelerator,
            optimizer=optimizer, lr_scheduler=scheduler, training_progress=0., loss_type=unwrapped.loss_type,
            vlm_loss_weight=0. if args.stage == "stage2_aux" else 1.,
            action_expert_loss_weight=0. if args.stage == "stage2_aux" else 1., next_global_step=step,
            training_stage=args.stage, slot_config=unwrapped.slot_config, optical_flow_config=unwrapped.optical_flow_config)
    if args.resume_only:
        step = resume_model_optimizer_scheduler(model, str(args.output), scheduler)
        assert step == 1
        restore_stage_runtime(checkpoint, unwrapped, accelerator, step)
        restore_rng_state(unwrapped.pending_resume_rng)
        expected = torch.load(args.output / f"expected-step2-rank{accelerator.process_index}.pt", weights_only=True)
        resumed = run(2)
        assert resumed["optimizer_update_applied"]
        for name, parameter in unwrapped.named_parameters():
            torch.testing.assert_close(parameter.detach().cpu(), expected["parameters"][name], rtol=0, atol=0)
        assert scheduler.state_dict() == expected["scheduler"]
        if accelerator.is_main_process:
            assert tracker.manifest(epoch=0, global_step=2) == expected["data_seen"]
        tracker.save(args.output, epoch=0, global_step=2)
        result = {"world_size": world, "rank": accelerator.process_index, "stage": args.stage,
                  "fresh_process_model_optimizer_scheduler_rng_resume_exact": True, "data_seen_resume_exact": True,
                  "loss": float(resumed["loss"])}
        print(json.dumps(result), flush=True)
        if accelerator.is_main_process:
            (args.output / "resume-new-process.json").write_text(json.dumps(result, indent=2) + "\n")
        accelerator.end_training()
        return
    first = run(1)
    assert first["optimizer_update_applied"]
    assert query_before != module_checksum(unwrapped.backbone.difference_query)
    assert (before != module_checksum(unwrapped.backbone.model)) == (args.stage == "stage3_joint")
    changed_heads = {name: head_checksums[name] != module_checksum(module) for name, module in heads.items()}
    assert all(changed_heads.values())
    checkpoint_model_optimizer_scheduler(model, str(args.output), 1, scheduler, accelerator)
    tracker.save(args.output / "latest-model-optimizer-lr", epoch=0, global_step=1)
    saved = module_checksum(unwrapped)
    master = [value.detach().cpu().clone() for value in model.optimizer.single_partition_of_fp32_groups]
    optimizer_saved = copy.deepcopy(adam.state_dict())
    tracker.update(batches)
    second = run(2)
    expected = module_checksum(unwrapped)
    expected_parameters = {name: parameter.detach().cpu().clone() for name, parameter in unwrapped.named_parameters()}
    torch.save({"parameters": expected_parameters, "scheduler": scheduler.state_dict(),
                "data_seen": tracker.manifest(epoch=0, global_step=2)},
               args.output / f"expected-step2-rank{accelerator.process_index}.pt")
    step = resume_model_optimizer_scheduler(model, str(args.output), scheduler)
    print(json.dumps({"master_restore_max_diff": [float((old - value.detach().cpu()).abs().max())
        for old, value in zip(master, model.optimizer.single_partition_of_fp32_groups)],
        "optimizer_restore_max_diff": {str(key): {field: float((value - adam.state_dict()["state"][key][field]).abs().max())
            for field, value in state.items() if isinstance(value, torch.Tensor)}
            for key, state in optimizer_saved["state"].items()}}), flush=True)
    assert step == 1 and module_checksum(unwrapped) == saved
    restore_stage_runtime(args.output / "latest-model-optimizer-lr", unwrapped, accelerator, step)
    restore_rng_state(unwrapped.pending_resume_rng)
    resumed = run(2)
    differences = {name: float((parameter.detach().cpu().float() - expected_parameters[name].float()).abs().max())
        for name, parameter in unwrapped.named_parameters() if not torch.equal(parameter.detach().cpu(), expected_parameters[name])}
    print(json.dumps({"resume_differences": differences, "second_loss": float(second["loss"]),
                      "resumed_loss": float(resumed["loss"]), "scheduler": scheduler.state_dict(),
                      "gradient_diff": [float((a - b).abs().max()) for a, b in zip(gradients[-2], gradients[-1])]}), flush=True)
    assert resumed["optimizer_update_applied"] and module_checksum(unwrapped) == expected
    if args.stage == "stage2_aux":
        empty = []
        for batch in batches:
            item = dict(batch)
            for key, value in batch.items():
                if key.startswith("slot_") and key.endswith("_mask"):
                    item[key] = torch.zeros_like(value)
            item["flow_supervision_available"] = torch.zeros_like(batch["flow_supervision_available"])
            empty.append(item)
        if world > 1:
            mixed = list(empty)
            if accelerator.process_index == 0:
                mixed[0] = batches[0]
            query = module_checksum(unwrapped.backbone.difference_query)
            partial = run(3, mixed)
            assert partial["optimizer_update_applied"] and partial["flow_eligible_samples"] == 1
            assert query != module_checksum(unwrapped.backbone.difference_query)
            if unwrapped.slot_config.enabled:
                from utils.slot_labels import task_validity
                local_valid = task_validity(mixed[0], device=accelerator.device)
                for q, mask in local_valid.items():
                    expected_count = accelerator.reduce(mask.sum(), reduction="sum")
                    assert partial[f"slot_{q}_valid_count"] == expected_count
            print(json.dumps({"empty_rank_and_trailing_empty_microbatch": True, "flow_global_count": 1}), flush=True)
        checksum, scheduler_state = module_checksum(unwrapped), copy.deepcopy(scheduler.state_dict())
        skipped = run(4, empty)
        assert not skipped["optimizer_update_applied"] and module_checksum(unwrapped) == checksum and scheduler.state_dict() == scheduler_state
        current_step = 3 if world > 1 else 2
        assert advance_global_step(current_step, skipped, accelerator) == current_step
    result = {"world_size": world, "rank": accelerator.process_index, "stage": args.stage,
              "scope": "real four-dataset images/labels with tiny Qwen/Expert and production heads/loss/ZeRO2",
              "global_batch_size": 4, "GAS": gas, "updates": 2 + int(args.stage == "stage2_aux" and world > 1), "resumed_update_exact": True,
              "first_loss": float(first["loss"]), "second_loss": float(second["loss"]), "updated_heads": changed_heads}
    print(json.dumps(result), flush=True)
    if accelerator.is_main_process:
        (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    accelerator.end_training()


if __name__ == "__main__":
    main()
