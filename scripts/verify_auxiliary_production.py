#!/usr/bin/env python3
"""Bounded real-data diagnostics through the production loader, model and optimizer."""

import argparse
import copy
import hashlib
import json
import os
import shlex
import subprocess
import sys
from contextlib import ExitStack
from unittest.mock import patch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]

import torch
from safetensors.torch import load_file
import numpy as np
from accelerate import Accelerator
from transformers import AutoConfig, AutoProcessor, Qwen3VLForConditionalGeneration

from model.reasoning_vla_model import ZR0Model
from model.flow_matching_action_head import FlowmatchingActionHead
from utils.optical_flow_config import OpticalFlowConfig
from utils.slot_config import SlotConfig
from utils.slot_routing import load_slot_supervision
from utils.slot_labels import task_validity
from utils.optical_flow_loss import prepare_flow_targets
from utils.load_training_dataset import build_concat_streaming_dataset, create_dataloader_for_concat, custom_collate_fn
from utils.dataset_seen_tracker import DatasetSeenTracker
from utils.dataset_manifest import validate_resume_manifest, write_resolved_dataset_manifest
from utils.optical_flow_checkpoint import module_checksum, read_flow_artifacts
from model.difference_query import DIFFERENCE_QUERY_WEIGHT_KEY
from train_vla import run_optimizer_step_window
from utils.cli_options import parse_train_options
from test_difference_query_tiny_vla import TinyVlaDifferenceQueryTest as Tiny

MODEL = "/opt/data/private/lq/models/Qwen3-VL-2B-Instruct"
ARTIFACTS = "/opt/data/private/lq/ZR-0-artifacts/structured_slots_review3_v1"
SLOTS = ARTIFACTS + "/slot_sources_v3"
ENTRIES = ["stage05_droid_partial_mixed", "stage05_household_mixed", "stage05_tabletop_mixed", "stage05_rh20t_mixed"]


def emit(value):
    print(json.dumps(value, default=str), flush=True)


def checkpoint_identity(directory):
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(Path(directory).iterdir()) if path.is_file()}


def launch_options(stage, mode, source, run):
    environment = dict(os.environ, INIT_CHECKPOINT=str(source), OUTPUT_DIR=str(run),
        ACCELERATE_CONFIG="accelerate_configs/structured_slots_zero2_bf16.yaml",
        WANDB_PROJECT="ZR-0-Pretraining", WANDB_RUN_NAME=stage + "_" + mode,
        SLOT_SUPERVISION_DIR=SLOTS, SLOT_LOSS_WEIGHT="1", OPTICAL_FLOW_LOSS_WEIGHT="1",
        WITH_SLOT=str(int(mode in {"slot", "both"})), WITH_FLOW=str(int(mode in {"flow", "both"})))
    command = subprocess.check_output(["bash", "scripts/run_structured_slot_stage.sh", stage, "--print-command"],
        cwd=ROOT, env=environment, text=True).strip()
    tokens = shlex.split(command)
    # The diagnostic reduces only head width/depth; data and stage options are
    # parsed from the actual launcher's production command.
    arguments = tokens[tokens.index("train_vla.py") + 1:] + ["--max_length", "1024"]
    if mode in {"flow", "both"}:
        arguments += ["--flow_head_hidden_dim", "16", "--flow_head_num_layers", "1"]
    (run / "command.txt").write_text(command + "\n")
    emit({"production_command": command, "diagnostic_overrides": arguments[len(tokens[tokens.index('train_vla.py') + 1:]):]})
    return parse_train_options(arguments)


def tiny_base(path):
    config = Tiny.make_backbone().model.config
    actual = AutoConfig.from_pretrained(MODEL, local_files_only=True)
    config.text_config.vocab_size = actual.text_config.vocab_size
    config.text_config.max_position_embeddings = 2048
    for key in ("image_token_id", "video_token_id", "vision_start_token_id", "vision_end_token_id"):
        setattr(config, key, getattr(actual, key))
    Qwen3VLForConditionalGeneration(config).save_pretrained(path)
    AutoProcessor.from_pretrained(MODEL).save_pretrained(path)
    expert = Tiny.make_action_head().config
    expert.action_dim = expert.state_dim = 64
    expert.action_horizon = 32
    expert.max_seq_len = 64
    expert.diffusion_transformer_cfg["max_num_positional_embeddings"] = 64
    return expert


def model_stage(source, stage, expert, slot, flow, stats=None):
    return ZR0Model(str(source), None, expert, training_stage=stage,
        tune_vlm=stage != "stage2_aux", tune_action_expert=stage == "stage3_joint",
        use_difference_query=True, num_difference_queries=32, vlm_attention_backend="sdpa",
        slot_config=slot, optical_flow_config=flow, slot_supervision_stats=stats,
        init_from_checkpoint=str(source) if stage != "stage1_ar" else None)


def window(model, samples, accelerator, optimizer, scheduler, step=1):
    batches = [custom_collate_fn([sample]) for sample in samples]
    with torch.autocast("cpu", dtype=torch.bfloat16):
        return run_optimizer_step_window(model=model, batches=batches, accelerator=accelerator,
            optimizer=optimizer, lr_scheduler=scheduler, training_progress=0., loss_type=model.loss_type,
            vlm_loss_weight=0. if model.loss_type == "aux" else 1.,
            action_expert_loss_weight=0. if model.loss_type == "aux" else 1., next_global_step=step,
            training_stage=model.training_stage, slot_config=model.slot_config, optical_flow_config=model.optical_flow_config)


def additional_joint_samples(dataset):
    samples, descriptions = [], []
    for wrapped in dataset.datasets:
        base = wrapped.dataset
        episode = int(base.episodes[0]["episode_index"])
        rows = base._episode_rows(episode)
        candidates = [("non_anchor", row) for row in rows
                      if (episode, int(row["frame_index"])) not in wrapped.reader.anchors]
        candidates = candidates[:1] + [("tail", rows[-1])]
        for kind, row in candidates:
            global_index = int(row["index"])
            position = int(np.searchsorted(base.indices, global_index))
            if position >= len(base.indices) or int(base.indices[position]) != global_index:
                continue
            sample = wrapped[position]
            samples.append(sample)
            descriptions.append({"dataset": wrapped.spec.dataset_entry, "kind": kind,
                "episode": episode, "source_episode": base.flow_reader.contract["mapping"][str(episode)]["old_episode_index"],
                "frame": int(row["frame_index"]), "slot_anchor": bool(sample["slot_anchor"]),
                "flow_exclusion_reason": int(sample["flow_exclusion_reason"])})
    batch = custom_collate_fn(samples)
    counts = {q: int(mask.sum()) for q, mask in task_validity(batch).items()}
    emit({"additional_real_samples": descriptions, "Q_counts": counts})
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stages", nargs="+", default=["stage2_aux", "stage3_joint"])
    parser.add_argument("--modes", nargs="+", choices=["slot", "flow", "both", "none"], default=["slot", "flow", "both", "none"])
    parser.add_argument("--cross-stage-removals", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "experiment.md").write_text(
        "# Bounded Real Production Matrix\n\nSeed42, CPU BF16 tiny Qwen and production losses. "
        "No formal training or W&B run. Data/model/loss contract: docs/experiments/structured_slots_review5/experiment.md. "
        "Per-mode command.txt, experiment.md and result.json record execution.\n\n"
        + shlex.join([sys.executable, *sys.argv]) + "\n")
    torch.set_num_threads(2)
    torch.manual_seed(42)
    accelerator = Accelerator(cpu=True, gradient_accumulation_steps=4)
    expert = tiny_base(args.output / "tiny_base")
    one = model_stage(args.output / "tiny_base", "stage1_ar", expert, SlotConfig(), OpticalFlowConfig())
    one.save_pretrained(args.output / "stage1")
    del one
    stage2_modes = list(dict.fromkeys([mode for mode in args.modes if mode != "none"] +
        (["both"] if "none" in args.modes else []) + (["slot", "flow", "both"] if args.cross_stage_removals else [])))
    router = load_slot_supervision(SLOTS) if any(mode in {"slot", "both"} for mode in stage2_modes) else None
    for stage in args.stages:
        transitions = [(mode, None) for mode in stage2_modes] if stage == "stage2_aux" else [
            (mode, "both" if mode == "none" else mode) for mode in args.modes]
        if stage == "stage3_joint" and args.cross_stage_removals:
            transitions = list(dict.fromkeys(transitions + [("none", "slot"), ("none", "flow"),
                ("none", "both"), ("slot", "both"), ("flow", "both"), ("both", "both")]))
        for mode, source_mode in transitions:
            suffix = "" if source_mode in {None, mode, "both" if mode == "none" else mode} else "_from_" + source_mode
            run = args.output / (stage + "_" + mode + suffix)
            run.mkdir(exist_ok=True)
            source = args.output / "stage1" if stage == "stage2_aux" else args.output / ("stage2_aux_" + source_mode) / "checkpoint"
            source_identity = checkpoint_identity(source)
            options = launch_options(stage, mode, source, run)
            (run / "experiment.md").write_text(
                f"# {stage} {mode} bounded CPU diagnostic\n\n"
                f"Seed 42; production launcher command in command.txt; tiny Qwen/Expert, CPU BF16, "
                f"AdamW lr=0.001, GAS=4, four real samples. No formal training or W&B run. "
                f"Source: {source}. See docs/experiments/structured_slots_review5/experiment.md for the full contract.\n")
            slot = SlotConfig(**{name: getattr(options, name) for name in SlotConfig.__dataclass_fields__})
            flow = OpticalFlowConfig(**{name: getattr(options, name) for name in OpticalFlowConfig.__dataclass_fields__})
            if not slot.enabled or not flow.enabled:
                payload = json.loads(Path(options.aux_dataset_config).read_text())
                for route in payload["datasets"].values():
                    if not flow.enabled:
                        route.update(optical_flow_data_root="/unavailable/review5-flow", optical_flow_manifest="/unavailable/review5-flow/manifest.jsonl",
                            flow_candidate_index="/unavailable/review5-flow/candidates.npy")
                    if not slot.enabled:
                        route["slot_supervision_dir"] = "/unavailable/review5-slot"
                    if mode == "none":
                        route["ar_sidecar_path"] = "/unavailable/review5-aux"
                options.aux_dataset_config = str(run / "disabled-label-routes.json")
                Path(options.aux_dataset_config).write_text(json.dumps(payload, indent=2) + "\n")
            guards = ExitStack()
            if not flow.enabled:
                for target in ("utils.optical_flow_reader.OpticalFlowReader", "utils.aux_data_contract.flow_contract", "h5py.File"):
                    guards.enter_context(patch(target, side_effect=AssertionError("disabled auxiliary label access")))
            if not slot.enabled:
                guards.enter_context(patch("utils.slot_supervision.SlotSupervisionReader", side_effect=AssertionError("disabled Slot label access")))
            with guards:
                dataset = build_concat_streaming_dataset(dataset_entries=options.dataset_entries, model_name_or_path=MODEL, fast_tokenizer_path=None,
                window_size=options.window_size, action_horizon=options.action_horizon, accelerator=None, loss_type=options.loss_type,
                max_length=options.max_length, slot_config=slot, slot_reader=router if slot.enabled else None, optical_flow_config=flow,
                aux_dataset_config=options.aux_dataset_config)
            # Construct the real production sampler/dataloader, then select one
            # deterministic admitted sample per source for bounded diagnostics.
                loader = create_dataloader_for_concat(dataset, batch_size_per_device=1, num_workers=0)
                offsets = [0, *dataset.cumulative_sizes[:-1]]
                diagnostic_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(dataset, offsets), batch_size=4,
                    num_workers=0, collate_fn=custom_collate_fn)
                production_batch = next(iter(diagnostic_loader))
                samples = [subdataset[0] for subdataset in dataset.datasets]
            batch = custom_collate_fn(samples)
            torch.testing.assert_close(batch["input_ids"], production_batch["input_ids"])
            tracker = DatasetSeenTracker(dataset, accelerator, flow_config=flow)
            assert tracker.enabled
            tracker.update([batch])
            valid = task_validity(batch)
            indices, _, masks = prepare_flow_targets(batch, flow, torch.device("cpu"))
            if slot.enabled:
                assert all(valid["Q1"].tolist()), "one dataset lost Slot supervision"
            if flow.enabled:
                assert len(indices) == 4, "one dataset lost Flow supervision"
            torch.save(samples, run / "samples.pt")
            write_resolved_dataset_manifest(run, dataset.resolved_dataset_manifest)
            validate_resume_manifest(dataset.resolved_dataset_manifest, run)
            if flow.enabled:
                corrupt = copy.deepcopy(dataset.resolved_dataset_manifest)
                corrupt["entries"][0]["auxiliary_contract"]["flow"]["manifest_sha256"] = "0" * 64
                try:
                    validate_resume_manifest(corrupt, run)
                except ValueError:
                    pass
                else:
                    raise AssertionError("Flow content change accepted on resume")
            with ExitStack() as head_guards:
                if not slot.enabled:
                    head_guards.enter_context(patch("model.structured_slot_head.StructuredSlotHead.__init__", side_effect=AssertionError("disabled Slot constructed")))
                if not flow.enabled:
                    head_guards.enter_context(patch("model.optical_flow_aux_head.DenseRegressionFlowHead.__init__", side_effect=AssertionError("disabled Flow constructed")))
                model = model_stage(source, stage, expert, slot, flow, router.stats if slot.enabled else None)
            source_query = load_file(str(source / "difference_query.safetensors"))[DIFFERENCE_QUERY_WEIGHT_KEY]
            torch.testing.assert_close(model.backbone.difference_query.weight, source_query, rtol=0, atol=0)
            if stage == "stage3_joint":
                assert read_flow_artifacts(source)["training_stage"] == "stage2_aux"
                source_layout = json.loads((source / "zr0_checkpoint_metadata.json").read_text())["query_role_layout"]
                assert model.query_role_layout == source_layout
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(42)
                    expected_expert = FlowmatchingActionHead(expert, True)
                assert module_checksum(model.action_expert) == module_checksum(expected_expert)
                del expected_expert
            model.resolved_dataset_manifest = dataset.resolved_dataset_manifest
            before = {name: module_checksum(module) for name, module in (("vlm", model.backbone.model),
                ("query", model.backbone.difference_query), ("slot", model.slot_aux), ("flow", model.optical_flow_aux), ("expert", model.action_expert)) if module is not None}
            optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
            optimizer_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
            optimizer_names = [name for name, p in model.named_parameters() if id(p) in optimizer_ids]
            assert slot.enabled == any(name.startswith("slot_aux.") for name in optimizer_names)
            assert flow.enabled == any(name.startswith("optical_flow_aux.") for name in optimizer_names)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
            expert_query_counts = []
            hook = model.action_expert.register_forward_pre_hook(
                lambda _m, values: expert_query_counts.append(values[0]["backbone_embeddings"].shape[1])) if model.action_expert is not None else None
            metrics = window(model, samples, accelerator, optimizer, scheduler)
            if hook is not None:
                hook.remove()
                assert expert_query_counts == [32] * 4
            assert metrics["optimizer_update_applied"]
            if mode == "none":
                assert not any(key.startswith(("flow_", "slot_Q")) for key in batch)
                assert model.slot_aux is None and model.optical_flow_aux is None
                assert metrics["ar_loss_count"] > 0 and metrics["flow_matching_loss_count"] > 0
                assert metrics["ar_loss"] > 0 and metrics["flow_matching_loss"] > 0
            changed = {name: before[name] != module_checksum(module) for name, module in (("vlm", model.backbone.model),
                ("query", model.backbone.difference_query), ("slot", model.slot_aux), ("flow", model.optical_flow_aux), ("expert", model.action_expert)) if module is not None}
            assert changed["query"] and changed["vlm"] == (stage == "stage3_joint")
            assert all(changed[name] for name in ("slot", "flow", "expert") if name in changed)
            model.save_pretrained(run / "checkpoint")
            restored = ZR0Model.from_pretrained(run / "checkpoint", training_stage=stage, resume_training=True,
                tune_vlm=stage == "stage3_joint", tune_action_expert=stage == "stage3_joint")
            assert module_checksum(restored) == module_checksum(model)
            assert restored.slot_config == model.slot_config and restored.optical_flow_config == model.optical_flow_config
            assert restored.query_role_layout == model.query_role_layout
            assert source_identity == checkpoint_identity(source)
            if stage == "stage3_joint":
                inference = ZR0Model.from_pretrained(run / "checkpoint", for_action_inference=True)
                assert inference.slot_aux is None and inference.optical_flow_aux is None
                del inference
            result = {"stage": stage, "mode": mode, "dataset_lengths": [len(ds) for ds in dataset.datasets],
                "source": str(source), "source_mode": source_mode, "source_identity": source_identity,
                "query_exactly_inherited": True, "query_role_layout": model.query_role_layout,
                "expert_query_counts": expert_query_counts, "expert_seed42_initialization": stage == "stage3_joint",
                "slot_config": slot.to_dict(), "optical_flow_config": flow.to_dict(), "optimizer_parameter_names": optimizer_names,
                "identities": list(zip(batch["dataset_id"].tolist(), batch["episode_id"].tolist(), batch["frame_id"].tolist())),
                "flow_count": len(indices), "flow_pixels": int(masks.sum()), "Q_counts": {q: int(v.sum()) for q, v in valid.items()},
                "loss": float(metrics["loss"]), "changed": changed, "checkpoint_model_roundtrip": True,
                "loss_counts": {key: float(metrics[key]) for key in ("ar_loss_count", "flow_matching_loss_count") if key in metrics},
                "disabled_label_access_guarded": mode != "both", "disabled_head_construction_guarded": mode != "both",
                "image_grids": batch["image_grid_thw"].tolist(),
                "vision_contracts": [ds.spec.vision_input_contract for ds in dataset.datasets],
                "seen": tracker.manifest(epoch=0, global_step=1)}
            emit(result)
            (run / "result.json").write_text(json.dumps(result, indent=2, default=str) + "\n")
            with (run / "experiment.md").open("a") as report:
                report.write(f"\nCompleted one update; loss={float(metrics['loss'])}; result.json records counts and parameter changes.\n")
            tracker.save(run, epoch=0, global_step=1)
            if stage == "stage3_joint" and mode == "both":
                extra = additional_joint_samples(dataset)
                # This bounded second window includes carried Slot labels and
                # clamped Flow tails, exercising real missing-label denominators.
                extra_metrics = window(model, extra[:4], accelerator, optimizer, scheduler, step=2)
                emit({"missing_label_backward": {key: float(value) if isinstance(value, torch.Tensor) else value
                    for key, value in extra_metrics.items() if key.endswith("count") or key in {"loss", "optimizer_update_applied"}}})
            for subdataset in dataset.datasets:
                base = getattr(subdataset, "dataset", subdataset)
                if base.flow_reader is not None:
                    base.flow_reader.close()
            del model, restored, optimizer, scheduler, dataset, loader


if __name__ == "__main__":
    main()
