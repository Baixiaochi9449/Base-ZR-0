"""Two real-data CPU optimizer updates per stage, with actual checkpoint reloads."""
import argparse
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from accelerate import Accelerator

from model.reasoning_vla_model import ZR0Model
from utils.action_expert_config import load_action_expert_config
from utils.constants import DATASET2FEATURE
from utils.dataset_adapters import LeRobotV3FutureDifferenceDataset
from utils.load_training_dataset import custom_collate_fn
from utils.optical_flow_config import OpticalFlowConfig
from utils.optical_flow_checkpoint import module_checksum
from utils.slot_config import SlotConfig
from utils.slot_supervision import SlotSupervisionReader, SlotSupervisedDataset
from train_vla import run_optimizer_step_window, json_scalar_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--supervision", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    if not (root / "experiment.md").is_file():
        raise ValueError("create experiment.md before running diagnostic updates")
    torch.set_num_threads(4)
    cfg = SlotConfig(slot_aux_type="structured_slots_v1", slot_loss_weight=1)
    reader = SlotSupervisionReader(args.supervision)
    source = Path(args.checkpoint)
    result = {"started": datetime.now(timezone.utc).isoformat(), "device": "cpu", "seed": 42,
              "checkpoint": str(source.resolve()), "stages": [], "gpu_smoke": "not_run_busy_devices"}
    try:
        for stage in ("stage2_aux", "stage3_joint"):
            torch.manual_seed(42)
            expert = load_action_expert_config(source / "action_expert_config.json").config
            model = ZR0Model(str(source), None, expert, training_stage=stage, init_from_checkpoint=str(source),
                tune_vlm=stage == "stage3_joint", tune_action_expert=stage == "stage3_joint",
                slot_config=cfg, slot_supervision_stats=reader.stats, optical_flow_config=OpticalFlowConfig(num_flow_queries=8))
            if stage == "stage3_joint":
                model.backbone.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            entry = dict(DATASET2FEATURE["molmoact_tabletop_v3_stage05"], dataset_entry="molmoact_tabletop_v3_stage05")
            base = LeRobotV3FutureDifferenceDataset(entry=entry, processor=model.backbone.processor, loss_type=model.loss_type,
                                                     max_length=1024, action_horizon=32)
            dataset = SlotSupervisedDataset(base, reader, cfg)
            selected = [0, 1] if stage == "stage2_aux" else [int(value["sample_index"]) for value in result["stages"][0]["samples"]]
            batches, identities = [], []
            for index in selected:
                sample = dataset[index]
                batches.append(custom_collate_fn([sample]))
                original = dataset.indices[index]
                identities.append({"sample_index": original, "episode_frame": dataset.identity(original)})
            optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-5, weight_decay=.01)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
            frozen_before = module_checksum(model.backbone.model) if stage == "stage2_aux" else None
            before_query = module_checksum(model.backbone.difference_query)
            before_slot = module_checksum(model.slot_aux)
            run = {"stage": stage, "started": datetime.now(timezone.utc).isoformat(), "samples": identities,
                   "action_expert_constructed": model.action_expert is not None,
                   "slot_parameters": sum(p.numel() for p in model.slot_aux.parameters()),
                   "vision_contract": base.spec.vision_input_contract,
                   "image_mean": model.backbone.processor.image_processor.image_mean,
                   "image_std": model.backbone.processor.image_processor.image_std,
                   "grid": batches[0]["image_grid_thw"].tolist(), "steps": []}
            accelerator = Accelerator(cpu=True)
            for step, batch in enumerate(batches, 1):
                with torch.autocast("cpu", dtype=torch.bfloat16):
                    metrics = run_optimizer_step_window(model=model, batches=[batch], accelerator=accelerator,
                        optimizer=optimizer, lr_scheduler=scheduler, training_progress=0, loss_type=model.loss_type,
                        vlm_loss_weight=1 if stage == "stage3_joint" else 0, action_expert_loss_weight=1 if stage == "stage3_joint" else 0,
                        slot_config=cfg, training_stage=stage, next_global_step=step, collect_training_diagnostics=True)
                assert metrics["optimizer_update_applied"]
                run["steps"].append(json_scalar_metrics(metrics))
                print(stage, step, float(metrics["loss"]), flush=True)
            run.update(query_updated=before_query != module_checksum(model.backbone.difference_query),
                       slot_updated=before_slot != module_checksum(model.slot_aux),
                       frozen_vlm_unchanged=frozen_before == module_checksum(model.backbone.model) if frozen_before else None)
            assert run["query_updated"] and run["slot_updated"]
            if stage == "stage2_aux":
                assert run["frozen_vlm_unchanged"] and model.action_expert is None
            checkpoint = root / stage
            checkpoint.mkdir(exist_ok=True)
            (checkpoint / "experiment.md").write_text((root / "experiment.md").read_text())
            model.save_pretrained(checkpoint)
            model.eval()
            with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
                queries = model.backbone(model.backbone.prepare_inputs(batches[0]), compute_vlm_loss=False)["backbone_embeddings"]
                prediction = model.slot_aux(queries)
            del model, optimizer, scheduler
            gc.collect()
            restored = ZR0Model.from_pretrained(checkpoint, tune_vlm=stage == "stage3_joint", tune_action_expert=stage == "stage3_joint")
            restored.eval()
            with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
                loaded_query = restored.backbone(restored.backbone.prepare_inputs(batches[0]), compute_vlm_loss=False)["backbone_embeddings"]
                loaded = restored.slot_aux(loaded_query)
            for q in prediction:
                torch.testing.assert_close(prediction[q], loaded[q], rtol=0, atol=0)
            run.update(roundtrip_exact=True, checkpoint=str(checkpoint), ended=datetime.now(timezone.utc).isoformat())
            result["stages"].append(run)
            (root / "results.json").write_text(json.dumps(result, indent=2) + "\n")
            del restored, base, dataset, batches, prediction, loaded
            gc.collect()
            source = checkpoint
    except Exception as error:
        result["failure"] = repr(error)
        raise
    finally:
        result["ended"] = datetime.now(timezone.utc).isoformat()
        (root / "results.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
