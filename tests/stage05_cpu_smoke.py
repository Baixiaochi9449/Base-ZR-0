"""Two CPU optimizer steps through production Stage05 AR and Joint boundaries."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
from types import SimpleNamespace
from unittest.mock import patch

import torch
import yaml
from accelerate import Accelerator
from accelerate.utils import set_seed
from transformers import AutoProcessor

from model.reasoning_vla_model import ZR0Model
from test_difference_query_image import DifferenceQueryImageTest, MODEL_DIR
from train_vla import build_adamw_optimizer, resolve_action_expert_config, run_optimizer_step_window
from utils.dataset_manifest import build_resolved_dataset_manifest
from utils.dataset_spec import resolve_dataset_spec, resolve_objective_requirements
from utils.load_training_dataset import custom_collate_fn
from utils.stage05_dataset import Stage05MixedPretrainingDataset


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("run this CPU-only smoke with CUDA_VISIBLE_DEVICES empty")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(
        ROOT / "docs/experiments/stage05_necessary_repairs_cpu/experiment.md",
        output / "experiment.md",
    )
    torch.set_num_threads(2)
    set_seed(42)
    processor = AutoProcessor.from_pretrained(MODEL_DIR, local_files_only=True)
    base = output / "tiny-base"
    tiny = DifferenceQueryImageTest.make_model()
    tiny.save_pretrained(base)
    processor.save_pretrained(base)
    del tiny
    payload = json.loads((ROOT / "configs/stage05_four_dataset_action_expert.json").read_text())
    payload.update(vlm_output_embedding_dim=32, action_or_state_token_embedding_dim=32,
                   mlp_hidden_size=16)
    payload["diffusion_transformer_cfg"].update(
        num_attention_heads=4, attention_head_dim=8, output_dim=16,
        num_layers=2, dropout=0.0, final_dropout=False,
    )
    config_path = output / "tiny_action_expert_config.json"
    config_path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    registry = yaml.safe_load((ROOT / "dataset2feature.yaml").read_text())
    from utils.stage05_checkpoint_contract import STAGE05_DATASET_ENTRIES

    records = {}
    previous_state = None
    for loss_type in ("vlm", "vlm_and_action"):
        stage = "ar" if loss_type == "vlm" else "joint"
        source = base if stage == "ar" else output / "ar-checkpoint"
        options = SimpleNamespace(
            vlm_name_or_path=str(source), action_expert_name_or_path=None,
            action_expert_config_path=str(config_path), resume_training=False,
            checkpoint_load_purpose=None if stage == "ar" else "stage05_ar_to_joint",
            loss_type=loss_type, action_horizon=32, max_pad_state_and_action_length=64,
            use_difference_query=True, num_difference_queries=32, vlm_attention_backend="sdpa",
        )
        resolved = resolve_action_expert_config(options, return_resolved=True)
        set_seed(42)
        model = ZR0Model(
            vlm_name_or_path=options.vlm_name_or_path, action_expert_name_or_path=None,
            action_expert_config=resolved.config, tune_vlm=True,
            tune_action_expert=stage == "joint", loss_type=loss_type,
            use_difference_query=True, num_difference_queries=32, vlm_attention_backend="sdpa",
            checkpoint_load_purpose=options.checkpoint_load_purpose,
            action_expert_config_path=str(config_path),
        ).float()
        if previous_state is not None:
            assert set(previous_state) == set(model.backbone.state_dict())
            for name, parameter in model.backbone.state_dict().items():
                torch.testing.assert_close(parameter, previous_state[name], rtol=0, atol=0)
        assert (model.action_expert is None) == (stage == "ar")
        specs = []
        for name in STAGE05_DATASET_ENTRIES:
            entry = {**registry[name], "dataset_entry": name}
            requirements = resolve_objective_requirements(
                loss_type, adapter="stage05_mixed_pretraining",
                target_text_field="train_data", dataset_type="vla", dataset_entry=name,
            )
            specs.append(resolve_dataset_spec(name, entry, action_horizon=32,
                                              window_size=1, requirements=requirements))
        entry = {**registry["stage05_tabletop_mixed"], "dataset_entry": "stage05_tabletop_mixed"}
        dataset = Stage05MixedPretrainingDataset(
            entry=entry, processor=processor, loss_type=loss_type,
            max_length=941, action_horizon=32, dataset_id=2,
        )
        sample = dataset[0]
        batch = custom_collate_fn([sample])
        model.resolved_dataset_manifest = build_resolved_dataset_manifest(specs, loss_type)
        model.action_expert_config_source_bytes = config_path.read_bytes()
        model.action_expert_config_source_sha256 = hashlib.sha256(config_path.read_bytes()).hexdigest()
        accelerator = Accelerator(cpu=True, gradient_accumulation_steps=1)
        optimizer = build_adamw_optimizer(model, learning_rate=1e-5,
                                          beta1=0.9, beta2=0.95, epsilon=1e-8)
        assert not optimizer.state
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        optimizer, prepared, scheduler = accelerator.prepare(optimizer, model, scheduler)
        before = model.backbone.difference_query.weight.detach().clone()
        prepared.train()
        metrics = run_optimizer_step_window(
            model=prepared, batches=[batch], accelerator=accelerator,
            optimizer=optimizer, lr_scheduler=scheduler, training_progress=0.0,
            loss_type=loss_type, vlm_loss_weight=1.0,
            action_expert_loss_weight=0.0 if stage == "ar" else 5.0,
            next_global_step=1,
        )
        assert torch.isfinite(metrics["total_loss"])
        assert not torch.equal(before, model.backbone.difference_query.weight)
        assert optimizer.state
        # The production loader loads BF16. Save in that dtype for an exact
        # serialization comparison while CPU optimizer computation stays FP32.
        model.backbone.model.to(torch.bfloat16)
        previous_state = {name: value.detach().float().clone()
                          for name, value in model.backbone.state_dict().items()}
        checkpoint = output / f"{stage}-checkpoint"
        model.save_pretrained(checkpoint)
        records[stage] = {
            "optimizer_steps": 1, "device": str(accelerator.device),
            "sample_global_index": int(sample["sample_global_index"]),
            "episode_id": int(sample["episode_id"]), "frame_id": int(sample["frame_id"]),
            "image_grid_thw": batch["image_grid_thw"].tolist(),
            "checkpoint": str(checkpoint),
            "expert_present": model.action_expert is not None,
            "fresh_optimizer": True, "query_updated": True,
            "exact_ar_backbone_restore": stage == "joint",
            "metrics": {name: float(value) for name, value in metrics.items()},
        }
        accelerator.free_memory()
        del prepared, model, optimizer, scheduler, dataset, batch
    result = {"status": "passed", "total_cpu_optimizer_steps": 2, "stages": records}
    (output / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("RESULT " + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    # CPU unwrapping needs no DeepSpeed engine; the installed optional Triton
    # package cannot initialize without a GPU driver. Keep this harness-local.
    with patch("accelerate.utils.other.is_deepspeed_available", return_value=False):
        main()
