"""Bounded CPU audit, actual checkpoint initialization, and real-label head overfit."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from safetensors import safe_open
from safetensors.torch import load_file

from model import ZR0Model
from model.optical_flow_aux_head import build_optical_flow_head
from utils.action_expert_config import load_action_expert_config
from utils.optical_flow_config import OpticalFlowConfig
from utils.optical_flow_reader import OpticalFlowReader
from utils.optical_flow_loss import optical_flow_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--flow-root", default="/opt/data/private/lq/datasets/lerobot/libero/stage06_flow/libero_delta10")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(42)
    result = {"started": datetime.now(timezone.utc).isoformat(),
              "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "checkpoint": str(Path(args.checkpoint).resolve()), "device": "cpu", "seed": 42}
    reader = OpticalFlowReader(args.flow_root, "manifest.2849ed69240ad542.jsonl")
    result["data_audit"] = {"episodes": len(reader.episodes), "frames": len(reader.rows),
                            "manifest_sha256": reader.manifest_sha256}
    checkpoint = Path(args.checkpoint)
    expert_config = load_action_expert_config(checkpoint / "action_expert_config.json").config
    flow_config = OpticalFlowConfig(optical_flow_aux_type="dense_regression_v1", num_flow_queries=8, optical_flow_loss_weight=1.)
    model = ZR0Model(str(checkpoint), None, expert_config, tune_vlm=True, tune_action_expert=False,
                     training_stage="stage2_aux", optical_flow_config=flow_config, init_from_checkpoint=str(checkpoint))
    assert model.action_expert is None
    assert not any("action_expert" in name for name, _ in model.named_parameters())
    state = model.backbone.model.state_dict()
    count = 0
    for path in sorted(checkpoint.glob("model*.safetensors")):
        with safe_open(str(path), framework="pt", device="cpu") as saved:
            for key in saved.keys():
                expected = saved.get_tensor(key)
                assert key in state and torch.equal(state[key].cpu(), expected), key
                count += 1
    assert count > 0
    query = load_file(str(checkpoint / "difference_query.safetensors"))
    assert len(query) == 1
    assert torch.equal(model.backbone.difference_query.weight.detach().cpu(), next(iter(query.values())))
    result["real_initialization"] = {"exact_vlm_tensors": count, "exact_query": True, "action_expert_absent": True,
                                     "head_parameters": model.optical_flow_aux.parameter_counts(),
                                     **model.aux_initialization}
    del model, state
    samples = [reader.read(0, frame) for frame in range(16)]
    assert all(bool(sample["flow_supervision_available"]) for sample in samples)
    cfg = OpticalFlowConfig(optical_flow_aux_type="dense_regression_v1", num_flow_queries=2,
                            optical_flow_loss_weight=1., flow_head_hidden_dim=32, flow_head_num_layers=1)
    queries = torch.randn(16, 4, 32)
    head = build_optical_flow_head(32, cfg)
    optimizer = torch.optim.AdamW(head.parameters(), lr=.0003, weight_decay=0)
    def data(indices):
        return {"input_ids": torch.ones(len(indices), 1, dtype=torch.long),
                **{key: {i: samples[index][key] for i, index in enumerate(indices)} if key in {"flow_target", "flow_valid_mask"}
                   else torch.stack([samples[index][key] for index in indices]) for key in samples[0]}}
    def evaluate():
        with torch.no_grad():
            sums = []
            for start in range(0, 16, 4):
                indices = list(range(start, start + 4))
                sums.append(optical_flow_loss(head(queries[indices]), data(indices), cfg))
            return {key: sum(float(value[key]) for value in sums) / 16
                    for key in ("optical_flow_loss_sum", "flow_epe_sum", "flow_zero_epe_sum")}
    initial = evaluate()
    minimum = float("inf")
    for step in range(320):
        indices = list(range(16))
        optimizer.zero_grad(set_to_none=True)
        outputs = optical_flow_loss(head(queries[indices]), data(indices), cfg)
        loss = outputs["optical_flow_loss"]
        assert torch.isfinite(loss)
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters())
        optimizer.step()
        minimum = min(minimum, float(loss.detach()))
    final = evaluate()
    result["overfit"] = {"samples": 16, "steps": 320, "sample_presentations": 5120, "initial": initial,
                          "final": final, "minimum_loss": minimum, "config": cfg.to_dict(),
                          "epe_decreased": final["flow_epe_sum"] < initial["flow_epe_sum"],
                          "beats_zero_baseline": final["flow_epe_sum"] < final["flow_zero_epe_sum"]}
    reader.close()
    result["ended"] = datetime.now(timezone.utc).isoformat()
    result["nan_oom_or_interruption"] = False
    Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2))
    assert result["overfit"]["epe_decreased"]
    assert result["overfit"]["beats_zero_baseline"]


if __name__ == "__main__":
    main()
