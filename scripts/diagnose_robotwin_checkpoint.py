"""Compare legacy/current conditioning on recorded RoboTwin demonstrations."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image
from torchvision.transforms import ToTensor
from transformers import AutoProcessor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model import ZR0Model
from utils.load_training_dataset import (
    custom_collate_fn,
    prepare_action_expert_inputs_cpu,
    prepare_qwen_vl_inputs_cpu,
)
from utils.normalization import min_max_denorm

CAMERAS = (
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)
JOINTS = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]


def video_frame(path, timestamp, fps):
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        stream.codec_context.thread_count = 2
        container.seek(int(timestamp / stream.time_base), stream=stream)
        for frame in container.decode(stream):
            if float(frame.time) >= timestamp - 0.5 / fps:
                if abs(float(frame.time) - timestamp) > 1.5 / fps:
                    raise ValueError(f"Video timestamp mismatch: {path}, {timestamp}, {frame.time}")
                return frame.to_ndarray(format="rgb24")
    raise ValueError(f"No frame at {timestamp}: {path}")


def action_metrics(prediction, target):
    delta = np.diff(prediction[:, JOINTS], axis=0)
    return {
        "joint_mae_rad": float(np.abs(prediction[:, JOINTS] - target[:, JOINTS]).mean()),
        "gripper_mae": float(np.abs(prediction[:, [6, 13]] - target[:, [6, 13]]).mean()),
        "joint_step_mean_rad": float(np.abs(delta).mean()),
        "joint_step_max_rad": float(np.abs(delta).max()),
        "joint_second_difference_mean_rad": float(np.abs(np.diff(delta, axis=0)).mean()),
        "finite": bool(np.isfinite(prediction).all()),
    }


def log_metrics(path):
    chunks, chunk = [], None
    for line in path.read_text().splitlines():
        if line == "action_chunk:":
            chunk = []
        elif chunk is not None and line.startswith("["):
            try:
                action = json.loads(line)
            except json.JSONDecodeError:
                continue
            if len(action) == 14:
                chunk.append(action)
                if len(chunk) == 16:
                    chunks.append(chunk)
                    chunk = None
    if not chunks:
        return {"path": str(path), "complete_chunks": 0}
    actions = np.asarray(chunks)
    delta = np.diff(actions[:, :, JOINTS], axis=1)
    return {
        "path": str(path), "complete_chunks": len(chunks),
        "finite": bool(np.isfinite(actions).all()),
        "joint_step_abs_quantiles_rad": np.quantile(np.abs(delta), [0.5, 0.95, 0.99, 1]).tolist(),
        "joint_second_difference_mean_rad": float(np.abs(np.diff(delta, axis=1)).mean()),
        "joint_step_max_per_dimension_rad": np.abs(delta).max(axis=(0, 1)).tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, nargs="+", default=[0, 1, 550, 1100])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43])
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    result_path = args.output / "diagnosis.json"
    if result_path.exists():
        raise FileExistsError(result_path)
    torch.set_num_threads(4)
    if args.device.startswith("cuda"):
        torch.cuda.set_per_process_memory_fraction(0.12, args.device)
    stats = {
        key: {name: np.asarray(value, dtype=np.float32) for name, value in values.items()}
        for key, values in json.loads(args.stats.read_text()).items()
    }
    report = {
        "checkpoint": str(args.checkpoint), "dataset": str(args.dataset),
        "stats_sha256": hashlib.sha256(args.stats.read_bytes()).hexdigest(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "device": args.device, "seeds": args.seeds,
        "denoising_steps": 5, "action_horizon": 16,
        "compile": False, "model_dtype": "bfloat16",
        "logs": [log_metrics(p) for p in sorted(args.evaluation.glob("workers/gpu*/server.log"))],
        "samples": [],
    }
    print(json.dumps({"logs": report["logs"]}), flush=True)
    info = json.loads((args.dataset / "meta/info.json").read_text())
    metadata = pq.read_table(args.dataset / "meta/episodes").to_pylist()
    metadata = {row["episode_index"]: row for row in metadata if row["episode_index"] in args.episodes}
    processor = AutoProcessor.from_pretrained(args.checkpoint)
    model = ZR0Model.from_pretrained(
        str(args.checkpoint), for_action_inference=True,
        checkpoint_load_purpose="inference",
    ).to(device=args.device, dtype=torch.bfloat16).eval()
    if model.use_difference_query:
        raise ValueError("This comparison is only defined for legacy checkpoints without Difference Query")
    norm = model.backbone.model.model.language_model.norm
    captured = []
    handle = norm.register_forward_pre_hook(lambda module, inputs: captured.append(inputs[0]))
    arrays = {}
    try:
        with torch.inference_mode():
            for episode_index in args.episodes:
                meta = metadata[episode_index]
                data_path = args.dataset / info["data_path"].format(
                    chunk_index=meta["data/chunk_index"], file_index=meta["data/file_index"]
                )
                rows = pq.read_table(data_path, filters=[("episode_index", "=", episode_index)]).to_pylist()
                for frame_index in sorted({0, (len(rows) - 16) // 3, 2 * (len(rows) - 16) // 3}):
                    row = rows[frame_index]
                    sample = {"task": meta["tasks"][0], "observation.state": torch.tensor([row["observation.state"]])}
                    sample_id = f"episode{episode_index}_frame{frame_index}"
                    original_shapes = {}
                    for key in CAMERAS:
                        video_path = args.dataset / info["video_path"].format(
                            video_key=key,
                            chunk_index=meta[f"videos/{key}/chunk_index"],
                            file_index=meta[f"videos/{key}/file_index"],
                        )
                        timestamp = meta[f"videos/{key}/from_timestamp"] + row["timestamp"]
                        pixels = video_frame(video_path, timestamp, info["fps"])
                        original_shapes[key] = list(pixels.shape)
                        sample[key] = ToTensor()(pixels)
                        if key == CAMERAS[0]:
                            Image.fromarray(pixels).save(args.output / f"{sample_id}.png")
                    expert_inputs = prepare_action_expert_inputs_cpu(sample, stats, 64, True, 16, 14)
                    vl_inputs = prepare_qwen_vl_inputs_cpu(
                        sample, list(CAMERAS), [], processor, "eval", "", None,
                    )
                    batch = custom_collate_fn([{**expert_inputs, **vl_inputs, "sub_task_flag": torch.tensor(0)}])
                    batch = {key: value.to(args.device) for key, value in batch.items()}
                    captured.clear()
                    features = model.backbone(model.backbone.prepare_inputs(batch), compute_vlm_loss=False)
                    if len(captured) != 2:
                        raise ValueError(f"Expected model final norm plus current extra norm, got {len(captured)}")
                    raw = captured[-1]
                    normalized = features["backbone_embeddings"]
                    target = np.asarray([r["action"] for r in rows[frame_index:frame_index + 16]], dtype=np.float32)
                    record = {
                        "id": sample_id, "task": sample["task"], "original_shapes": original_shapes,
                        "image_grid_thw": batch["image_grid_thw"].tolist(),
                        "raw_rms": float(raw.float().square().mean().sqrt()),
                        "normalized_rms": float(normalized.float().square().mean().sqrt()),
                        "raw_equals_final_norm_input": bool(torch.equal(captured[0], raw)),
                        "target": action_metrics(target, target), "predictions": [],
                    }
                    arrays[sample_id + "_target"] = target
                    ae_inputs = model.action_expert.prepare_inputs(batch)
                    for seed in args.seeds:
                        for mode, embeddings in (("current_extra_norm", normalized), ("original_raw", raw)):
                            torch.manual_seed(seed)
                            outputs = model.action_expert.get_action(
                                {**features, "backbone_embeddings": embeddings}, ae_inputs, 5,
                            )["action_pred"][0, :, :14]
                            prediction = min_max_denorm(outputs, stats["action"], True).float().cpu().numpy()
                            fp32_prediction = min_max_denorm(outputs.float(), stats["action"], True).cpu().numpy()
                            record["predictions"].append({
                                "mode": mode, "seed": seed,
                                **action_metrics(prediction, target),
                                "bf16_denorm_max_error": float(np.abs(prediction - fp32_prediction).max()),
                                "fp32_denorm_metrics": action_metrics(fp32_prediction, target),
                            })
                            arrays[f"{sample_id}_{mode}_seed{seed}"] = prediction
                            arrays[f"{sample_id}_{mode}_seed{seed}_fp32_denorm"] = fp32_prediction
                    report["samples"].append(record)
                    print(json.dumps(record), flush=True)
                    captured.clear()
    finally:
        handle.remove()
    if args.device.startswith("cuda"):
        report["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(args.device)
    report["summary"] = {}
    for mode in ("current_extra_norm", "original_raw"):
        rows = [p for s in report["samples"] for p in s["predictions"] if p["mode"] == mode]
        report["summary"][mode] = {
            key: float(np.mean([r[key] for r in rows]))
            for key in ("joint_mae_rad", "gripper_mae", "joint_step_mean_rad", "joint_second_difference_mean_rad")
        }
    result_path.write_text(json.dumps(report, indent=2) + "\n")
    np.savez_compressed(args.output / "actions.npz", **arrays)
    print(json.dumps({"summary": report["summary"]}), flush=True)


if __name__ == "__main__":
    main()
