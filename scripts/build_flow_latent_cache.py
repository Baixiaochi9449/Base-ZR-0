"""Build a bounded, source-verified V2 latent cache sample."""

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import torch

from utils.optical_flow_config import OpticalFlowConfig
from utils.optical_flow_v2 import WanFlowTargetBuilder, flow_to_fixed_rgb, select_v2_flow_indices
from utils.optical_flow_reader import (
    CURRENT_FILE_SCHEMA, CURRENT_MANIFEST_SCHEMA, FlowFileIntegrityVerifier,
)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _episode_mapping(path):
    if path is None:
        return {}
    payload = json.loads(Path(path).read_text())
    pairs = payload.get("matches")
    if payload.get("version") != 1 or not isinstance(pairs, list):
        raise ValueError("unsupported flow episode map")
    mapping = {int(row["flow_episode"]): int(row["full_episode"]) for row in pairs}
    if len(mapping) != len(pairs) or len(set(mapping.values())) != len(mapping):
        raise ValueError("flow episode map is not one-to-one")
    return mapping


def _sample_batch(handle, row, position, manifest_sha256, dataset_episode):
    frame = int(handle["frame_index"][position])
    target_frame = int(handle["target_frame_index"][position])
    source_time = float(handle["source_timestamp_s"][position])
    target_time = float(handle["target_timestamp_s"][position])
    delta_s = float(handle["actual_delta_s"][position])
    return {
        "input_ids": torch.ones(1, 1, dtype=torch.long),
        "flow_supervision_available": torch.tensor([True]),
        "flow_nominal_delta_frames": torch.tensor([int(handle.attrs["nominal_delta_frames"])]),
        "flow_actual_delta_frames": torch.tensor([int(handle["actual_delta_frames"][position])]),
        "flow_label_source": torch.tensor([int(handle["label_source"][position])]),
        "flow_target": {0: torch.from_numpy(handle["flow"][position].astype("float32"))},
        "flow_valid_mask": {0: torch.from_numpy(handle["valid_mask"][position].astype(bool))},
        "flow_dataset_id": [str(row["dataset_id"])],
        "flow_camera": [str(row["camera_key"])],
        "flow_manifest_sha256": [manifest_sha256],
        "flow_generation_identity": [str(row["generation_identity"])],
        "flow_label_identity": [str(row["label_identity"])],
        "flow_label_file_sha256": [str(row["sha256"])],
        "flow_episode_id": torch.tensor([dataset_episode]),
        "flow_label_episode_id": torch.tensor([int(row["merged_episode_index"])]),
        "flow_frame_index": torch.tensor([frame]),
        "flow_target_frame_index": torch.tensor([target_frame]),
        "flow_fps": torch.tensor([float(handle.attrs["fps"])], dtype=torch.float64),
        "flow_source_timestamp_s": torch.tensor([source_time], dtype=torch.float64),
        "flow_target_timestamp_s": torch.tensor([target_time], dtype=torch.float64),
        "flow_actual_delta_s": torch.tensor([delta_s], dtype=torch.float64),
    }


def build_cache(root, manifest, output, builder, *, limit=8, episode_map=None,
                cache_dtype=torch.float32, max_fp16_error=1e-3):
    if type(limit) is not int or limit < 1:
        raise ValueError("cache limit must be positive")
    root, manifest, output = Path(root), Path(manifest), Path(output)
    manifest_sha256 = _sha256(manifest)
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    mapping = _episode_mapping(episode_map)
    written = skipped = 0
    max_error = 0.0
    records = []
    complete = False
    integrity = FlowFileIntegrityVerifier()
    for row in rows:
        path = root / row["hdf5_path"]
        source_description = (f"cache dataset={row.get('dataset_id')}, camera={row.get('camera_key')}, "
                              f"flow_episode={row.get('merged_episode_index')}")
        integrity.verify(path, row.get("sha256"), source=source_description)
        if row.get("schema_version") != CURRENT_MANIFEST_SCHEMA:
            raise ValueError("official V2 cache generation requires the current Stage06 Flow manifest schema")
        dataset_episode = mapping.get(int(row["merged_episode_index"]), int(row["merged_episode_index"]))
        with h5py.File(path, "r") as handle:
            if handle.attrs.get("schema_version") != CURRENT_FILE_SCHEMA or "valid_fraction" not in handle:
                raise ValueError("official V2 cache generation requires the current Stage06 Flow file schema")
            for name in ("generation_identity", "label_identity", "dataset_id", "camera_key"):
                if handle.attrs.get(name) != row.get(name):
                    raise ValueError(f"flow manifest/HDF5 identity mismatch: {name}")
            for position in range(int(row["frame_count"])):
                batch = _sample_batch(handle, row, position, manifest_sha256, dataset_episode)
                if not select_v2_flow_indices(batch, builder.config):
                    continue
                flow = batch["flow_target"][0].to(builder.device)
                mask = batch["flow_valid_mask"][0].to(builder.device)
                rgb = flow_to_fixed_rgb(flow, mask, builder.config.flow_color_scale).unsqueeze(0).unsqueeze(2)
                latent_fp32 = builder._encode(rgb, device=builder.device)[0, :, 0].cpu()
                latent = latent_fp32.to(cache_dtype)
                if cache_dtype == torch.float16:
                    error = float((latent.float() - latent_fp32).abs().max())
                    max_error = max(max_error, error)
                    if error > max_fp16_error:
                        raise ValueError(f"FP16 cache error {error} exceeds {max_fp16_error}")
                cache_path, created = builder.write_cache_entry(batch, 0, latent, root=output)
                records.append(builder.cache_manifest_record(cache_path, batch, 0, root=output))
                written += int(created)
                skipped += int(not created)
                if written + skipped >= limit:
                    complete = True
                    break
        integrity.verify(path, row.get("sha256"), source=source_description)
        if complete:
            break
    if not records:
        raise ValueError("no qualified V2 cache entries were produced")
    cache_manifest_path, cache_manifest_identity, manifest_created = builder.publish_cache_manifest(output, records)
    return {"written": written, "verified_existing": skipped,
            "latent_shape": list(builder.latent_shape), "cache_dtype": str(cache_dtype),
            "max_fp16_abs_error": max_error, "protocol_fingerprint": builder.protocol_fingerprint,
            "vae_config_sha256": builder.vae_config_hash,
            "vae_weights_sha256": builder.vae_weights_hash, "manifest_sha256": manifest_sha256,
            "cache_manifest": str(cache_manifest_path),
            "cache_manifest_sha256": cache_manifest_identity["sha256"],
            "cache_manifest_entry_count": cache_manifest_identity["entry_count"],
            "cache_manifest_created": manifest_created}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--vae-model-path", required=True)
    parser.add_argument("--scale", type=float, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--expected-actual-delta", type=int, required=True)
    parser.add_argument("--label-source", type=int, default=1)
    parser.add_argument("--min-valid-fraction", type=float, default=0.95)
    parser.add_argument("--episode-map", default=None)
    parser.add_argument("--cache-dtype", choices=("float32", "float16"), default="float32")
    parser.add_argument("--max-fp16-error", type=float, default=1e-3)
    args = parser.parse_args()
    config = OpticalFlowConfig(
        optical_flow_aux_type="wan_vae_latent_v2", num_flow_queries=1,
        optical_flow_loss_weight=1.0, flow_vae_model_path=args.vae_model_path,
        flow_color_scale=args.scale, flow_delta_frames=args.expected_actual_delta,
        flow_label_source=args.label_source,
        flow_sample_min_valid_fraction=args.min_valid_fraction,
    )
    builder = WanFlowTargetBuilder(config, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    result = build_cache(args.root, args.manifest, args.output, builder, limit=args.limit,
                         episode_map=args.episode_map,
                         cache_dtype=torch.float16 if args.cache_dtype == "float16" else torch.float32,
                         max_fp16_error=args.max_fp16_error)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
