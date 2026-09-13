"""Wan VAE latent targets, filtering, and versioned cache identities."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import uuid

import torch

from utils.preparation_audit_cache import stat_identity


COLOR_PROTOCOL = "middlebury_baker2007_float_rgb_fixed_scale_v2"
TARGET_PROTOCOL_VERSION = 2
CACHE_ENTRY_VERSION = 3
CACHE_MANIFEST_VERSION = 1
CACHE_MANIFEST_NAME = "cache_manifest.v1.json"
CACHE_DIGEST_PROTOCOL = "sha256_dtype_shape_c_order_bytes_v1"
_HEX_DIGEST_LENGTH = 64


def canonical_json_hash(value) -> str:
    payload = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _cache_dtype_name(dtype) -> str:
    names = {torch.float16: "float16", torch.float32: "float32"}
    if dtype not in names:
        raise ValueError("V2 latent cache supports only float16 or float32")
    return names[dtype]


def latent_content_identity(latent: torch.Tensor) -> dict:
    """Hash dtype, shape and contiguous CPU latent bytes, independent of torch.save."""
    if not isinstance(latent, torch.Tensor):
        raise ValueError("V2 cached latent must be a tensor")
    value = latent.detach().to(device="cpu").contiguous()
    dtype_name = _cache_dtype_name(value.dtype)
    shape = [int(dimension) for dimension in value.shape]
    digest = hashlib.sha256()
    digest.update(json.dumps({"dtype": dtype_name, "shape": shape}, sort_keys=True,
                             separators=(",", ":")).encode())
    digest.update(b"\0")
    digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return {"digest_protocol": CACHE_DIGEST_PROTOCOL, "sha256": digest.hexdigest(),
            "dtype": dtype_name, "shape": shape}


def _validate_content_identity(identity, expected_shape=None) -> None:
    required = {"digest_protocol", "sha256", "dtype", "shape"}
    if (not isinstance(identity, dict) or set(identity) != required
            or identity.get("digest_protocol") != CACHE_DIGEST_PROTOCOL
            or not _is_sha256(identity.get("sha256"))
            or identity.get("dtype") not in {"float16", "float32"}
            or not isinstance(identity.get("shape"), list)
            or not identity["shape"] or any(type(value) is not int or value < 1 for value in identity["shape"])
            or (expected_shape is not None and identity["shape"] != list(expected_shape))):
        raise ValueError("invalid V2 latent content identity")


def validate_cache_manifest_identity(identity, config, protocol_fingerprint) -> None:
    expected = config.flow_latent_cache_manifest_sha256
    if config.flow_latent_cache_mode != "strict":
        if identity is not None or expected is not None:
            raise ValueError("online V2 checkpoint cannot lock a strict cache manifest")
        return
    required = {"version", "sha256", "entry_count", "entry_version",
                "protocol_fingerprint", "digest_protocol"}
    if (not isinstance(identity, dict) or set(identity) != required
            or identity.get("version") != CACHE_MANIFEST_VERSION
            or identity.get("sha256") != expected or not _is_sha256(identity.get("sha256"))
            or type(identity.get("entry_count")) is not int or identity["entry_count"] < 1
            or identity.get("entry_version") != CACHE_ENTRY_VERSION
            or identity.get("protocol_fingerprint") != protocol_fingerprint
            or identity.get("digest_protocol") != CACHE_DIGEST_PROTOCOL):
        raise ValueError("V2 strict cache manifest identity differs from configuration/checkpoint")


def _make_colorwheel() -> torch.Tensor:
    sections = ((15, 0, 1), (6, 1, 0), (4, 1, 2), (11, 2, 1), (13, 2, 0), (6, 0, 2))
    wheel = []
    for length, start, end in sections:
        for i in range(length):
            row = [0.0, 0.0, 0.0]
            row[start] = 255.0
            row[end] = 255.0 * i / length
            if start == 1 and end == 0:
                row[start] = 255.0 - 255.0 * i / length
            if start == 2 and end == 1:
                row[start] = 255.0 - 255.0 * i / length
            if start == 2 and end == 0:
                row[start] = 255.0
                row[end] = 255.0 * i / length
            if start == 0 and end == 2:
                row[start] = 255.0
                row[end] = 255.0 - 255.0 * i / length
            wheel.append(row)
    out = torch.tensor(wheel, dtype=torch.float32)
    out[0:15, 0] = 255
    out[0:15, 1] = torch.floor(255 * torch.arange(15) / 15)
    out[15:21, 0] = 255 - torch.floor(255 * torch.arange(6) / 6)
    out[15:21, 1] = 255
    out[21:25, 1] = 255
    out[21:25, 2] = torch.floor(255 * torch.arange(4) / 4)
    out[25:36, 1] = 255 - torch.floor(255 * torch.arange(11) / 11)
    out[25:36, 2] = 255
    out[36:49, 2] = 255
    out[36:49, 0] = torch.floor(255 * torch.arange(13) / 13)
    out[49:55, 2] = 255 - torch.floor(255 * torch.arange(6) / 6)
    out[49:55, 0] = 255
    return out / 255.0


def flow_to_fixed_rgb(flow: torch.Tensor, valid: torch.Tensor, scale: float) -> torch.Tensor:
    """Convert normalized source-extent [2,H,W] flow to float [3,H,W] RGB."""
    if flow.ndim != 3 or flow.shape[0] != 2 or valid.shape != (1, flow.shape[1], flow.shape[2]):
        raise ValueError("flow/valid shapes must be [2,H,W]/[1,H,W]")
    if not math.isfinite(float(scale)) or float(scale) <= 0:
        raise ValueError("flow color scale must be finite and positive")
    if not torch.isfinite(flow).all() or not ((valid == 0) | (valid == 1)).all():
        raise ValueError("flow must be finite and valid mask must be binary")
    u, v = flow[0].float() / float(scale), flow[1].float() / float(scale)
    rad = torch.sqrt(u.square() + v.square())
    a = torch.atan2(-v, -u) / torch.pi
    wheel = _make_colorwheel().to(flow.device)
    fk = (a + 1) / 2 * (wheel.shape[0] - 1)
    k0 = fk.floor().long()
    k1 = (k0 + 1) % wheel.shape[0]
    f = fk - k0
    colors = torch.stack([(1 - f) * wheel[k0, c] + f * wheel[k1, c] for c in range(3)])
    colors = torch.where((rad <= 1).unsqueeze(0),
                         1 - rad.unsqueeze(0) * (1 - colors), colors * 0.75)
    colors = torch.where(valid.bool().expand_as(colors), colors, torch.ones_like(colors))
    return colors.clamp(0, 1)


def _batch_size(batch) -> int:
    value = batch.get("input_ids")
    if not isinstance(value, torch.Tensor) or value.ndim < 1 or value.shape[0] < 1:
        raise ValueError("V2 batch requires nonempty tensor input_ids")
    return int(value.shape[0])


def _sample_value(batch, name, index, batch_size, *, required=True):
    value = batch.get(name)
    if value is None:
        if required:
            raise ValueError(f"V2 supervised batch is missing {name}")
        return None
    if isinstance(value, torch.Tensor):
        if value.ndim == 0 or value.shape[0] != batch_size:
            raise ValueError(f"V2 batch field {name} must have one value per sample")
        return value[index]
    if isinstance(value, (list, tuple)):
        if len(value) != batch_size:
            raise ValueError(f"V2 batch field {name} must have one value per sample")
        return value[index]
    raise ValueError(f"V2 batch field {name} has unsupported type")


def select_v2_flow_indices(batch, config) -> list[int]:
    """Return exactly the samples used by V2 loss and global normalization."""
    config.validate()
    batch_size = _batch_size(batch)
    available = batch.get("flow_supervision_available")
    if available is None:
        return []
    if not isinstance(available, torch.Tensor) or available.ndim == 0 or available.shape[0] != batch_size:
        raise ValueError("V2 flow_supervision_available must have one value per sample")
    if config.flow_delta_frames <= 0:
        raise ValueError("V2 configured nominal delta must be positive")
    selected = []
    for index in range(batch_size):
        if not bool(available[index]):
            continue
        nominal_values = batch.get("flow_nominal_delta_frames")
        expected_delta = (config.flow_delta_frames if nominal_values is None else
                          int(_sample_value(batch, "flow_nominal_delta_frames", index, batch_size)))
        if expected_delta <= 0:
            raise ValueError("invalid per-source flow nominal delta")
        actual_delta = int(_sample_value(batch, "flow_actual_delta_frames", index, batch_size))
        label_source = int(_sample_value(batch, "flow_label_source", index, batch_size))
        if actual_delta != expected_delta or label_source != config.flow_label_source:
            continue
        targets, masks = batch.get("flow_target"), batch.get("flow_valid_mask")
        if not isinstance(targets, dict) or index not in targets:
            raise ValueError("V2 supervised sample is missing flow_target")
        if not isinstance(masks, dict) or index not in masks:
            raise ValueError("V2 supervised sample is missing flow_valid_mask")
        target, mask = targets[index], masks[index]
        if not isinstance(target, torch.Tensor) or target.shape != (2, 224, 224):
            raise ValueError("V2 flow_target must be [2,224,224]")
        if not isinstance(mask, torch.Tensor) or mask.shape != (1, 224, 224):
            raise ValueError("V2 flow_valid_mask must be [1,224,224]")
        if not torch.isfinite(target).all() or not ((mask == 0) | (mask == 1)).all():
            raise ValueError("V2 flow target/mask is nonfinite or nonbinary")
        if float(mask.float().mean()) >= config.flow_sample_min_valid_fraction:
            selected.append(index)
    return selected


def _sha256_file(path: Path, digest) -> None:
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)


def _is_sha256(value) -> bool:
    return (isinstance(value, str) and len(value) == _HEX_DIGEST_LENGTH
            and all(character in "0123456789abcdef" for character in value))


def read_vae_identity(model_path) -> tuple[Path, dict]:
    """Hash the VAE config and weights once without importing or loading Diffusers."""
    model_root = Path(model_path).expanduser().resolve()
    vae_root = model_root / "vae" if (model_root / "vae").is_dir() else model_root
    config_path = vae_root / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"V2 path must be a Wan VAE directory or Diffusers model with vae/config.json: {model_root}"
        )
    config_bytes = config_path.read_bytes()
    try:
        vae_config = json.loads(config_bytes)
    except Exception as error:
        raise ValueError(f"invalid Wan VAE config: {config_path}") from error
    z_dim = vae_config.get("z_dim")
    means, stds = vae_config.get("latents_mean"), vae_config.get("latents_std")
    if (vae_config.get("_class_name") != "AutoencoderKLWan" or type(z_dim) is not int or z_dim < 1
            or not isinstance(means, list) or not isinstance(stds, list)
            or len(means) != z_dim or len(stds) != z_dim
            or any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in means)
            or any(not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0 for value in stds)):
        raise ValueError("Wan VAE config has invalid class/z_dim/latent normalization")
    weight_paths = sorted(path for path in vae_root.rglob("*") if path.is_file()
                          and (path.suffix in {".safetensors", ".bin", ".pt"}
                               or path.name.endswith(".index.json")))
    if not weight_paths:
        raise FileNotFoundError(f"Wan VAE weight files are missing under {vae_root}")
    digest = hashlib.sha256()
    files = []
    for path in weight_paths:
        relative = path.relative_to(vae_root).as_posix()
        size = path.stat().st_size
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(str(size).encode())
        digest.update(b"\0")
        _sha256_file(path, digest)
        files.append({"name": relative, "size_bytes": size})
    identity = {
        "class_name": "AutoencoderKLWan",
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "weights_sha256": digest.hexdigest(),
        "weight_files": files,
        "z_dim": z_dim,
        "latents_mean": [float(value) for value in means],
        "latents_std": [float(value) for value in stds],
    }
    return vae_root, identity


def _validate_vae_identity(identity) -> None:
    required = {"class_name", "config_sha256", "weights_sha256", "weight_files",
                "z_dim", "latents_mean", "latents_std"}
    if not isinstance(identity, dict) or set(identity) != required:
        raise ValueError("V2 protocol has incomplete VAE identity")
    z_dim = identity.get("z_dim")
    means, stds, files = (identity.get("latents_mean"), identity.get("latents_std"),
                          identity.get("weight_files"))
    if (identity.get("class_name") != "AutoencoderKLWan" or not _is_sha256(identity.get("config_sha256"))
            or not _is_sha256(identity.get("weights_sha256")) or type(z_dim) is not int or z_dim < 1
            or not isinstance(files, list) or not files
            or not isinstance(means, list) or len(means) != z_dim
            or not isinstance(stds, list) or len(stds) != z_dim
            or any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in means)
            or any(not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0
                   for value in stds)
            or any(not isinstance(item, dict) or set(item) != {"name", "size_bytes"}
                   or not isinstance(item["name"], str) or not item["name"]
                   or isinstance(item["size_bytes"], bool) or not isinstance(item["size_bytes"], int)
                   or item["size_bytes"] < 1 for item in files)):
        raise ValueError("V2 protocol has invalid VAE identity")


def build_target_protocol(config, latent_shape, vae_identity) -> dict:
    config.validate()
    _validate_vae_identity(vae_identity)
    shape = tuple(int(value) for value in latent_shape)
    if len(shape) != 3 or any(value < 1 for value in shape) or shape[0] != vae_identity["z_dim"]:
        raise ValueError("V2 latent shape is incompatible with VAE z_dim")
    return {
        "version": TARGET_PROTOCOL_VERSION,
        "color": {
            "protocol": COLOR_PROTOCOL,
            "flow_units": "normalized_source_image_extent",
            "scale": float(config.flow_color_scale),
            "components": "u_right_positive_v_down_positive_source_to_target",
            "channel_order": "RGB",
            "angle": "atan2(-v,-u)",
            "magnitude": "sqrt(u^2+v^2)/fixed_scale_no_sample_or_batch_rescale",
            "quantization": "none_float",
            "outside_scale": "wheel_color_times_0.75",
            "rgb_range": [0.0, 1.0],
            "vae_input_transform": "rgb*2-1",
            "invalid_fill": "zero_flow_white",
        },
        "filter": {
            "configured_nominal_delta_frames": int(config.flow_delta_frames),
            "expected_actual_delta": "per_sample_nominal_or_configured_nominal",
            "label_source": int(config.flow_label_source),
            "tail_short_delta": "excluded",
            "min_valid_fraction": float(config.flow_sample_min_valid_fraction),
        },
        "latent": {
            "shape_c_h_w": list(shape),
            "temporal_size": 1,
            "encode": "AutoencoderKLWan.encode.latent_dist.mode",
            "normalization": "(latent-latents_mean)/latents_std",
            "target_dtype": "float32",
        },
        "vae": vae_identity,
    }


def validate_target_protocol(protocol, config, latent_shape, *, actual_vae_identity=None) -> None:
    if not isinstance(protocol, dict):
        raise ValueError("V2 target protocol is missing")
    identity = protocol.get("vae") if actual_vae_identity is None else actual_vae_identity
    try:
        expected = build_target_protocol(config, latent_shape, identity)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid V2 target protocol: {error}") from error
    if protocol != expected:
        raise ValueError("V2 target protocol conflicts with runtime VAE/config; rebuild cache or use matching artifacts")


_CACHE_SOURCE_STRING_FIELDS = (
    "flow_dataset_id", "flow_camera", "flow_manifest_sha256", "flow_generation_identity",
    "flow_label_identity", "flow_label_file_sha256",
)


def cache_source_identity(batch, index) -> dict:
    batch_size = _batch_size(batch)
    strings = {name: _sample_value(batch, name, index, batch_size) for name in _CACHE_SOURCE_STRING_FIELDS}
    if any(not isinstance(value, str) or not value for value in strings.values()):
        raise ValueError("V2 cache source identity contains an empty/non-string field")
    for name in ("flow_manifest_sha256", "flow_generation_identity", "flow_label_identity",
                 "flow_label_file_sha256"):
        if not _is_sha256(strings[name]):
            raise ValueError(f"V2 cache source identity has invalid {name}")
    identity = {
        "dataset_id": strings["flow_dataset_id"],
        "camera": strings["flow_camera"],
        "manifest_sha256": strings["flow_manifest_sha256"],
        "generation_identity": strings["flow_generation_identity"],
        "label_identity": strings["flow_label_identity"],
        "label_file_sha256": strings["flow_label_file_sha256"],
        "episode": int(_sample_value(batch, "flow_episode_id", index, batch_size)),
        "label_episode": int(_sample_value(batch, "flow_label_episode_id", index, batch_size)),
        "source_frame": int(_sample_value(batch, "flow_frame_index", index, batch_size)),
        "target_frame": int(_sample_value(batch, "flow_target_frame_index", index, batch_size)),
        "actual_delta_frames": int(_sample_value(batch, "flow_actual_delta_frames", index, batch_size)),
        "fps": float(_sample_value(batch, "flow_fps", index, batch_size)),
        "source_timestamp_s": float(_sample_value(batch, "flow_source_timestamp_s", index, batch_size)),
        "target_timestamp_s": float(_sample_value(batch, "flow_target_timestamp_s", index, batch_size)),
        "actual_delta_s": float(_sample_value(batch, "flow_actual_delta_s", index, batch_size)),
    }
    if (identity["episode"] < 0 or identity["label_episode"] < 0 or identity["source_frame"] < 0
            or identity["target_frame"] - identity["source_frame"] != identity["actual_delta_frames"]
            or identity["actual_delta_frames"] < 0 or not math.isfinite(identity["fps"])
            or identity["fps"] <= 0 or any(not math.isfinite(identity[name]) for name in
                ("source_timestamp_s", "target_timestamp_s", "actual_delta_s"))
            or not math.isclose(identity["target_timestamp_s"] - identity["source_timestamp_s"],
                                identity["actual_delta_s"], rel_tol=0, abs_tol=2e-5)):
        raise ValueError("V2 cache source frame/time identity is inconsistent")
    return identity


def cache_path_for(batch, index, cache_root, protocol_fingerprint) -> Path:
    source = cache_source_identity(batch, index)
    source_fingerprint = canonical_json_hash(source)
    namespace = canonical_json_hash({"dataset_id": source["dataset_id"], "camera": source["camera"]})[:16]
    return (Path(cache_root) / f"source_{namespace}" / f"episode_{source['episode']:06d}" /
            f"frame_{source['source_frame']:06d}.target_{source['target_frame']:06d}."
            f"delta_{source['actual_delta_frames']:03d}.{source_fingerprint[:16]}."
            f"{protocol_fingerprint[:16]}.pt")


class WanFlowTargetBuilder:
    def __init__(self, config, *, device: torch.device):
        self.config = config.validate()
        self.device = torch.device(device)
        self.vae_root, self.vae_identity = read_vae_identity(config.flow_vae_model_path)
        self.model_path = Path(config.flow_vae_model_path).expanduser().resolve()
        self.vae = None
        if config.flow_latent_cache_mode == "strict":
            if config.flow_latent_shape is None:
                raise ValueError("strict V2 cache mode requires flow_latent_shape from cache metadata")
            self.latent_shape = tuple(config.flow_latent_shape)
        else:
            self.vae = self._load_vae(self.model_path)
            self._freeze_vae()
            self.latent_shape = self._probe_shape()
            if config.flow_latent_shape is not None and tuple(config.flow_latent_shape) != self.latent_shape:
                raise ValueError(f"configured latent shape {config.flow_latent_shape} != VAE {self.latent_shape}")
        self.target_protocol = build_target_protocol(self.config, self.latent_shape, self.vae_identity)
        self.protocol_fingerprint = canonical_json_hash(self.target_protocol)
        self.cache_manifest_identity = None
        self._locked_cache_manifest = None
        self._locked_cache_manifest_stat = None
        self._locked_cache_manifest_pid = os.getpid()
        if config.flow_latent_cache_mode == "strict":
            _, self.cache_manifest_identity = self._read_locked_cache_manifest()

    def _load_vae(self, path):
        try:
            from diffusers import AutoencoderKLWan
        except ImportError as error:
            raise ImportError("wan_vae_latent_v2 target encoding requires diffusers AutoencoderKLWan") from error
        subfolder = "vae" if (path / "vae").is_dir() else None
        kwargs = {} if subfolder is None else {"subfolder": subfolder}
        return AutoencoderKLWan.from_pretrained(str(path), **kwargs).to(self.device)

    def _freeze_vae(self) -> None:
        if self.vae is None:
            return
        self.vae.eval()
        for parameter in self.vae.parameters():
            parameter.requires_grad_(False)

    def ensure_device(self, device) -> torch.device:
        """Move the one loaded encoder only when the prediction device changes."""
        requested = torch.device(device)
        if self.vae is None:
            raise RuntimeError("strict V2 cache mode has no VAE encoder")
        parameter = next(self.vae.parameters(), None)
        buffer = next(self.vae.buffers(), None)
        current = parameter.device if parameter is not None else buffer.device if buffer is not None else self.device
        if current != requested:
            self.vae.to(requested)
        self.device = requested
        self._freeze_vae()
        parameter = next(self.vae.parameters(), None)
        if parameter is not None and parameter.device != requested:
            raise RuntimeError("Wan VAE did not move to the requested target device")
        return requested

    def _normalize(self, latent):
        mean = torch.tensor(self.vae_identity["latents_mean"], device=latent.device,
                            dtype=latent.dtype).view(1, self.vae_identity["z_dim"], 1, 1, 1)
        std = torch.tensor(self.vae_identity["latents_std"], device=latent.device,
                           dtype=latent.dtype).view(1, self.vae_identity["z_dim"], 1, 1, 1)
        return (latent - mean) / std

    @torch.no_grad()
    def _encode(self, rgb, *, device=None):
        if not isinstance(rgb, torch.Tensor) or rgb.ndim != 5 or rgb.shape[1] != 3 or rgb.shape[2] != 1:
            raise ValueError("Wan VAE input must be [B,3,1,H,W]")
        if not torch.isfinite(rgb).all() or rgb.min().item() < 0 or rgb.max().item() > 1:
            raise ValueError("Wan VAE RGB input must be finite in [0,1]")
        target_device = self.ensure_device(self.device if device is None else device)
        parameter = next(self.vae.parameters(), None)
        vae_dtype = parameter.dtype if parameter is not None else rgb.dtype
        input_tensor = rgb.to(target_device, dtype=vae_dtype) * 2 - 1
        encoded = self.vae.encode(input_tensor)
        latent_dist = getattr(encoded, "latent_dist", None)
        if latent_dist is None or not callable(getattr(latent_dist, "mode", None)):
            raise ValueError("Wan VAE encode must expose deterministic latent_dist.mode()")
        latent = self._normalize(latent_dist.mode())
        if (latent.ndim != 5 or latent.shape[1] != self.vae_identity["z_dim"]
                or latent.shape[2] != 1):
            raise ValueError(f"Wan VAE must return [B,Cz,1,Hz,Wz], got {tuple(latent.shape)}")
        latent = latent.to(device=target_device, dtype=torch.float32)
        if not torch.isfinite(latent).all():
            raise ValueError("Wan VAE produced nonfinite normalized latent")
        return latent

    def _probe_shape(self):
        rgb = torch.ones(1, 3, 1, 224, 224, device=self.device)
        shape = self._encode(rgb, device=self.device).shape
        return (int(shape[1]), int(shape[3]), int(shape[4]))

    @property
    def vae_config_hash(self):
        return self.vae_identity["config_sha256"]

    @property
    def vae_weights_hash(self):
        return self.vae_identity["weights_sha256"]

    def cache_path(self, batch, index, *, root=None) -> Path:
        cache_root = self.config.flow_latent_cache_dir if root is None else root
        if not cache_root:
            raise ValueError("V2 cache path requires an explicit cache directory")
        return cache_path_for(batch, index, cache_root, self.protocol_fingerprint)

    def _cache_root(self, root=None) -> Path:
        value = self.config.flow_latent_cache_dir if root is None else root
        if not value:
            raise ValueError("V2 cache path requires an explicit cache directory")
        return Path(value)

    def _validate_cache_manifest(self, manifest, root) -> dict:
        rebuild = "rebuild it with scripts/build_flow_latent_cache.py"
        required = {"version", "status", "entry_version", "digest_protocol",
                    "target_protocol_fingerprint", "entries"}
        if (not isinstance(manifest, dict) or set(manifest) != required
                or manifest.get("version") != CACHE_MANIFEST_VERSION
                or manifest.get("status") != "complete"
                or manifest.get("entry_version") != CACHE_ENTRY_VERSION
                or manifest.get("digest_protocol") != CACHE_DIGEST_PROTOCOL
                or manifest.get("target_protocol_fingerprint") != self.protocol_fingerprint
                or not isinstance(manifest.get("entries"), dict) or not manifest["entries"]):
            raise ValueError(f"unsupported or incomplete V2 cache manifest; {rebuild}")
        for relative, record in manifest["entries"].items():
            candidate = Path(relative)
            if (not isinstance(relative, str) or not relative or candidate.is_absolute()
                    or ".." in candidate.parts or candidate.as_posix() != relative):
                raise ValueError(f"invalid V2 cache manifest path; {rebuild}")
            required_record = {"source", "source_fingerprint", "protocol_fingerprint", "content"}
            if (not isinstance(record, dict) or set(record) != required_record
                    or record.get("source_fingerprint") != canonical_json_hash(record.get("source"))
                    or record.get("protocol_fingerprint") != self.protocol_fingerprint):
                raise ValueError(f"invalid V2 cache manifest entry {relative}; {rebuild}")
            _validate_content_identity(record.get("content"), self.latent_shape)
            path = (root / candidate).resolve()
            if not path.is_relative_to(root.resolve()):
                raise ValueError(f"V2 cache manifest entry escapes its root; {rebuild}")
        return manifest

    def _read_cache_manifest(self, root, *, expected_sha256=None):
        root = Path(root)
        path = root / CACHE_MANIFEST_NAME
        if not path.is_file():
            raise FileNotFoundError(
                f"V2 cache completion manifest is missing: {path}; "
                "rebuild it with scripts/build_flow_latent_cache.py"
            )
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if expected_sha256 is not None and digest != expected_sha256:
            raise ValueError("V2 cache manifest content differs from the locked SHA256")
        try:
            manifest = json.loads(data)
        except Exception as error:
            raise ValueError(f"invalid V2 cache manifest {path}") from error
        self._validate_cache_manifest(manifest, root)
        identity = {"version": CACHE_MANIFEST_VERSION, "sha256": digest,
                    "entry_count": len(manifest["entries"]), "entry_version": CACHE_ENTRY_VERSION,
                    "protocol_fingerprint": self.protocol_fingerprint,
                    "digest_protocol": CACHE_DIGEST_PROTOCOL}
        return manifest, identity

    def _read_locked_cache_manifest(self):
        root = self._cache_root()
        if self._locked_cache_manifest_pid != os.getpid():
            self._locked_cache_manifest = None
            self._locked_cache_manifest_stat = None
            self._locked_cache_manifest_pid = os.getpid()
        manifest_path = root / CACHE_MANIFEST_NAME
        current_stat = stat_identity(manifest_path)
        if (self._locked_cache_manifest is not None
                and current_stat == self._locked_cache_manifest_stat):
            return self._locked_cache_manifest, self.cache_manifest_identity
        manifest, identity = self._read_cache_manifest(
            root, expected_sha256=self.config.flow_latent_cache_manifest_sha256)
        validate_cache_manifest_identity(identity, self.config, self.protocol_fingerprint)
        if self.cache_manifest_identity is not None and identity != self.cache_manifest_identity:
            raise ValueError("V2 cache manifest changed after builder initialization")
        self._locked_cache_manifest = manifest
        self._locked_cache_manifest_stat = stat_identity(manifest_path)
        if self._locked_cache_manifest_stat != current_stat:
            self._locked_cache_manifest = None
            self._locked_cache_manifest_stat = None
            raise ValueError("V2 cache manifest changed while it was being validated")
        return manifest, identity

    def _validate_cache_payload(self, payload, batch, index, path, *, manifest_record=None) -> torch.Tensor:
        rebuild = "rebuild it with scripts/build_flow_latent_cache.py"
        if not isinstance(payload, dict) or payload.get("version") != CACHE_ENTRY_VERSION:
            raise ValueError(f"unsupported V2 latent cache entry {path}; {rebuild}")
        source = cache_source_identity(batch, index)
        if (payload.get("protocol") != self.target_protocol
                or payload.get("protocol_fingerprint") != self.protocol_fingerprint):
            raise ValueError(f"V2 latent cache protocol mismatch at {path}; {rebuild}")
        if (payload.get("source") != source
                or payload.get("source_fingerprint") != canonical_json_hash(source)):
            raise ValueError(f"V2 latent cache source/label identity mismatch at {path}; {rebuild}")
        latent = payload.get("latent")
        if (not isinstance(latent, torch.Tensor) or tuple(latent.shape) != self.latent_shape
                or not torch.isfinite(latent).all()):
            raise ValueError(f"V2 latent cache shape/value mismatch at {path}; {rebuild}")
        try:
            actual_content = latent_content_identity(latent)
            _validate_content_identity(payload.get("content"), self.latent_shape)
        except ValueError as error:
            raise ValueError(f"V2 latent cache content identity is invalid at {path}; {rebuild}") from error
        if payload.get("content") != actual_content:
            raise ValueError(f"V2 latent cache content digest mismatch at {path}; {rebuild}")
        expected_record = {"source": source, "source_fingerprint": canonical_json_hash(source),
                           "protocol_fingerprint": self.protocol_fingerprint,
                           "content": actual_content}
        if manifest_record is not None and manifest_record != expected_record:
            raise ValueError(f"V2 latent cache differs from its locked manifest at {path}; {rebuild}")
        return latent

    def _load_cache_payload(self, path, batch, index, *, manifest_record=None):
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except Exception as error:
            raise ValueError(f"cannot read V2 latent cache {path}; rebuild it") from error
        return self._validate_cache_payload(
            payload, batch, index, path, manifest_record=manifest_record)

    def cache_manifest_record(self, path, batch, index, *, root=None):
        root = self._cache_root(root).resolve()
        path = Path(path).resolve()
        if not path.is_relative_to(root):
            raise ValueError("V2 cache entry is outside the requested cache root")
        latent = self._load_cache_payload(path, batch, index)
        source = cache_source_identity(batch, index)
        return path.relative_to(root).as_posix(), {
            "source": source, "source_fingerprint": canonical_json_hash(source),
            "protocol_fingerprint": self.protocol_fingerprint,
            "content": latent_content_identity(latent),
        }

    @staticmethod
    def _atomic_publish(path, data) -> bool:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
        try:
            with temporary.open("xb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            try:
                os.link(temporary, path)
                return True
            except FileExistsError:
                return False
        finally:
            temporary.unlink(missing_ok=True)

    def publish_cache_manifest(self, root, records):
        root = self._cache_root(root)
        entries = dict(records)
        if not entries or len(entries) != len(records):
            raise ValueError("V2 cache manifest requires nonempty unique entry paths")
        manifest = {"version": CACHE_MANIFEST_VERSION, "status": "complete",
                    "entry_version": CACHE_ENTRY_VERSION, "digest_protocol": CACHE_DIGEST_PROTOCOL,
                    "target_protocol_fingerprint": self.protocol_fingerprint,
                    "entries": entries}
        self._validate_cache_manifest(manifest, root)
        for relative, record in entries.items():
            path = root / relative
            payload = torch.load(path, map_location="cpu", weights_only=True)
            source = record["source"]
            latent = payload.get("latent")
            if (not isinstance(payload, dict) or payload.get("version") != CACHE_ENTRY_VERSION
                    or payload.get("protocol") != self.target_protocol
                    or not isinstance(latent, torch.Tensor) or tuple(latent.shape) != self.latent_shape
                    or not torch.isfinite(latent).all()
                    or latent_content_identity(latent) != record["content"]
                    or payload.get("source") != source
                    or payload.get("source_fingerprint") != record["source_fingerprint"]
                    or payload.get("protocol_fingerprint") != self.protocol_fingerprint
                    or payload.get("content") != record["content"]):
                raise ValueError(f"V2 cache entry changed before manifest publication: {path}")
        data = (json.dumps(manifest, ensure_ascii=True, sort_keys=True,
                           separators=(",", ":")) + "\n").encode()
        path = root / CACHE_MANIFEST_NAME
        created = self._atomic_publish(path, data)
        if not created and path.read_bytes() != data:
            raise ValueError(f"refusing to replace an existing V2 cache manifest: {path}")
        digest = hashlib.sha256(data).hexdigest()
        _, identity = self._read_cache_manifest(root, expected_sha256=digest)
        return path, identity, created

    def write_cache_entry(self, batch, index, latent, *, root=None) -> tuple[Path, bool]:
        root = self._cache_root(root)
        path = self.cache_path(batch, index, root=root)
        source = cache_source_identity(batch, index)
        latent = latent.detach().to(device="cpu").contiguous()
        if tuple(latent.shape) != self.latent_shape or not torch.isfinite(latent).all():
            raise ValueError("refusing to cache invalid V2 latent")
        content = latent_content_identity(latent)

        def validate_existing():
            existing = self._load_cache_payload(path, batch, index)
            if latent_content_identity(existing) != content or not torch.equal(existing, latent):
                raise ValueError(f"existing V2 cache entry conflicts with the supplied target: {path}")
            manifest_path = root / CACHE_MANIFEST_NAME
            if manifest_path.exists():
                manifest, _ = self._read_cache_manifest(root)
                relative = path.resolve().relative_to(root.resolve()).as_posix()
                record = manifest["entries"].get(relative)
                if record is None:
                    raise ValueError("completed V2 cache manifest does not contain the existing entry")
                self._validate_cache_payload(torch.load(path, map_location="cpu", weights_only=True),
                                             batch, index, path, manifest_record=record)
            return path, False

        if (root / CACHE_MANIFEST_NAME).exists() and not path.exists():
            raise ValueError("refusing to extend a completed V2 cache manifest; build a new cache directory")
        if path.exists():
            try:
                return validate_existing()
            except Exception as error:
                raise ValueError(f"existing V2 cache entry is incompatible and was not overwritten: {path}") from error
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"version": CACHE_ENTRY_VERSION, "latent": latent,
                   "protocol": self.target_protocol, "protocol_fingerprint": self.protocol_fingerprint,
                   "source": source, "source_fingerprint": canonical_json_hash(source),
                   "content": content}
        temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
        try:
            with temporary.open("xb") as stream:
                torch.save(payload, stream)
                stream.flush()
                os.fsync(stream.fileno())
            try:
                os.link(temporary, path)
            except FileExistsError:
                return validate_existing()
        finally:
            temporary.unlink(missing_ok=True)
        return path, True

    def build_targets(self, batch, device):
        selected = select_v2_flow_indices(batch, self.config)
        outputs = []
        target_device = torch.device(device)
        for index in selected:
            if self.config.flow_latent_cache_mode == "strict":
                path = self.cache_path(batch, index)
                if not path.is_file():
                    raise FileNotFoundError(
                        f"strict V2 latent cache missing: {path}; rebuild it with scripts/build_flow_latent_cache.py"
                    )
                manifest, _ = self._read_locked_cache_manifest()
                relative = path.resolve().relative_to(self._cache_root().resolve()).as_posix()
                record = manifest["entries"].get(relative)
                if record is None:
                    raise ValueError(f"strict V2 cache entry is absent from the locked manifest: {relative}")
                latent = self._load_cache_payload(path, batch, index, manifest_record=record)
                outputs.append(latent.to(device=target_device, dtype=torch.float32))
            else:
                flow = batch["flow_target"][index].to(device=target_device, dtype=torch.float32)
                mask = batch["flow_valid_mask"][index].to(device=target_device)
                rgb = flow_to_fixed_rgb(flow, mask, self.config.flow_color_scale).unsqueeze(0).unsqueeze(2)
                outputs.append(self._encode(rgb, device=target_device)[0, :, 0])
        if not selected:
            return (torch.empty(0, *self.latent_shape, device=target_device),
                    torch.empty(0, dtype=torch.long, device=target_device))
        return torch.stack(outputs), torch.tensor(selected, device=target_device, dtype=torch.long)
