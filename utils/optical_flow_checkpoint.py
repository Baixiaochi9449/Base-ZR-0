"""Flow artifacts and explicit cross-stage initialization contracts."""

import hashlib
import json
import math
import warnings
from pathlib import Path
import torch

from safetensors.torch import load_file, save_file

from utils.optical_flow_config import OpticalFlowConfig, resolve_stage, stage_description, stage_loss_metadata


CONFIG_NAME = "optical_flow_aux_config.json"
WEIGHTS_NAME = "optical_flow_aux.safetensors"
COMPUTED_FLAGS = ("ar_loss_computed", "slot_loss_computed", "flow_loss_computed", "fm_loss_computed")
STAGE_STATE_SCOPE = "current_stage_completed_optimizer_windows"
V1_OPTIONAL_CONFIG_FIELDS = {
    "flow_vae_model_path", "flow_color_scale", "flow_label_source",
    "flow_sample_min_valid_fraction", "flow_latent_cache_dir", "flow_latent_cache_mode",
    "flow_latent_cache_manifest_sha256",
    "flow_v2_hidden_dim", "flow_v2_num_heads", "flow_v2_num_layers", "flow_v2_mlp_ratio",
    "flow_latent_shape",
}


def initial_stage_training_state(stage, config, slot_aux_type="none"):
    return {**stage_loss_metadata(stage, slot_enabled=slot_aux_type != "none"),
            **({"slot_aux_type": slot_aux_type} if slot_aux_type != "none" else {}),
            "stage_description": stage_description(stage, config, slot_aux_type),
            "stage_metadata_version": 2, "loss_computed_scope": STAGE_STATE_SCOPE,
            "legacy_history_unknown": False, "completed_optimizer_windows": 0}


def read_stage_training_state(payload, config):
    """Read cumulative training evidence; missing legacy evidence is unknown, never true."""
    slot_type = payload.get("slot_aux_type", "none")
    state = initial_stage_training_state(payload["training_stage"], config, slot_type)
    if (slot_type == "none" and payload["training_stage"] == "stage2_aux"
            and payload.get("stage_description") == "stage2 OF-only / Slot not implemented"):
        payload = {**payload, "stage_description": state["stage_description"]}
    if payload.get("stage_metadata_version") in (None, 1):
        warnings.warn("legacy checkpoint has no verified optimizer-update history; history is unknown",
                      RuntimeWarning, stacklevel=2)
        state.update({key: None for key in COMPUTED_FLAGS if key != "slot_loss_computed"})
        state["legacy_history_unknown"] = True
        # Only verified updates are counted; legacy_history_unknown marks this lower bound.
        state["completed_optimizer_windows"] = 0
        return state
    if payload.get("stage_metadata_version") != 2 or payload.get("loss_computed_scope") != STAGE_STATE_SCOPE:
        raise ValueError("invalid checkpoint stage metadata version/scope")
    if type(payload.get("legacy_history_unknown")) is not bool:
        raise ValueError("invalid checkpoint legacy_history_unknown")
    count = payload.get("completed_optimizer_windows")
    if type(count) is not int or count < 0:
        raise ValueError("invalid checkpoint completed_optimizer_windows")
    for key in ("training_stage", "stage_description", "provisional"):
        if payload.get(key) != state[key] or type(payload.get(key)) is not type(state[key]):
            raise ValueError(f"checkpoint stage metadata mismatch: {key}")
    for key in COMPUTED_FLAGS:
        if key not in payload or (type(payload[key]) is not bool
                                 and not (payload[key] is None and payload["legacy_history_unknown"])):
            raise ValueError(f"invalid checkpoint computed flag: {key}")
    if slot_type == "none" and payload["slot_loss_computed"] is not False:
        raise ValueError("Slot loss recorded while Slot is disabled")
    allowed = {"stage1_ar": {"ar_loss_computed"}, "stage2_aux": {"flow_loss_computed"},
               "stage3_joint": {"ar_loss_computed", "fm_loss_computed"} |
                               ({"flow_loss_computed"} if config.enabled else set())}[payload["training_stage"]]
    if slot_type != "none" and payload["training_stage"] in {"stage2_aux", "stage3_joint"}:
        allowed.add("slot_loss_computed")
    if any(payload[key] is True and key not in allowed for key in COMPUTED_FLAGS):
        raise ValueError("checkpoint computed flags conflict with training stage")
    return {key: payload[key] for key in state}


def checkpoint_stage_metadata(model):
    state = getattr(model, "stage_training_state", None)
    if state is None:
        state = initial_stage_training_state(model.training_stage, model.optical_flow_config,
                                            getattr(getattr(model, "slot_config", None), "slot_aux_type", "none"))
    if state.get("training_stage") != model.training_stage:
        raise ValueError("checkpoint training state belongs to a different stage")
    return read_stage_training_state(state, model.optical_flow_config)


def record_stage_training_window(model, metrics):
    """Called after an optimizer window, using flags already reduced across ranks."""
    if getattr(model, "training_stage", None) is None:
        return
    from utils.optimizer_step_loss import optimizer_update_applied
    if not optimizer_update_applied(metrics):
        return
    state = checkpoint_stage_metadata(model)
    if metrics.get("training_stage") != model.training_stage:
        raise ValueError("optimizer window stage does not match checkpoint training state")
    for key in COMPUTED_FLAGS:
        if type(metrics.get(key)) is not bool:
            raise ValueError(f"optimizer window is missing boolean {key}")
        if metrics[key]:
            state[key] = True
    state["completed_optimizer_windows"] += 1
    model.stage_training_state = read_stage_training_state(state, model.optical_flow_config)


def reject_flow_zero3(*, model=None, accelerator=None, config=None, flow_enabled=False):
    """Reject sharded exports until a collective full-parameter exporter exists."""
    head = getattr(model, "optical_flow_aux", None)
    slot_head = getattr(model, "slot_aux", None)
    if head is None:
        head = slot_head
    flow_enabled = flow_enabled or head is not None or getattr(getattr(model, "optical_flow_config", None), "enabled", False)
    if not flow_enabled:
        return
    if config is None and accelerator is not None:
        plugin = getattr(getattr(accelerator, "state", None), "deepspeed_plugin", None)
        config = getattr(plugin, "deepspeed_config", None)
    zero = (config or {}).get("zero_optimization", {})
    stage = zero.get("stage", 0) if isinstance(zero, dict) else zero
    partitioned = head is not None and any(hasattr(p, "ds_id") for p in head.parameters())
    if str(stage) == "3" or partitioned:
        raise RuntimeError("Optical Flow does not support DeepSpeed ZeRO-3: full-parameter gather/export "
                           "is not implemented; refusing partitioned weights. Use ZeRO-2 or non-ZeRO.")
    if str(stage) not in {"0", "2"}:
        raise RuntimeError(f"Optical Flow does not support ZeRO stage {stage!r}; use ZeRO-2 or non-ZeRO.")


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def module_checksum(module):
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _v2_adapter_metadata(config, latent_shape, input_dim, parameter_count):
    return {"num_flow_queries": config.num_flow_queries,
            "input_dim": int(input_dim), "latent_shape": list(latent_shape),
            "hidden_dim": config.flow_v2_hidden_dim, "num_heads": config.flow_v2_num_heads,
            "num_layers": config.flow_v2_num_layers, "mlp_ratio": config.flow_v2_mlp_ratio,
            "init_seed": config.flow_init_seed, "parameter_count": int(parameter_count)}


def _validate_v2_checkpoint_payload(payload, config, query, actual_shapes):
    if payload.get("version") != 4:
        raise ValueError("legacy/incomplete V2 checkpoint protocol is unsupported; rebuild the V2 checkpoint")
    shape = tuple(config.flow_latent_shape or ())
    if len(shape) != 3 or payload.get("latent_shape") != list(shape):
        raise ValueError("V2 checkpoint is missing its complete latent shape")
    metadata = payload.get("v2_protocol")
    required = {"target_protocol", "target_protocol_fingerprint", "cache_entry_version",
                "cache_protocol_fingerprint", "cache_manifest", "adapter", "adapter_sha256"}
    if not isinstance(metadata, dict) or set(metadata) != required:
        raise ValueError("V2 checkpoint protocol metadata is incomplete")
    if payload.get("v2_metadata_sha256") != json_hash(metadata):
        raise ValueError("V2 checkpoint protocol metadata hash mismatch")
    from utils.optical_flow_v2 import canonical_json_hash, validate_target_protocol
    validate_target_protocol(metadata["target_protocol"], config, shape)
    if metadata["target_protocol_fingerprint"] != canonical_json_hash(metadata["target_protocol"]):
        raise ValueError("V2 checkpoint target protocol fingerprint mismatch")
    from utils.optical_flow_v2 import CACHE_ENTRY_VERSION, validate_cache_manifest_identity
    if (metadata["cache_entry_version"] != CACHE_ENTRY_VERSION
            or metadata["cache_protocol_fingerprint"] != metadata["target_protocol_fingerprint"]):
        raise ValueError("V2 checkpoint cache protocol fingerprint/version mismatch")
    validate_cache_manifest_identity(metadata["cache_manifest"], config,
                                     metadata["target_protocol_fingerprint"])
    parameter_count = sum(math.prod(dimensions) for dimensions in actual_shapes.values())
    expected_adapter = _v2_adapter_metadata(config, shape, query.hidden_size, parameter_count)
    if metadata["adapter"] != expected_adapter or metadata["adapter_sha256"] != json_hash(expected_adapter):
        raise ValueError("V2 checkpoint Adapter configuration/parameter count mismatch")


def validate_v2_runtime_payload(payload, builder, head):
    """Compare a loaded checkpoint with the actual local VAE and constructed head."""
    if payload is None or payload.get("replace_flow_head"):
        return
    if payload["config"]["optical_flow_aux_type"] != "wan_vae_latent_v2":
        return
    metadata = payload["v2_protocol"]
    if (metadata["target_protocol"] != builder.target_protocol
            or metadata["target_protocol_fingerprint"] != builder.protocol_fingerprint
            or metadata["cache_protocol_fingerprint"] != builder.protocol_fingerprint
            or metadata["cache_manifest"] != builder.cache_manifest_identity):
        raise ValueError("V2 checkpoint VAE weights/config/protocol differ from the runtime builder")
    adapter = _v2_adapter_metadata(builder.config, head.latent_shape,
                                   head.input_norm.normalized_shape[0], sum(p.numel() for p in head.parameters()))
    if metadata["adapter"] != adapter:
        raise ValueError("V2 checkpoint Adapter metadata differs from the runtime head")


def is_cross_stage_initialization(source_stage, stage, *, initialize, resume):
    from utils.optical_flow_config import STAGES
    return (initialize and not resume and source_stage in STAGES and stage in STAGES
            and source_stage != stage)


def read_flow_artifacts(directory):
    root = Path(directory)
    config_path, weight_path = root / CONFIG_NAME, root / WEIGHTS_NAME
    if not config_path.exists():
        if weight_path.exists():
            raise ValueError("OF weights exist without OF configuration")
        metadata_path = root / "zr0_checkpoint_metadata.json"
        if metadata_path.is_file() and json.loads(metadata_path.read_text()).get("training_stage") is not None:
            raise ValueError("staged checkpoint is missing its OF configuration sidecar")
        return None
    payload = json.loads(config_path.read_text())
    if payload.get("version") in (2, 3):
        raise ValueError("legacy/incomplete V2 checkpoint protocol is unsupported; rebuild the V2 checkpoint")
    if not isinstance(payload.get("config"), dict) or not set(OpticalFlowConfig.__dataclass_fields__).issuperset(payload["config"]):
        raise ValueError("incomplete OF config fields")
    missing = set(OpticalFlowConfig.__dataclass_fields__) - set(payload["config"])
    if missing and (payload.get("version") != 1 or not missing.issubset(V1_OPTIONAL_CONFIG_FIELDS)):
        raise ValueError("incomplete OF config fields")
    config_payload = {name: payload["config"].get(name, getattr(OpticalFlowConfig(), name))
                      for name in OpticalFlowConfig.__dataclass_fields__}
    if config_payload.get("flow_latent_shape") is not None:
        config_payload["flow_latent_shape"] = tuple(config_payload["flow_latent_shape"])
    if payload.get("version") not in (1, 4) or payload.get("config_sha256") != json_hash(payload["config"]):
        raise ValueError("OF config hash/version mismatch")
    config = OpticalFlowConfig(**config_payload).validate()
    if ((config.optical_flow_aux_type == "wan_vae_latent_v2") != (payload.get("version") == 4)):
        raise ValueError("V1/V2 checkpoint type and metadata version conflict")
    from utils.slot_checkpoint import resolve_slot_checkpoint
    slot, _ = resolve_slot_checkpoint(root)
    if slot.slot_aux_type != payload.get("slot_aux_type", "none"):
        raise ValueError("OF sidecar declares Slot but artifacts are missing or conflicting")
    resolve_stage(payload["training_stage"], flow=config, slot=slot)
    stage_state = read_stage_training_state(payload, config)
    metadata_path = root / "zr0_checkpoint_metadata.json"
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text())
        if "stage_metadata_version" in payload or "stage_metadata_version" in metadata:
            compared_keys = stage_state.keys() if payload.get("stage_metadata_version") == 2 else stage_state.keys() & payload.keys()
            if any(key not in metadata or type(metadata[key]) is not type(payload.get(key))
                   or metadata[key] != payload.get(key) for key in compared_keys):
                raise ValueError("ordinary checkpoint and OF sidecar stage metadata mismatch")
    payload.update(stage_state)
    if config.enabled:
        if not weight_path.is_file():
            raise ValueError("OF configuration exists without OF weights")
        if hashlib.sha256(weight_path.read_bytes()).hexdigest() != payload["weights_sha256"]:
            raise ValueError("OF weights hash mismatch")
    elif weight_path.exists():
        raise ValueError("disabled OF checkpoint contains OF weights")
    from model.difference_query import _read_checkpoint_query_data
    query = _read_checkpoint_query_data(root)
    if config.enabled:
        if query is None or not query.enabled:
            raise ValueError("OF checkpoint requires complete enabled Difference Query artifacts")
        if config.num_flow_queries > query.num_difference_queries:
            raise ValueError("OF query count exceeds checkpoint Difference Query count")
        from safetensors import safe_open
        with safe_open(str(weight_path), framework="pt", device="cpu") as tensors:
            actual = {key: tensors.get_slice(key).get_shape() for key in tensors.keys()}
        if config.optical_flow_aux_type == "wan_vae_latent_v2":
            shape = tuple(config.flow_latent_shape or ())
            if len(shape) != 3:
                raise ValueError("V2 checkpoint is missing latent shape")
            hidden, input_dim = config.flow_v2_hidden_dim, query.hidden_size
            expected = {"input_norm.weight": [input_dim], "input_norm.bias": [input_dim],
                        "input_projection.weight": [hidden, input_dim], "input_projection.bias": [hidden],
                        "spatial_tokens": [1, int(shape[1]) * int(shape[2]), hidden],
                        "output_norm.weight": [hidden], "output_norm.bias": [hidden],
                        "output.weight": [int(shape[0]), hidden], "output.bias": [int(shape[0])]}
            for index in range(config.flow_v2_num_layers):
                for key, value in {"q_norm.weight": [hidden], "q_norm.bias": [hidden],
                                   "kv_norm.weight": [hidden], "kv_norm.bias": [hidden],
                                   "attention.in_proj_weight": [3 * hidden, hidden],
                                   "attention.in_proj_bias": [3 * hidden],
                                   "attention.out_proj.weight": [hidden, hidden],
                                   "attention.out_proj.bias": [hidden],
                                   "mlp_norm.weight": [hidden], "mlp_norm.bias": [hidden],
                                   "mlp.0.weight": [config.flow_v2_mlp_ratio * hidden, hidden],
                                   "mlp.0.bias": [config.flow_v2_mlp_ratio * hidden],
                                   "mlp.2.weight": [hidden, config.flow_v2_mlp_ratio * hidden],
                                   "mlp.2.bias": [hidden]}.items():
                    expected[f"layers.{index}.{key}"] = value
        else:
            hidden, input_dim = config.flow_head_hidden_dim, query.hidden_size
            expected = {"input_norm.weight": [input_dim], "input_norm.bias": [input_dim],
                        "input_projection.weight": [hidden, input_dim], "input_projection.bias": [hidden],
                        "grid": [1, config.flow_grid_size ** 2, hidden],
                        "refinement.0.weight": [hidden, hidden, 3, 3], "refinement.0.bias": [hidden],
                        "refinement.1.weight": [hidden], "refinement.1.bias": [hidden],
                        "output.weight": [2, hidden, 3, 3], "output.bias": [2]}
        shapes = expected
        if config.optical_flow_aux_type != "wan_vae_latent_v2":
          for index in range(config.flow_head_num_layers):
            for key, shape in {"attention.in_proj_weight": [3 * hidden, hidden],
                "attention.in_proj_bias": [3 * hidden], "attention.out_proj.weight": [hidden, hidden],
                "attention.out_proj.bias": [hidden], "norm.weight": [hidden], "norm.bias": [hidden],
                "mlp.0.weight": [4 * hidden, hidden], "mlp.0.bias": [4 * hidden],
                "mlp.2.weight": [hidden, 4 * hidden], "mlp.2.bias": [hidden],
                "output_norm.weight": [hidden], "output_norm.bias": [hidden]}.items():
                shapes[f"layers.{index}.{key}"] = shape
        if actual != shapes:
            raise ValueError("OF weight shape / Difference Query hidden size mismatch")
        if config.optical_flow_aux_type == "wan_vae_latent_v2":
            _validate_v2_checkpoint_payload(payload, config, query, actual)
    if query is not None and config.num_flow_queries is not None and metadata_path.is_file():
        from utils.slot_config import resolve_query_layout
        layout = resolve_query_layout(query.num_difference_queries, config.num_flow_queries,
            slot_enabled=slot.enabled, query_enabled=query.enabled)
        if metadata.get("query_role_layout", layout) != layout:
            raise ValueError("OF/common checkpoint query role layout mismatch")
    return payload


def resolve_flow_checkpoint(directory, requested=None, *, explicit_fields=None, stage=None, resume=False, initialize=False):
    payload = read_flow_artifacts(directory) if directory else None
    if payload is None:
        if resume and stage is not None:
            raise ValueError("new-stage resume requires stage/OF checkpoint metadata")
        return (requested or OpticalFlowConfig()).validate(), None
    saved_payload = {name: payload["config"].get(name, getattr(OpticalFlowConfig(), name))
                     for name in OpticalFlowConfig.__dataclass_fields__}
    if saved_payload.get("flow_latent_shape") is not None:
        saved_payload["flow_latent_shape"] = tuple(saved_payload["flow_latent_shape"])
    saved = OpticalFlowConfig(**saved_payload)
    if resume and payload["training_stage"] != stage:
        raise ValueError("resume_from_checkpoint requires the same training_stage")
    if requested is None:
        return saved, payload
    fields = set(requested.to_dict()) if explicit_fields is None else set(explicit_fields)
    cross_stage = is_cross_stage_initialization(
        payload["training_stage"], stage, initialize=initialize, resume=resume)
    replacing = (saved.enabled and requested.enabled and "optical_flow_aux_type" in fields
                 and requested.optical_flow_aux_type != saved.optical_flow_aux_type and cross_stage)
    removing = (saved.enabled and "optical_flow_aux_type" in fields and not requested.enabled and cross_stage)
    if removing and explicit_fields is None:
        defaults = OpticalFlowConfig()
        fields = {name for name in fields if getattr(requested, name) != getattr(defaults, name)} | {
            "optical_flow_aux_type", "optical_flow_loss_weight"}
    if saved.num_flow_queries is not None and "num_flow_queries" in fields and requested.num_flow_queries != saved.num_flow_queries:
        raise ValueError("explicit num_flow_queries conflicts with checkpoint role layout")
    for name in fields:
        if removing and name in {"optical_flow_aux_type", "optical_flow_loss_weight"}:
            continue
        if not replacing and (saved.enabled or resume) and getattr(requested, name) != getattr(saved, name):
            raise ValueError(f"explicit OF config conflicts with checkpoint: {name}")
    merged = {**saved.to_dict(), **{name: getattr(requested, name) for name in fields}}
    if saved.num_flow_queries is not None:
        merged["num_flow_queries"] = saved.num_flow_queries
    if removing and "optical_flow_loss_weight" not in fields:
        merged["optical_flow_loss_weight"] = 0.
    if replacing:
        payload = {**payload, "replace_flow_head": True}
    return OpticalFlowConfig(**merged).validate(), payload


def save_flow_artifacts(model, directory):
    reject_flow_zero3(model=model)
    root = Path(directory)
    config = model.optical_flow_config
    stage_state = checkpoint_stage_metadata(model)
    weight_path = root / WEIGHTS_NAME
    if model.optical_flow_aux is not None:
        save_file(model.optical_flow_aux.state_dict(), str(weight_path))
    elif weight_path.exists():
        weight_path.unlink()
    payload = {"version": 4 if config.optical_flow_aux_type == "wan_vae_latent_v2" else 1, **stage_state,
               "slot_aux_type": getattr(getattr(model, "slot_config", None), "slot_aux_type", "none"),
               "config": config.to_dict(), "config_sha256": json_hash(config.to_dict()),
               "weights_sha256": hashlib.sha256(weight_path.read_bytes()).hexdigest() if config.enabled else None,
               "initialization": getattr(model, "aux_initialization", {})}
    if config.optical_flow_aux_type == "wan_vae_latent_v2":
        builder = getattr(model, "flow_target_builder", None)
        if builder is None or model.optical_flow_aux is None:
            raise ValueError("V2 checkpoint requires its runtime target builder and Adapter")
        shape = tuple(model.optical_flow_aux.latent_shape)
        if tuple(config.flow_latent_shape or ()) != shape:
            raise ValueError("V2 checkpoint config/head latent shapes differ")
        from utils.optical_flow_v2 import CACHE_ENTRY_VERSION, validate_target_protocol
        validate_target_protocol(builder.target_protocol, config, shape,
                                 actual_vae_identity=builder.vae_identity)
        adapter = _v2_adapter_metadata(config, shape,
            model.optical_flow_aux.input_norm.normalized_shape[0],
            sum(parameter.numel() for parameter in model.optical_flow_aux.parameters()))
        v2_protocol = {"target_protocol": builder.target_protocol,
                       "target_protocol_fingerprint": builder.protocol_fingerprint,
                       "cache_entry_version": CACHE_ENTRY_VERSION,
                       "cache_protocol_fingerprint": builder.protocol_fingerprint,
                       "cache_manifest": builder.cache_manifest_identity,
                       "adapter": adapter, "adapter_sha256": json_hash(adapter)}
        payload.update({"latent_shape": list(shape), "v2_protocol": v2_protocol,
                        "v2_metadata_sha256": json_hash(v2_protocol)})
    (root / CONFIG_NAME).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def load_flow_weights(head, directory, payload):
    if payload and payload["config"]["optical_flow_aux_type"] != "none" and not payload.get("replace_flow_head"):
        if head is None:
            raise ValueError("cannot discard a saved OF head during training")
        head.load_state_dict(load_file(str(Path(directory) / WEIGHTS_NAME)), strict=True)
