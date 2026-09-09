"""Strict source inventories and per-tensor initialization verification."""

import json
from pathlib import Path

import torch
from safetensors import safe_open

from utils.stage05_sidecar import sha256_file

BASE = Path("/opt/data/private/lq/models/ZR-0")


def file_identity(path):
    path = Path(path).resolve()
    return {"path": str(path), "size": path.stat().st_size, "sha256": sha256_file(path)}


def weight_map(directory):
    root = Path(directory)
    index = root / "model.safetensors.index.json"
    if index.is_file():
        mapping = json.loads(index.read_text())["weight_map"]
    else:
        with safe_open(str(root / "model.safetensors"), framework="pt") as stream:
            mapping = {key: "model.safetensors" for key in stream.keys()}
    actual = {}
    for name in sorted(set(mapping.values())):
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError("weight shard path escapes source")
        with safe_open(str(path), framework="pt") as stream:
            for key in stream.keys():
                if key in actual:
                    raise ValueError("duplicate tensor across source shards")
                actual[key] = name
    if actual != mapping:
        raise ValueError("weight index differs from complete shard keys")
    return mapping


def audit_base(directory=BASE):
    from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration
    from model.flow_matching_action_head import FlowmatchingActionHead
    from utils.action_expert_config import load_action_expert_config
    root = Path(directory).resolve()
    if root != BASE:
        raise ValueError("three-stage validation has one fixed base source")
    config = Qwen3VLConfig.from_pretrained(root, local_files_only=True)
    expert = load_action_expert_config(root / "action_expert_config.json", action_horizon_override=10,
        expected_action_dim=64, expected_state_dim=64, expected_vlm_hidden_size=config.text_config.hidden_size)
    mapping = weight_map(root)
    with torch.device("meta"):
        vlm = Qwen3VLForConditionalGeneration(config)
    from accelerate import init_empty_weights
    with init_empty_weights():
        action = FlowmatchingActionHead(expert.config, True)
    expected = vlm.state_dict()
    missing = set(expected) - set(mapping)
    allowed = {"lm_head.weight"} if config.tie_word_embeddings else set()
    if missing - allowed or set(mapping) - set(expected):
        raise ValueError("base VLM key set differs from its actual architecture")
    if config.tie_word_embeddings and vlm.lm_head.weight is not vlm.model.language_model.embed_tokens.weight:
        raise ValueError("base VLM tied weights are not shared")
    for shard in sorted(set(mapping.values())):
        with safe_open(str(root / shard), framework="pt") as stream:
            for key in stream.keys():
                if list(expected[key].shape) != stream.get_slice(key).get_shape():
                    raise ValueError(f"base VLM tensor shape mismatch: {key}")
    with safe_open(str(root / "action_expert.safetensors"), framework="pt") as stream:
        if set(stream.keys()) != set(action.state_dict()):
            raise ValueError("base Action Expert key set differs from its actual architecture")
        for key, value in action.state_dict().items():
            if list(value.shape) != stream.get_slice(key).get_shape():
                raise ValueError(f"base Action Expert tensor shape mismatch: {key}")
    names = set(mapping.values()) | {"action_expert.safetensors", "action_expert_config.json", "config.json"}
    names.update(p.name for p in root.iterdir() if p.is_file() and p.suffix in {".json", ".txt", ".jinja"})
    return {"files": [file_identity(root / name) for name in sorted(names)],
        "vlm_tensor_keys": len(mapping), "vlm_parameters": sum(p.numel() for p in vlm.parameters()),
        "action_expert_parameters": sum(p.numel() for p in action.parameters()),
        "tied_word_embeddings": config.tie_word_embeddings, "hidden_size": config.text_config.hidden_size,
        "action_expert_source_horizon": expert.source_action_horizon,
        "action_expert_runtime_horizon": 10, "action_expert_config": expert.payload}


def compare_component(module, directory, *, weights=None):
    root = Path(directory)
    state = module.state_dict()
    if weights == "difference_query.safetensors":
        state = {"difference_query": module.weight}
    if weights:
        with safe_open(str(root / weights), framework="pt") as stream:
            mapping = {key: weights for key in stream.keys()}
    else:
        mapping = weight_map(root)
    aliases = {"lm_head.weight": "model.language_model.embed_tokens.weight"}
    if set(state) - set(mapping) - (set(aliases) if not weights else set()) or set(mapping) - set(state):
        raise ValueError("loaded component/source key sets differ")
    compared = set()
    for filename in sorted(set(mapping.values())):
        with safe_open(str(root / filename), framework="pt", device="cpu") as stream:
            for key in (key for key, name in mapping.items() if name == filename):
                actual = state[key].detach().cpu()
                expected = stream.get_tensor(key).to(dtype=actual.dtype)
                if not torch.equal(actual, expected):
                    raise RuntimeError(f"loaded component tensor differs from source: {root}/{filename}:{key}")
                compared.add(key)
    for key in set(state) - compared:
        if not torch.equal(state[key], state[aliases[key]]):
            raise RuntimeError("loaded shared embedding tensors differ")
    return {"source": str(root.resolve()), "weights": weights or "VLM shards", "tensors_exact": len(state),
            "parameters": sum(p.numel() for p in module.parameters()),
            "trainable_parameters": sum(p.numel() for p in module.parameters() if p.requires_grad)}


def verify_initialization(model, options):
    stage = options.training_stage
    source = Path(options.vlm_name_or_path).resolve()
    modules = {"vlm": model.backbone.model, "query": model.backbone.difference_query,
        "slot_aux": model.slot_aux, "optical_flow_aux": model.optical_flow_aux,
        "action_expert": model.action_expert}
    for name, module in modules.items():
        expected_present = name in {"vlm", "query"} or (
            stage != "stage1_ar" if name in {"slot_aux", "optical_flow_aux"} else stage == "stage3_joint")
        if (module is not None) != expected_present:
            raise ValueError(f"{stage}: unexpected component construction: {name}")
        if module is not None:
            expected_trainable = name != "vlm" or stage != "stage2_aux"
            if any(p.requires_grad != expected_trainable for p in module.parameters()):
                raise ValueError(f"{stage}: unexpected trainable/frozen parameters: {name}")
    if not options.resume_training:
        if stage == "stage1_ar" and source != BASE:
            raise ValueError("Stage 1 must initialize from the fixed ZR-0 base")
        if stage == "stage3_joint" and Path(options.action_expert_name_or_path or "").resolve() != BASE:
            raise ValueError("Stage 3 Action Expert must initialize only from the fixed ZR-0 base")
        if stage != "stage1_ar":
            meta = json.loads((source / "zr0_checkpoint_metadata.json").read_text())
            if meta.get("training_stage") != {"stage2_aux": "stage1_ar", "stage3_joint": "stage2_aux"}[stage]:
                raise ValueError("cross-stage initialization source is not the preceding stage")
    records = {"vlm": compare_component(model.backbone.model, source)}
    if options.resume_training or stage != "stage1_ar":
        records["query"] = compare_component(model.backbone.difference_query, source, weights="difference_query.safetensors")
    if options.resume_training or stage == "stage3_joint":
        for name, filename in (("slot_aux", "slot_head.safetensors"), ("optical_flow_aux", "optical_flow_aux.safetensors")):
            module = getattr(model, name)
            if module is not None:
                records[name] = compare_component(module, source, weights=filename)
    if model.action_expert is not None:
        records["action_expert"] = compare_component(model.action_expert,
            source if options.resume_training else BASE, weights="action_expert.safetensors")
    if model.query_role_layout["num_flow_queries"] != 16 or model.num_difference_queries != 32:
        raise ValueError("three-stage Query role partition changed")
    for name, module in modules.items():
        if module is not None and name not in records:
            records[name] = {"source": "fresh random initialization", "parameters": sum(p.numel() for p in module.parameters()),
                "trainable_parameters": sum(p.numel() for p in module.parameters() if p.requires_grad)}
    return records


def verify_checkpoint_serialization(model, options):
    """Exercise the actual model artifact writer before any bounded update."""
    import tempfile
    from utils.optical_flow_checkpoint import read_flow_artifacts
    from utils.stage05_checkpoint_contract import (
        STAGE05_AR_TO_JOINT, STAGE05_JOINT_RESUME,
        validate_action_expert_config_provenance, validate_stage05_checkpoint_for_purpose,
    )

    with tempfile.TemporaryDirectory(prefix=".checkpoint-contract-", dir=options.output_ckpt_dir) as temporary:
        directory = Path(temporary)
        model.save_pretrained(directory)
        metadata = json.loads((directory / "zr0_checkpoint_metadata.json").read_text())
        provenance = validate_action_expert_config_provenance(directory, metadata)
        flow = read_flow_artifacts(directory)
        if flow["training_stage"] != options.training_stage:
            raise ValueError("serialized checkpoint training stage mismatch")
        if options.training_stage != "stage2_aux":
            validate_stage05_checkpoint_for_purpose(directory,
                purpose=STAGE05_AR_TO_JOINT if options.training_stage == "stage1_ar" else STAGE05_JOINT_RESUME,
                resume_training=options.training_stage == "stage3_joint",
                external_config_path=directory / "action_expert_config.json",
                requested_action_horizon=options.action_horizon,
                expected_num_difference_queries=options.num_difference_queries,
                expected_action_dim=options.max_pad_state_and_action_length,
                expected_state_dim=options.max_pad_state_and_action_length)
        result = {"status": "passed", "training_stage": options.training_stage,
            "optimizer_updates": 0, "temporary_model_artifact_removed": True,
            "action_expert_config_provenance": provenance, "checkpoint_metadata": metadata}
    with (Path(options.output_ckpt_dir) / "checkpoint_serialization_verified.json").open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
    return result
