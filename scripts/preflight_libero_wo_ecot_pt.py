#!/usr/bin/env python3
import argparse
import importlib.metadata
import json
import os
import shutil
import site
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))



EXPECTED_EPISODES = 1693
EXPECTED_FRAMES = 273465
EXPECTED_TASKS = 40
EXPECTED_CAMERA_KEYS = {
    "observation.images.image",
    "observation.images.image2",
}


class PreflightError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the fixed LIBERO w/o ECoT PT experiment inputs and runtime."
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("/opt/data/private/lq/models/Qwen3-VL-2B-Instruct"),
    )
    parser.add_argument(
        "--fast-path",
        type=Path,
        default=Path("/opt/data/private/lq/ZR-0/fast"),
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=Path("/opt/data/private/lq/datasets/HuggingFaceVLA/libero"),
    )
    parser.add_argument(
        "--static-only",
        action="store_true",
        help="Skip Python package and CUDA checks.",
    )
    parser.add_argument("--min-cgroup-memory-headroom-gib", type=float, default=120.0)
    parser.add_argument("--min-gpu-free-gib", type=float, default=70.0)
    parser.add_argument("--output-path", type=Path)
    parser.add_argument("--min-output-free-gib", type=float, default=0.0)
    parser.add_argument(
        "--expected-checkpoint-kind",
        choices=("ar_only", "joint", "action_only"),
    )
    parser.add_argument("--expected-num-difference-queries", type=int)
    parser.add_argument("--expected-source-action-horizon", type=int)
    parser.add_argument("--reference-action-expert-config", type=Path)
    parser.add_argument("--require-wandb", action="store_true")
    return parser.parse_args()


def require_directory(path: Path, label: str) -> None:
    if not path.is_dir():
        raise PreflightError(f"Missing {label} directory: {path}")


def require_files(root: Path, names: tuple[str, ...], label: str) -> None:
    missing = [str(root / name) for name in names if not (root / name).is_file()]
    if missing:
        raise PreflightError(f"Missing {label} files: {', '.join(missing)}")


def load_json(path: Path) -> dict:
    try:
        with path.open(encoding="utf-8") as file:
            return json.load(file)
    except (OSError, json.JSONDecodeError) as error:
        raise PreflightError(f"Cannot read JSON file {path}: {error}") from error


def count_nonempty_lines(path: Path) -> int:
    try:
        with path.open(encoding="utf-8") as file:
            return sum(bool(line.strip()) for line in file)
    except OSError as error:
        raise PreflightError(f"Cannot read metadata file {path}: {error}") from error


def require_equal(actual: object, expected: object, label: str) -> None:
    if actual != expected:
        raise PreflightError(f"Expected {label}={expected}, found {actual}")


def require_safetensors(path: Path, label: str, *, expected_shape=None) -> None:
    try:
        from safetensors import safe_open

        with safe_open(path, framework="pt", device="cpu") as source:
            keys = list(source.keys())
            if not keys:
                raise PreflightError(f"{label} has no tensors: {path}")
            if expected_shape is not None:
                if len(keys) != 1:
                    raise PreflightError(
                        f"{label} must contain exactly one tensor, found {keys}"
                    )
                tensor = source.get_tensor(keys[0])
                require_equal(list(tensor.shape), list(expected_shape), f"{label} shape")
                if not tensor.isfinite().all().item():
                    raise PreflightError(f"{label} contains NaN or Inf: {path}")
    except PreflightError:
        raise
    except Exception as error:
        raise PreflightError(f"Cannot read {label} {path}: {error}") from error


def validate_model_weight_files(model_path: Path) -> None:
    single_weight = model_path / "model.safetensors"
    if single_weight.is_file():
        require_safetensors(single_weight, "VLM weights")
        return

    index_path = model_path / "model.safetensors.index.json"
    if not index_path.is_file():
        raise PreflightError(
            f"Missing VLM model.safetensors or model.safetensors.index.json: {model_path}"
        )
    index = load_json(index_path)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise PreflightError(f"Invalid VLM weight map: {index_path}")
    relative_paths = sorted(set(weight_map.values()))
    for relative_path in relative_paths:
        if (
            not isinstance(relative_path, str)
            or not relative_path
            or Path(relative_path).is_absolute()
            or ".." in Path(relative_path).parts
        ):
            raise PreflightError(f"Invalid VLM shard path in {index_path}: {relative_path!r}")
        require_safetensors(model_path / relative_path, "VLM weight shard")


def comparable_action_config(config: dict) -> dict:
    comparable = dict(config)
    comparable.pop("action_horizon", None)
    return comparable


def validate_checkpoint_artifacts(
    model_path: Path,
    expected_kind: str,
    expected_num_difference_queries: int | None,
    expected_source_action_horizon: int | None,
    reference_action_expert_config: Path | None,
) -> None:
    metadata_path = model_path / "zr0_checkpoint_metadata.json"
    require_files(model_path, (metadata_path.name,), "ZR-0 checkpoint metadata")
    metadata = load_json(metadata_path)
    require_equal(metadata.get("version"), 1, "checkpoint metadata version")
    require_equal(metadata.get("checkpoint_kind"), expected_kind, "checkpoint kind")
    validate_model_weight_files(model_path)

    if expected_kind in {"joint", "action_only"}:
        require_files(
            model_path,
            ("action_expert_config.json", "action_expert.safetensors"),
            "Action Expert checkpoint",
        )
        action_config_path = model_path / "action_expert_config.json"
        action_config = load_json(action_config_path)
        try:
            from utils.action_expert_config import load_action_expert_config, read_vlm_hidden_size
            from utils.stage05_checkpoint_contract import (
                _checkpoint_has_stage05_identity,
                validate_action_expert_weights,
                validate_generic_action_expert_contract,
            )
            checkpoint_metadata = load_json(metadata_path)
            if isinstance(checkpoint_metadata.get("action_expert_contract"), dict):
                resolved_config = validate_generic_action_expert_contract(
                    model_path,
                    expected_vlm_hidden_size=read_vlm_hidden_size(model_path),
                    validate_weights=True,
                )
            else:
                resolved_config = load_action_expert_config(
                    action_config_path,
                    expected_vlm_hidden_size=read_vlm_hidden_size(model_path),
                )
                validate_action_expert_weights(
                    model_path, resolved_config.config, label="Action Expert weights"
                )
        except ValueError as error:
            raise PreflightError(str(error)) from error
        metadata = load_json(metadata_path)
        if _checkpoint_has_stage05_identity(model_path, metadata) and not isinstance(
            metadata.get("stage05_ar_joint_contract"), dict
        ):
            raise PreflightError(
                "checkpoint carries Stage05 dataset identity but its Stage05 contract is missing"
            )
        if expected_source_action_horizon is not None:
            require_equal(
                action_config.get("action_horizon"),
                expected_source_action_horizon,
                "source Action Expert horizon",
            )
        if reference_action_expert_config is not None:
            if not reference_action_expert_config.is_file():
                raise PreflightError(
                    "Missing reference Action Expert config: "
                    f"{reference_action_expert_config}"
                )
            reference_config = load_json(reference_action_expert_config)
            require_equal(
                comparable_action_config(action_config),
                comparable_action_config(reference_config),
                "Action Expert architecture excluding action_horizon",
            )

    if expected_num_difference_queries is not None:
        require_files(
            model_path,
            ("difference_query_config.json", "difference_query.safetensors"),
            "Difference Query checkpoint",
        )
        query_config = load_json(model_path / "difference_query_config.json")
        require_equal(query_config.get("enabled"), True, "Difference Query enabled")
        require_equal(
            query_config.get("num_difference_queries"),
            expected_num_difference_queries,
            "Difference Query count",
        )
        require_equal(query_config.get("attention_backend"), "sdpa", "attention backend")
        hidden_size = query_config.get("hidden_size")
        if not isinstance(hidden_size, int) or hidden_size < 1:
            raise PreflightError(f"Invalid Difference Query hidden size: {hidden_size!r}")
        require_safetensors(
            model_path / "difference_query.safetensors",
            "Difference Query weights",
            expected_shape=(expected_num_difference_queries, hidden_size),
        )


def existing_parent(path: Path) -> Path:
    candidate = path.expanduser().resolve()
    while not candidate.exists():
        if candidate.parent == candidate:
            raise PreflightError(f"Cannot resolve an existing parent for output: {path}")
        candidate = candidate.parent
    return candidate


def validate_output_space(output_path: Path, min_output_free_gib: float) -> float:
    free_gib = shutil.disk_usage(existing_parent(output_path)).free / 1024**3
    if free_gib < min_output_free_gib:
        raise PreflightError(
            f"Output filesystem has {free_gib:.1f} GiB free; "
            f"at least {min_output_free_gib:.1f} GiB is required"
        )
    return free_gib


def validate_static_inputs(
    model_path: Path,
    fast_path: Path,
    dataset_path: Path,
    *,
    expected_checkpoint_kind: str | None = None,
    expected_num_difference_queries: int | None = None,
    expected_source_action_horizon: int | None = None,
    reference_action_expert_config: Path | None = None,
) -> None:
    require_directory(model_path, "Qwen3-VL model")
    require_files(model_path, ("config.json",), "Qwen3-VL model")
    model_config = load_json(model_path / "config.json")
    require_equal(model_config.get("model_type"), "qwen3_vl", "model_type")
    if expected_checkpoint_kind is not None:
        validate_checkpoint_artifacts(
            model_path,
            expected_checkpoint_kind,
            expected_num_difference_queries,
            expected_source_action_horizon,
            reference_action_expert_config,
        )

    require_directory(fast_path, "FAST tokenizer")
    require_files(
        fast_path,
        (
            "processor_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "processing_action_tokenizer.py",
        ),
        "FAST tokenizer",
    )

    require_directory(dataset_path, "LIBERO dataset")
    incomplete_files = list(dataset_path.rglob("*.incomplete"))
    if incomplete_files:
        preview = ", ".join(str(path) for path in incomplete_files[:3])
        raise PreflightError(f"LIBERO download is incomplete: {preview}")

    meta_path = dataset_path / "meta"
    require_directory(meta_path, "LIBERO metadata")
    require_files(
        meta_path,
        ("info.json", "stats.json", "episodes.jsonl", "tasks.jsonl"),
        "LIBERO metadata",
    )

    info = load_json(meta_path / "info.json")
    require_equal(info.get("total_episodes"), EXPECTED_EPISODES, "total_episodes")
    require_equal(info.get("total_frames"), EXPECTED_FRAMES, "total_frames")
    require_equal(info.get("total_tasks"), EXPECTED_TASKS, "total_tasks")
    require_equal(
        count_nonempty_lines(meta_path / "episodes.jsonl"),
        EXPECTED_EPISODES,
        "episodes.jsonl line count",
    )
    require_equal(
        count_nonempty_lines(meta_path / "tasks.jsonl"),
        EXPECTED_TASKS,
        "tasks.jsonl line count",
    )

    features = info.get("features", {})
    camera_keys = {
        key
        for key, value in features.items()
        if key.startswith("observation.images.")
        and value.get("dtype") in {"image", "video"}
    }
    require_equal(camera_keys, EXPECTED_CAMERA_KEYS, "camera keys")
    require_equal(features.get("observation.state", {}).get("shape"), [8], "state shape")
    require_equal(features.get("action", {}).get("shape"), [7], "action shape")

    parquet_files = list((dataset_path / "data").glob("chunk-*/episode_*.parquet"))
    require_equal(len(parquet_files), EXPECTED_EPISODES, "episode parquet count")

    print(
        "Static inputs OK: "
        f"{EXPECTED_EPISODES} episodes, {EXPECTED_FRAMES} frames, "
        f"{EXPECTED_TASKS} tasks, camera keys={sorted(camera_keys)}"
    )


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError as error:
        raise PreflightError(f"Required package is not installed: {name}") from error


def validate_cgroup_memory_headroom(
    limit_path: Path,
    usage_path: Path,
    min_headroom_gib: float,
) -> float:
    try:
        limit_bytes = int(limit_path.read_text(encoding="utf-8").strip())
        usage_bytes = int(usage_path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError) as error:
        raise PreflightError(f"Cannot read cgroup memory limits: {error}") from error

    headroom_gib = (limit_bytes - usage_bytes) / 1024**3
    if headroom_gib < min_headroom_gib:
        raise PreflightError(
            f"cgroup memory headroom is {headroom_gib:.1f} GiB; "
            f"at least {min_headroom_gib:.1f} GiB is required"
        )
    return headroom_gib


def validate_wandb() -> None:
    if not os.environ.get("WANDB_API_KEY"):
        raise PreflightError("WANDB_API_KEY is not set")
    require_equal(package_version("wandb"), "0.29.0", "wandb version")

    import wandb

    try:
        viewer = wandb.Api(timeout=20).viewer
    except Exception as error:
        raise PreflightError(
            f"W&B authentication or API connectivity failed: {type(error).__name__}"
        ) from error
    if not viewer:
        raise PreflightError("W&B authentication returned no viewer")


def validate_runtime(
    model_path: Path,
    fast_path: Path,
    min_cgroup_memory_headroom_gib: float,
    min_gpu_free_gib: float,
    require_wandb: bool,
    output_path: Path | None = None,
    min_output_free_gib: float = 0.0,
) -> None:
    if sys.version_info[:2] != (3, 10):
        raise PreflightError(f"Expected Python 3.10, found {sys.version.split()[0]}")
    if os.environ.get("CONDA_DEFAULT_ENV") != "ZR-0":
        raise PreflightError(
            "Activate the clean ZR-0 Conda environment before running this experiment."
        )
    if site.ENABLE_USER_SITE:
        raise PreflightError("Python user-site packages are enabled; set PYTHONNOUSERSITE=1")

    import torch
    from transformers import AutoConfig, AutoProcessor
    from transformers.utils import is_flash_attn_2_available

    if torch.__version__.split("+")[0] != "2.6.0":
        raise PreflightError(f"Expected torch 2.6.0, found {torch.__version__}")
    if not torch.version.cuda or not torch.version.cuda.startswith("12."):
        raise PreflightError(f"Expected a CUDA 12 PyTorch build, found {torch.version.cuda}")
    if torch._C._GLIBCXX_USE_CXX11_ABI:
        raise PreflightError("Expected torch cxx11abi=False for the selected flash-attn wheel")
    if not torch.cuda.is_available():
        raise PreflightError("CUDA is not available in PyTorch")
    if torch.cuda.device_count() < 4:
        raise PreflightError(f"Expected at least 4 GPUs, found {torch.cuda.device_count()}")

    gpu_free_gib = []
    for index in range(4):
        free_bytes, _ = torch.cuda.mem_get_info(index)
        free_gib = free_bytes / 1024**3
        gpu_free_gib.append(free_gib)
        if free_gib < min_gpu_free_gib:
            raise PreflightError(
                f"GPU {index} has {free_gib:.1f} GiB free; "
                f"at least {min_gpu_free_gib:.1f} GiB is required"
            )

    cgroup_headroom_gib = validate_cgroup_memory_headroom(
        Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
        Path("/sys/fs/cgroup/memory/memory.usage_in_bytes"),
        min_cgroup_memory_headroom_gib,
    )
    output_free_gib = None
    if output_path is not None:
        output_free_gib = validate_output_space(output_path, min_output_free_gib)

    require_equal(package_version("flash-attn"), "2.7.3", "flash-attn version")
    require_equal(package_version("transformers"), "4.57.1", "transformers version")
    require_equal(package_version("accelerate"), "1.6.0", "accelerate version")
    require_equal(package_version("deepspeed"), "0.15.4", "deepspeed version")
    package_version("lerobot")
    if require_wandb:
        validate_wandb()
    if not is_flash_attn_2_available():
        raise PreflightError("Transformers cannot use FlashAttention 2")

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    require_equal(config.model_type, "qwen3_vl", "loaded Qwen model_type")
    AutoProcessor.from_pretrained(fast_path, trust_remote_code=True)

    devices = [torch.cuda.get_device_name(index) for index in range(4)]
    print(
        "Runtime OK: "
        f"Python {sys.version.split()[0]}, torch {torch.__version__}, "
        f"CUDA {torch.version.cuda}, flash-attn 2.7.3, GPUs={devices}, "
        f"GPU free GiB={[round(value, 1) for value in gpu_free_gib]}, "
        f"cgroup memory headroom={cgroup_headroom_gib:.1f} GiB, "
        f"output free GiB={round(output_free_gib, 1) if output_free_gib is not None else 'not checked'}, "
        f"W&B={'OK' if require_wandb else 'not required'}"
    )


def main() -> int:
    args = parse_args()
    try:
        validate_static_inputs(
            args.model_path,
            args.fast_path,
            args.dataset_path,
            expected_checkpoint_kind=args.expected_checkpoint_kind,
            expected_num_difference_queries=args.expected_num_difference_queries,
            expected_source_action_horizon=args.expected_source_action_horizon,
            reference_action_expert_config=args.reference_action_expert_config,
        )
        if not args.static_only:
            validate_runtime(
                args.model_path,
                args.fast_path,
                args.min_cgroup_memory_headroom_gib,
                args.min_gpu_free_gib,
                args.require_wandb,
                args.output_path,
                args.min_output_free_gib,
            )
    except PreflightError as error:
        print(f"Preflight failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
