#!/usr/bin/env python3
import argparse
import importlib.metadata
import json
import os
import site
import sys
from pathlib import Path


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


def validate_static_inputs(model_path: Path, fast_path: Path, dataset_path: Path) -> None:
    require_directory(model_path, "Qwen3-VL model")
    require_files(model_path, ("config.json",), "Qwen3-VL model")
    model_config = load_json(model_path / "config.json")
    require_equal(model_config.get("model_type"), "qwen3_vl", "model_type")

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
        f"W&B={'OK' if require_wandb else 'not required'}"
    )


def main() -> int:
    args = parse_args()
    try:
        validate_static_inputs(args.model_path, args.fast_path, args.dataset_path)
        if not args.static_only:
            validate_runtime(
                args.model_path,
                args.fast_path,
                args.min_cgroup_memory_headroom_gib,
                args.min_gpu_free_gib,
                args.require_wandb,
            )
    except PreflightError as error:
        print(f"Preflight failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
