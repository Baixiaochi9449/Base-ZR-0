#!/usr/bin/env python3
"""CPU regression runner; intentionally does not validate a DeepSpeed runtime."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "lerobot"), str(ROOT / "tests")]


def main():
    import accelerate.utils.other
    import pytest
    import torch
    # Installed Triton initializes a driver when Accelerate imports DeepSpeed
    # merely to unwrap a CPU module. CPU suites exercise native/DDP contracts.
    accelerate.utils.other.is_deepspeed_available = lambda: False
    code = pytest.main(sys.argv[1:])
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU regression unexpectedly initialized CUDA")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
