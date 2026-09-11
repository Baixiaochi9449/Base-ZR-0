"""Explicit resume entrypoint retaining the pinned production training loop."""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]


def load_override_module():
    spec = importlib.util.spec_from_file_location(
        "zr0_resume_loss_weights", ROOT / "utils/resume_loss_weights.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--loss-weight-resume-config", type=Path, required=True)
    args, training_args = parser.parse_known_args()
    config = json.loads(args.loss_weight_resume_config.read_text())
    if not config.get("enabled") or not config.get("authorization"):
        raise ValueError("loss-weight resume requires explicit enabled authorization")
    runtime = Path(config["runtime_root"])
    sys.path[:0] = [str(runtime), str(runtime / "lerobot")]
    os.chdir(runtime)
    initialize = config.get("allow_stage2_init", False)
    if (training_args[training_args.index("--training_stage") + 1] != "stage3_joint" or
            ("--resume_from_checkpoint" not in training_args and not initialize)):
        raise ValueError("this entrypoint requires Stage 3 resume or explicit Stage 2 initialization")
    output = Path(training_args[training_args.index("--output_ckpt_dir") + 1])

    def record_override(event):
        with (output / f"loss_weight_overrides_rank{os.environ.get('RANK', '0')}.jsonl").open("a") as stream:
            stream.write(json.dumps(event, sort_keys=True) + "\n")

    restore = load_override_module().install_loss_weight_overrides(
        slot_weight=config["outer_weights"]["slot"],
        optical_flow_weight=config["outer_weights"]["optical_flow"], on_override=record_override,
        allow_stage2_init=initialize)
    restore_horizon = None
    try:
        if initialize:
            from utils.cli_options import parse_train_options
            options = parse_train_options(training_args)
            expected = dict(training_stage="stage3_joint", action_horizon=50, max_train_steps=150000,
                vlm_loss_weight=1.0, slot_loss_weight=0.1, optical_flow_loss_weight=1.0,
                action_expert_loss_weight=5.0, expected_global_batch_size=128,
                per_device_train_batch_size=16, gradient_accumulation_steps=2)
            if any(getattr(options, key) != value for key, value in expected.items()):
                raise ValueError("Stage 3 H50 runtime differs from the explicit experiment contract")
            if not options.resume_training and (
                    options.init_from_checkpoint != config["init_from_checkpoint"] or
                    options.action_expert_name_or_path != "/opt/data/private/lq/models/ZR-0"):
                raise ValueError("Stage 3 H50 component sources differ from the experiment contract")
            spec = importlib.util.spec_from_file_location(
                "zr0_stage3_horizon_runtime", ROOT / "utils/stage3_horizon_runtime.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            routes = json.loads(Path(options.aux_dataset_config).read_text())["datasets"]
            restore_horizon = module.install_horizon_adaptation(routes)
        sys.argv = [str(runtime / "train_vla.py"), *training_args]
        runpy.run_path(str(runtime / "train_vla.py"), run_name="__main__")
    finally:
        if restore_horizon is not None:
            restore_horizon()
        restore()


if __name__ == "__main__":
    main()
