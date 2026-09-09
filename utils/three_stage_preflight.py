"""Frozen preparation certificates checked by both launcher and trainer."""

import json
from pathlib import Path

from utils.stage05_sidecar import sha256_file

ROOT = Path(__file__).resolve().parents[1]


def preparation_directory(config):
    revision = config.get("preparation_revision")
    if revision not in (None, "flow_exclusion_108"):
        raise ValueError("unknown preparation revision")
    output = Path(config["output_root"])
    return output / revision if revision else output


def implementation_identity():
    paths = [ROOT / "train_vla.py", ROOT / "scripts/run_three_stage_validation.py",
             ROOT / "scripts/prepare_three_stage_validation.py", ROOT / "dataset2feature.yaml",
             ROOT / "lerobot/lerobot/common/datasets/video_utils.py"]
    for directory in ("model", "utils"):
        paths.extend(sorted((ROOT / directory).glob("*.py")))
    return {str(p.relative_to(ROOT)): sha256_file(p) for p in paths}


def validate_runtime_options(options, config):
    from scripts.run_three_stage_validation import training_command
    from utils.cli_options import parse_train_options
    _, command = training_command(config, options.training_stage, options.save_and_exit_after_updates)
    expected = vars(parse_train_options(command[command.index(str(ROOT / "train_vla.py")) + 1:]))
    differences = [key for key, value in expected.items() if getattr(options, key, None) != value]
    if differences:
        raise ValueError(f"bounded runtime differs from its authorized expanded command: {differences}")


def validate_preparation(config):
    from scripts.run_three_stage_validation import validate_config
    validate_config(config)
    output = preparation_directory(config)
    if (output / "preparation_blocked.json").is_file():
        blocked = json.loads((output / "preparation_blocked.json").read_text())
        raise ValueError(f"saved preparation audit is blocked; no automatic re-audit: {blocked['blockers']}")
    certificate = json.loads((output / "preparation_complete.json").read_text())
    if certificate.get("version") != 2 or certificate.get("status") != "passed":
        raise ValueError("three-stage CPU preparation has not passed")
    if certificate["config"] != config or certificate["implementation"] != implementation_identity():
        raise ValueError("three-stage configuration or implementation changed since preparation")
    from utils.preparation_audit_cache import load_preparation_audit_cache, stat_identity
    snapshot = output / "audit_snapshot.json"
    if stat_identity(snapshot) != certificate["audit_snapshot_stat"]:
        raise ValueError("saved preparation audit identity changed")
    cache = load_preparation_audit_cache(str(snapshot))
    cache.check_all()
    return certificate
