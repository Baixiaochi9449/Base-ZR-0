#!/usr/bin/env python3
"""Prepare one isolated experiment using the existing Stage05 generators."""

import argparse
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import yaml

from utils.stage05_sidecar import build_stage05_sidecar, load_stage05_sidecar, sha256_file


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    output = Path(config["output_root"])
    output.mkdir(parents=True, exist_ok=False)
    document = ROOT / config["experiment_doc"]
    shutil.copy2(document, output / "experiment.md")
    write_json(output / "requested_config.json", config)
    (output / "worktree.patch").write_bytes(subprocess.check_output(["git", "diff", "HEAD", "--binary"], cwd=ROOT))
    view = output / "config_view"
    (view / "scripts").mkdir(parents=True)
    (view / "utils").symlink_to(ROOT / "utils", target_is_directory=True)
    # Byte-identical audit source gives the existing repository_root binding
    # its own registry without changing either the audit or historical registry.
    for name in ("audit_stage05_token_lengths.py", "build_stage05_sidecars.py"):
        shutil.copy2(ROOT / "scripts" / name, view / "scripts" / name)
    registry = yaml.safe_load((ROOT / config["source_registry"]).read_text())
    original_entries = {name: dict(registry[f"stage05_{name}_mixed"])
                        for name in ("droid", "household", "tabletop", "rh20t")}
    for name in original_entries:
        for phase in ("ar", "joint"):
            registry[f"stage05_{name}_mixed"][f"{phase}_sidecar_path"] = str(output / "sidecars_h10" / phase / name)
    (view / "dataset2feature.yaml").write_text(yaml.safe_dump(registry, sort_keys=False))
    expert = json.loads((ROOT / config["source_action_expert_config"]).read_text())
    expert["action_horizon"] = config["action_horizon"]
    write_json(output / "action_expert_config.json", expert)
    counts = {}
    try:
        for phase in ("ar", "joint"):
            for name, old in original_entries.items():
                entry = registry[f"stage05_{name}_mixed"]
                destination = Path(entry[f"{phase}_sidecar_path"])
                print(f"PREPARE {phase} {name} {destination}", flush=True)
                manifest = build_stage05_sidecar(
                    root=entry["dataset_path"], output=destination,
                    dataset_id=entry["stats_key"], kind=entry["stage05_kind"],
                    embedded_images=entry["embedded_images"], horizon=config["action_horizon"],
                    video_backend=entry.get("video_backend", "pyav"), build_joint=phase == "joint",
                )
                load_stage05_sidecar(destination, verify_source=False,
                                     expected_generation={"horizon": config["action_horizon"]})
                old_path = Path(old[f"{phase}_sidecar_path"])
                if phase == "ar":
                    if not np.array_equal(np.load(destination / "ar_indices.npy"), np.load(old_path / "ar_indices.npy")):
                        raise ValueError(f"{name}: H10 AR eligibility changed")
                item = {"counts": manifest["counts"], "manifest_content_hash": manifest["content_hash"]}
                if phase == "joint":
                    previous = json.loads((old_path / "stats.json").read_text())["statistics"]
                    current = json.loads((destination / "stats.json").read_text())["statistics"]
                    item["statistics_equal_to_h32"] = {key: previous[key] == current[key] for key in current}
                counts[f"{phase}/{name}"] = item
                write_json(output / "data_preparation_progress.json", counts)
        ar_count = sum(counts[f"ar/{name}"]["counts"]["ar_eligible_frames"] for name in original_entries)
        joint_counts = {name: counts[f"joint/{name}"]["counts"]["joint_action_eligible_frames"] for name in original_entries}
        if ar_count != config["ar_eligible_frames"]:
            raise ValueError(f"AR count mismatch: {ar_count}")
        total = sum(joint_counts.values())
        if total != config["joint_expected_frames"]:
            raise ValueError(f"H10 count differs from compact-index prediction: {total}")
        steps = math.ceil(math.ceil(total / 4) / 64)
        summary = {"ar_frames": ar_count, "joint_frames": total, "joint_counts": joint_counts,
                   "joint_probabilities": {key: value / total for key, value in joint_counts.items()},
                   "joint_steps": steps, "joint_warmup_steps": int(steps * config["warmup_ratio"]),
                   "ar_steps": config["ar_optimizer_steps"],
                   "ar_warmup_steps": int(config["ar_optimizer_steps"] * config["warmup_ratio"]),
                   "rank_padding": 4 * math.ceil(total / 4) - total,
                   "final_global_samples": 4 * (math.ceil(total / 4) % 64 or 64)}
        write_json(output / "data_summary.json", summary)
        report = output / "token_audit_h10_format2.json"
        command = [sys.executable, str(view / "scripts/audit_stage05_token_lengths.py"),
                   "--sidecar-root", str(output / "sidecars_h10"), "--processor-path", config["base_model"],
                   "--output", str(report)]
        print("TOKEN_AUDIT " + json.dumps(command), flush=True)
        subprocess.run(command, check=True, cwd=ROOT)
        audit = json.loads(report.read_text())
        spec = {"experiment": config["experiment"], "schema_version": 1,
                "token_audit": {"format_version": 2, "required_max_length": audit["required_max_length"],
                                "report_content_hash": audit["content_hash"],
                                "report_file_sha256": sha256_file(report),
                                "implementation_identity_sha256": audit["implementation_identity"]["sha256"]}}
        write_json(output / "trusted_spec.json", spec)
        subprocess.run([sys.executable, str(view / "scripts/audit_stage05_token_lengths.py"),
                        "--validate-only", "--report", str(report), "--processor-path", config["base_model"],
                        "--max-length", str(config["max_length"]), "--trusted-spec", str(output / "trusted_spec.json")], check=True)
        write_json(output / "preparation_complete.json", {"status": "passed", "completed_at": datetime.now(timezone.utc).isoformat(),
                   "registry_sha256": sha256_file(view / "dataset2feature.yaml"), "data": summary})
        with (output / "experiment.md").open("a") as stream:
            stream.write("\n## Actual Preparation\n\n```json\n" + json.dumps(summary, indent=2) + "\n```\n")
        print("PREPARATION_COMPLETE " + json.dumps(summary), flush=True)
    except BaseException as error:
        write_json(output / "preparation_failed.json", {"error_type": type(error).__name__, "error": str(error),
                   "time": datetime.now(timezone.utc).isoformat()})
        raise


if __name__ == "__main__":
    main()
