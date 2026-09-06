#!/usr/bin/env python3
"""Save deterministic trajectory evidence for the Stage05 canonical contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.stage05_canonical import (
    canonical_droid_arrays,
    canonical_molmo_arrays,
    canonical_rh20t_arrays,
    wrap_angles,
)
from utils.stage05_sidecar import load_stage05_sidecar, sha256_file


DATASETS = {
    "droid": ("/opt/data/private/lq/datasets/droid_1.0.1_stage05_full_95658_20260831", "droid"),
    "household": ("/opt/data/private/lq/datasets/molmoact_dataset_household-v3_stage05", "molmo"),
    "tabletop": ("/opt/data/private/lq/datasets/molmoact_dataset_tabletop-v3_stage05", "molmo"),
    "rh20t": ("/opt/data/private/lq/datasets/RH20T-v30_stage05", "rh20t"),
}


def _source_episodes(root: Path) -> dict[int, dict]:
    result = {}
    for path in sorted((root / "meta/episodes").glob("**/*.parquet")):
        columns = [
            "episode_index", "data/chunk_index", "data/file_index",
            "dataset_from_index", "dataset_to_index",
        ]
        for row in pq.read_table(path, columns=columns).to_pylist():
            result[int(row["episode_index"])] = row
    return result


def _rows(root: Path, source: dict, sidecar_record: dict, kind: str) -> list[dict]:
    path = root / "data" / f"chunk-{int(source['data/chunk_index']):03d}" / f"file-{int(source['data/file_index']):03d}.parquet"
    if kind == "molmo":
        columns = ["episode_index", "index", "state", "actions"]
    elif kind == "droid":
        columns = [
            "episode_index", "index", "observation.state.cartesian_position",
            "observation.state.gripper_position", "action.cartesian_position",
            "action.gripper_position",
        ]
    else:
        columns = ["episode_index", "index", "observation.state", "action", "action.valid"]
    table = pq.read_table(path, columns=columns)
    file_start = int(table.column("index")[0].as_py())
    global_start = int(sidecar_record["dataset_from_index"])
    length = int(sidecar_record["dataset_to_index"]) - global_start
    episode_table = table.slice(global_start - file_start, length)
    if len(episode_table) != length:
        raise ValueError(f"episode slice mismatch for {path}")
    values = episode_table.to_pylist()
    if not values or any(int(row["episode_index"]) != int(sidecar_record["episode_index"]) for row in values):
        raise ValueError(f"episode identity mismatch for {path}")
    return values


def _canonical(rows: list[dict], kind: str):
    if kind == "molmo":
        return canonical_molmo_arrays(
            np.asarray([row["state"] for row in rows]),
            np.asarray([row["actions"] for row in rows]),
        )
    if kind == "droid":
        return canonical_droid_arrays(
            np.asarray([row["observation.state.cartesian_position"] for row in rows]),
            np.asarray([row["observation.state.gripper_position"] for row in rows]),
            np.asarray([row["action.cartesian_position"] for row in rows]),
            np.asarray([row["action.gripper_position"] for row in rows]),
        )
    return canonical_rh20t_arrays(
        np.asarray([row["observation.state"] for row in rows]),
        np.asarray([row["action"] for row in rows]),
        np.asarray([row["action.valid"] for row in rows]),
    )


def _summarize(errors: np.ndarray) -> dict:
    absolute = np.abs(errors)
    return {
        "pairs": int(len(errors)),
        "mae_by_dimension": absolute.mean(axis=0).tolist(),
        "p95_abs_error_by_dimension": np.quantile(absolute, 0.95, axis=0).tolist(),
        "rmse_by_dimension": np.sqrt(np.mean(errors**2, axis=0)).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sidecar-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes-per-dataset", type=int, default=3)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(4, 3, figsize=(15, 12), constrained_layout=True)
    report = {
        "format_version": 1,
        "method": (
            "deterministic episode sample; command-minus-observed diagnostics are not "
            "hard robot tracking-error eligibility thresholds"
        ),
        "datasets": {},
    }
    for dataset_row, (name, (root_value, kind)) in enumerate(DATASETS.items()):
        root = Path(root_value)
        sidecar = args.sidecar_root / "joint" / name
        manifest = load_stage05_sidecar(sidecar, verify_source=True)
        records = pq.read_table(sidecar / "episodes.parquet").to_pylist()
        candidates = [record for record in records if int(record["joint_action_eligible_frames"]) > 1]
        positions = np.linspace(0, len(candidates) - 1, args.episodes_per_dataset, dtype=int)
        selected = [candidates[int(position)] for position in positions]
        source_episodes = _source_episodes(root)
        all_errors = []
        gripper_current_matches = 0
        gripper_next_matches = 0
        gripper_pairs = 0
        plot_payload = None
        selected_ids = []
        for record in selected:
            episode = int(record["episode_index"])
            selected_ids.append(episode)
            rows = _rows(root, source_episodes[episode], record, kind)
            states, actions, state_valid, action_valid = _canonical(rows, kind)
            observed = states[1:, :6] - states[:-1, :6]
            observed[:, 3:6] = wrap_angles(observed[:, 3:6])
            valid = state_valid[:-1] & state_valid[1:] & action_valid[:-1]
            if valid.any():
                errors = actions[:-1, :6][valid] - observed[valid]
                errors[:, 3:6] = wrap_angles(errors[:, 3:6])
                all_errors.append(errors)
                current_gripper = (states[:-1, 6] >= 0.5).astype(np.float32)
                next_gripper = (states[1:, 6] >= 0.5).astype(np.float32)
                gripper_current_matches += int(
                    (actions[:-1, 6][valid] == current_gripper[valid]).sum()
                )
                gripper_next_matches += int(
                    (actions[:-1, 6][valid] == next_gripper[valid]).sum()
                )
                gripper_pairs += int(valid.sum())
            if plot_payload is None:
                plot_payload = (actions[:-1, :3], observed[:, :3], valid)
        errors = np.concatenate(all_errors)
        summary = _summarize(errors)
        summary.update(
            {
                "sampled_episode_ids": selected_ids,
                "gripper_action_vs_current_state_class_matches": gripper_current_matches,
                "gripper_action_vs_current_state_class_rate": (
                    gripper_current_matches / gripper_pairs if gripper_pairs else None
                ),
                "gripper_action_vs_next_state_class_matches": gripper_next_matches,
                "gripper_action_vs_next_state_class_rate": (
                    gripper_next_matches / gripper_pairs if gripper_pairs else None
                ),
                "gripper_pairs": gripper_pairs,
                "sidecar_manifest_content_hash": manifest["content_hash"],
            }
        )
        report["datasets"][name] = summary
        command, observed, valid = plot_payload
        limit = min(len(command), 160)
        for dimension, label in enumerate(("x", "y", "z")):
            axis = axes[dataset_row, dimension]
            steps = np.arange(limit)
            axis.plot(steps, command[:limit, dimension], label="canonical action", linewidth=1)
            axis.plot(steps, observed[:limit, dimension], label="next-current", linewidth=1)
            invalid = steps[~valid[:limit]]
            if len(invalid):
                axis.scatter(invalid, np.zeros(len(invalid)), marker="x", s=10, label="invalid")
            axis.set_title(f"{name}: d{label} (m)")
            axis.grid(alpha=0.2)
            if dataset_row == 0 and dimension == 0:
                axis.legend(fontsize=8)

    plot_path = args.output_dir / "canonical_trajectory_comparison.png"
    figure.savefig(plot_path, dpi=160)
    plt.close(figure)
    report["plot"] = str(plot_path)
    report["plot_sha256"] = sha256_file(plot_path)
    report["content_hash"] = hashlib.sha256(
        json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    report_path = args.output_dir / "canonical_trajectory_audit.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(report_path), "content_hash": report["content_hash"]}))


if __name__ == "__main__":
    main()
