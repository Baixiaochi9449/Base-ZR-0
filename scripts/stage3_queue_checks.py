"""CPU-only completion and startup checks for the opt-in training queue."""

import json
import math
from pathlib import Path


def read_events(path):
    if not path.exists():
        return []
    data = path.read_bytes()
    # A writer may still be appending the last JSONL record.
    return [json.loads(line) for line in data.splitlines(keepends=True) if line.endswith(b"\n")]


def predecessor_status(events, target, *, disabled=False, supervisor_alive=True):
    if disabled:
        raise RuntimeError("LIBERO was manually stopped; successor is not authorized to start on failure")
    if not events:
        raise RuntimeError("LIBERO has no supervisor events")
    last = events[-1]
    if last.get("status") == "complete":
        if last.get("step") != target or last.get("exit_code") != 0 or not last.get("checkpoint"):
            raise RuntimeError("LIBERO completion requires its target, checkpoint and exit code zero")
        return "waiting_for_exit" if supervisor_alive else "ready"
    if last.get("status") == "stopped" or not supervisor_alive:
        raise RuntimeError("LIBERO supervisor stopped without normal completion")
    return "waiting_for_completion"


def live_processes(process_identity, *, identities=(), sessions=(), output_root=None):
    live = {}
    for identity in identities:
        if identity and process_identity(identity["pid"]) == identity:
            live[identity["pid"]] = identity
    for path in Path("/proc").iterdir():
        if not path.name.isdigit():
            continue
        try:
            identity = process_identity(int(path.name))
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        if not identity:
            continue
        matched = identity["session"] in sessions
        if output_root is not None:
            try:
                args = (path / "cmdline").read_bytes().split(b"\0")
                matched |= any(arg.decode(errors="replace").startswith(str(output_root)) for arg in args)
            except (FileNotFoundError, ProcessLookupError, PermissionError):
                continue
        if matched:
            live[identity["pid"]] = identity
    return list(live.values())


def verify_startup(run, weights, count=20, *, initial_sources=None):
    rows = []
    path = run / "training_metrics.jsonl"
    if not path.exists():
        return None
    with path.open() as stream:
        for line in stream:
            if not line.endswith("\n"):
                break
            row = json.loads(line)
            if row.get("optimizer_update_applied"):
                rows.append(row)
                if len(rows) == count:
                    break
    if len(rows) < count:
        return None
    states = ([] if initial_sources is not None else
              [json.loads((run / f"saved_state_verified_rank{rank}.json").read_text()) for rank in range(4)])
    source, start = (initial_sources["vlm"], 0) if initial_sources is not None else (states[0]["source"], states[0]["global_step"])
    exact = ("master_parameters_exact", "optimizer_exact", "rng_exact", "sampler_cursor_exact", "scheduler_exact")
    for rank, state in enumerate(states):
        if (state["rank"] != rank or state["source"] != source or state["global_step"] != start
                or state["optimizer_updates"] != 0 or any(state.get(key) is not True for key in exact)):
            raise ValueError("loaded rank training state did not match the checkpoint exactly")
    if states and states[0].get("exposure_exact") is not True:
        raise ValueError("loaded exposure state did not match")
    components = json.loads((run / "component_sources_verified.json").read_text())
    counts = dict(vlm=626, query=1, slot_aux=34, optical_flow_aux=35, action_expert=149)
    if set(components) != set(counts):
        raise ValueError("expected all five inherited components")
    for name, value in components.items():
        expected_source = initial_sources[name] if initial_sources is not None else source
        if (value["source"] != expected_source or value["tensors_exact"] != counts[name]
                or value["trainable_parameters"] != value["parameters"] or value["parameters"] <= 0):
            raise ValueError("component inheritance/trainability changed")
    errors = []
    groups = ("vlm", "difference_query", "slot_aux", "optical_flow_aux", "action_expert")
    loss_keys = dict(ar="ar_loss", slot="slot_loss_raw", optical_flow="optical_flow_loss", fm="flow_matching_loss")
    weighted_keys = dict(ar="weighted_ar_loss", slot="slot_loss_weighted",
                         optical_flow="weighted_optical_flow_loss", fm="weighted_flow_matching_loss")
    count_keys = dict(ar="ar_active_token_count", slot="slot_active_supervision_count",
                      optical_flow="flow_active_sample_count", fm="fm_active_element_count")
    for offset, row in enumerate(rows, 1):
        if (row["step"] != start + offset or row["scheduler_step_before"] != row["step"] - 1
                or row["scheduler_step_after"] != row["step"] or row["optimizer_step_global_samples"] != 128
                or row["optimizer_microbatches_per_rank"] != 2 or row.get("optimizer_update_skipped")):
            raise ValueError("startup update/batch/scheduler contract failed")
        if (row.get("wandb_remote_enabled") != 1 or row.get("wandb_remote_available") != 1
                or row.get("wandb_consecutive_failures", 0) or row.get("wandb_remote_abandoned", 0)):
            raise ValueError("online W&B is required")
        expected = 0.0
        for name, key in loss_keys.items():
            weighted_key = weighted_keys[name]
            if (key not in row or weighted_key not in row) and row.get(count_keys[name]) != 0:
                raise ValueError(f"missing {name} metrics without confirmed absent supervision")
            value = weights[name] * row.get(key, 0.0)
            if not math.isclose(row.get(weighted_key, 0.0), value, rel_tol=1e-5, abs_tol=1e-6):
                raise ValueError(f"incorrect {name} outer weight")
            expected += value
        errors.append(abs(expected - row["total_loss"]))
        if not math.isclose(expected, row["total_loss"], rel_tol=1e-5, abs_tol=1e-6):
            raise ValueError("incorrect total loss formula")
        for group in groups:
            if not math.isfinite(row[group + "_grad_norm"]):
                raise ValueError("nonfinite component gradient")
            if row[group + "_active_supervision"] != row[group + "_update_applied"]:
                raise ValueError("component activity/update contract failed")
    return dict(status="passed", source=source, restored_step=start, verified_updates=count,
                component_sources=components, saved_state=states, first_metrics=rows[0], last_metrics=rows[-1],
                total_loss_formula_max_abs_error=max(errors), diagnostic_optimizer_updates=0,
                peak_reserved_gib=max(row["gpu_peak_memory_reserved_gib"] for row in rows),
                wandb=json.loads((run / "wandb_identity.json").read_text()))
