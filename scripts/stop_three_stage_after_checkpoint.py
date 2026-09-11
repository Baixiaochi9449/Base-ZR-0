"""Stop an explicitly selected run after its complete checkpoint is archived."""

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "outputs/runtime_snapshots/3eefb602417d3bd4b20bef2b47660b404aefb565"
sys.path[:0] = [str(RUNTIME), str(RUNTIME / "lerobot")]
from scripts.watch_three_stage_formal import file_inventory, last_metric, process_identity, validate_checkpoint


def members(groups):
    result = []
    for path in Path("/proc").iterdir():
        if not path.name.isdigit():
            continue
        try:
            identity = process_identity(int(path.name))
            if identity and identity["pgrp"] in groups:
                result.append(identity)
        except (ProcessLookupError, PermissionError, FileNotFoundError):
            continue
    return result


def inspect(output):
    supervisor = json.loads((output / "detached_launch.json").read_text())["identity"]
    events = [json.loads(line) for line in (output / "formal_events.jsonl").read_text().splitlines()]
    launched = next(row for row in reversed(events) if row.get("status") == "retry_running")
    launcher, run = launched["identity"], Path(launched["output_dir"])
    for identity in (supervisor, launcher):
        if not identity or process_identity(identity["pid"]) != identity:
            raise RuntimeError("selected supervisor/launcher is no longer running")
    ranks = []
    for path in Path("/proc").iterdir():
        if not path.name.isdigit():
            continue
        try:
            fields = (path / "stat").read_text().rsplit(") ", 1)[1].split()
            if int(fields[1]) != launcher["pid"]:
                continue
            args = (path / "cmdline").read_bytes()
            if b"train_vla_loss_weight_resume.py" not in args:
                continue
            identity = process_identity(int(path.name))
            if identity and os.readlink(path / "fd/1") == str(run / "train.log"):
                ranks.append(identity)
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    if len(ranks) != 4:
        raise RuntimeError(f"expected four identified training ranks, found {len(ranks)}")
    return supervisor, launcher, ranks, run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target", type=int, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    output, target = args.output.resolve(), args.target
    if target <= 0:
        raise ValueError("target must be positive")
    supervisor, launcher, ranks, run = inspect(output)
    identities = [supervisor, launcher, *ranks]
    prefix = f"stop_{target}"

    def write(name, data):
        with (output / name).open("x") as stream:
            json.dump(data, stream, indent=2)
            stream.write("\n")

    def emit(status, **values):
        event = dict(status=status, time=datetime.now(timezone.utc).isoformat(), **values)
        with (output / f"{prefix}_events.jsonl").open("a") as stream:
            stream.write(json.dumps(event) + "\n")
        print(json.dumps(event), flush=True)

    if not args.execute:
        print(json.dumps(dict(status="ready", step=last_metric(run)["step"], target=target,
                              run=str(run), identities=identities)), flush=True)
        return
    with (output / f"{prefix}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write(f"{prefix}_started.json", dict(pid=os.getpid(), target=target, identities=identities,
            run=str(run), authorization=f"User requested stopping after complete checkpoint {target}."))
        try:
            report = 0
            while True:
                if any(process_identity(value["pid"]) != value for value in identities):
                    raise RuntimeError("training identity changed; no other process will be signaled")
                metric = last_metric(run)
                if metric and metric["step"] >= target:
                    # The production log commits this metric after all save barriers.
                    for rank in ranks:
                        os.kill(rank["pid"], signal.SIGSTOP)
                    emit("ranks_paused_after_save", step=last_metric(run)["step"], target=target)
                    break
                if time.monotonic() - report >= 30:
                    emit("waiting_for_save", step=metric["step"] if metric else None, target=target)
                    report = time.monotonic()
                time.sleep(0.05 if metric and metric["step"] >= target - 3 else 1)
            archive = output / "recovery_checkpoints/stage3_joint" / f"step-{target:06d}-{run.name}"
            deadline = time.monotonic() + 900
            while not (archive / "checkpoint_complete.json").exists():
                if process_identity(supervisor["pid"]) != supervisor or time.monotonic() > deadline:
                    raise RuntimeError("archive not complete; ranks remain paused for inspection")
                time.sleep(0.25)
            checkpoint = archive / "latest-model-optimizer-lr"
            receipt = json.loads((archive / "checkpoint_complete.json").read_text())
            inventory = json.loads(json.dumps(file_inventory(checkpoint)))
            if (receipt["step"] != target or receipt["files"] != inventory
                    or validate_checkpoint(checkpoint, "stage3_joint", 150000) != target):
                raise RuntimeError("archived checkpoint validation failed; ranks remain paused")
            emit("checkpoint_verified", step=target, checkpoint=str(checkpoint), files=len(inventory))
            groups = {identity["pgrp"] for identity in [launcher, *ranks]}
            owned = members(groups)
            if not (output / "retry_disabled").exists():
                write("retry_disabled", dict(reason=f"User requested stop after checkpoint {target}",
                    step=target, checkpoint=str(checkpoint), time=datetime.now(timezone.utc).isoformat()))
            for identity in identities:
                if process_identity(identity["pid"]) == identity:
                    try:
                        os.kill(identity["pid"], signal.SIGTERM)
                    except ProcessLookupError:
                        pass
            for group in groups:
                try:
                    os.killpg(group, signal.SIGTERM)
                    os.killpg(group, signal.SIGCONT)
                except ProcessLookupError:
                    pass
            deadline = time.monotonic() + 35
            while members(groups) or process_identity(supervisor["pid"]) == supervisor:
                if time.monotonic() > deadline:
                    raise RuntimeError("checkpoint safe but owned processes remain after SIGTERM")
                time.sleep(0.25)
            result = dict(status="stopped", time=datetime.now(timezone.utc).isoformat(),
                checkpoint_step=target, checkpoint=str(checkpoint), file_count=len(inventory),
                last_logged_step=last_metric(run)["step"], retries_disabled=True,
                process_groups_empty=True, identities=identities, stopped_group_members=owned,
                unlogged_inflight_update_not_proven_absent=True)
            write(f"{prefix}_completed.json", result)
            for directory in (output, run):
                with (directory / "experiment.md").open("a") as stream:
                    stream.write("\nOperator-requested stop: `" + json.dumps(result) + "`\n")
            emit("stopped", checkpoint=str(checkpoint), last_logged_step=result["last_logged_step"])
        except BaseException as error:
            emit("monitor_failed", error=repr(error))
            raise


if __name__ == "__main__":
    main()
