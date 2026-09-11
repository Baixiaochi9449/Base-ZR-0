"""Explicitly queue a pinned Stage 3 resume after successful LIBERO completion."""

import argparse
import copy
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time

from stage3_queue_checks import live_processes, predecessor_status, read_events, verify_startup

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/three_stage_formal_stage3_resume14000_after_libero.json"
ENTRY = ROOT / "scripts/train_vla_loss_weight_resume.py"
ADAPTER = ROOT / "utils/resume_loss_weights.py"


def replace_arg(command, name, value):
    flag = "--" + name
    if flag in command:
        command[command.index(flag) + 1] = str(value)
    else:
        command.extend((flag, str(value)))


def load_adapter():
    spec = importlib.util.spec_from_file_location("zr0_resume_loss_weights", ADAPTER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Queue:
    def __init__(self, config):
        self.config = config.resolve()
        self.settings = json.loads(self.config.read_text())
        cfg = self.settings
        if (cfg.get("version") != 1 or cfg.get("enabled", False) is not True or not cfg.get("authorization")
                or cfg["runtime_commit"] != "3eefb602417d3bd4b20bef2b47660b404aefb565"
                or cfg["resume_step"] != 14000 or cfg["max_train_steps"] != 150000
                or cfg["outer_weights"] != dict(ar=1.0, slot=0.1, optical_flow=1.0, fm=5.0)
                or cfg["predecessor"]["target_updates"] != 34184 or cfg["startup_verify_updates"] != 20
                or not 1 <= cfg["predecessor"]["poll_seconds"] <= 60):
            raise ValueError("configuration differs from the authorized continuation")
        self.runtime = Path(cfg["runtime_root"])
        self.output = Path(cfg["output_root"])
        self.source = Path(cfg["resume_from_checkpoint"])
        self.previous = Path(cfg["source_experiment"])
        self.predecessor = Path(cfg["predecessor"]["output_root"])
        if self.output.exists() and self.output.resolve() in (self.previous.resolve(), self.predecessor.resolve()):
            raise ValueError("successor output must be independent")
        sys.path[:0] = [str(self.runtime), str(self.runtime / "lerobot")]
        from scripts import watch_three_stage_formal as recovery
        from scripts.run_three_stage_formal import write_json
        from utils.three_stage_preflight import implementation_identity
        import watch_libero_finetune as libero
        if Path(recovery.__file__).resolve() != self.runtime / "scripts/watch_three_stage_formal.py":
            raise ValueError("recovery imported from an unpinned runtime")
        self.recovery, self.write, self.runtime_identity, self.libero = recovery, write_json, implementation_identity, libero
        self.restore = load_adapter().install_loss_weight_overrides(
            slot_weight=cfg["outer_weights"]["slot"], optical_flow_weight=cfg["outer_weights"]["optical_flow"])

    def identity(self):
        files = [Path(__file__), ROOT / "scripts/stage3_queue_checks.py", ENTRY, ADAPTER, self.config,
                 ROOT / "scripts/watch_libero_finetune.py", ROOT / "scripts/train_libero_finetune.py",
                 self.previous / "launch_plan.json", self.previous / "code_identity.json",
                 self.source.parent / "checkpoint_complete.json"]
        return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}

    def check_identity(self):
        saved = json.loads((self.output / "code_identity.json").read_text())
        if saved["runtime"] != self.runtime_identity() or saved["queue"] != self.identity():
            raise ValueError("pinned runtime, queue or source configuration changed")
        for name, digest in saved["prepared_files"].items():
            if hashlib.sha256((self.output / name).read_bytes()).hexdigest() != digest:
                raise ValueError(f"prepared artifact changed: {name}")
        if (self.output / "retry_disabled").exists():
            raise KeyboardInterrupt("operator disabled successor monitoring/training")

    def check_source(self):
        receipt = json.loads((self.source.parent / "checkpoint_complete.json").read_text())
        inventory = json.loads(json.dumps(self.recovery.file_inventory(self.source)))
        if (receipt["step"] != self.settings["resume_step"] or receipt["stage"] != "stage3_joint"
                or inventory != receipt["files"]
                or self.recovery.validate_checkpoint(self.source, "stage3_joint", 150000) != 14000):
            raise ValueError("original step-14000 archive is incomplete or changed")
        return inventory

    def build_plan(self):
        old_identity = json.loads((self.previous / "code_identity.json").read_text())["runtime"]
        if old_identity != self.runtime_identity():
            raise ValueError("runtime differs from the previous Stage 3 experiment")
        plan = copy.deepcopy(json.loads((self.previous / "launch_plan.json").read_text()))
        item = next(item for item in plan["stages"] if item["stage"] == "stage3_joint")
        old = copy.deepcopy(item["options"])
        if old["max_train_steps"] != 150000 or item["updates"] != 150000:
            raise ValueError("the original scheduler budget changed")
        name = self.settings["experiment"] + "-stage3_joint"
        values = dict(vlm_loss_weight=1.0, slot_loss_weight=0.1, optical_flow_loss_weight=1.0,
                      action_expert_loss_weight=5.0, wandb_project=self.settings["wandb_project"],
                      wandb_group=self.settings["experiment"])
        for key, value in values.items():
            replace_arg(item["command"], key, value)
        for flag in ("--verify_resume_state", "--verify_three_stage_initialization"):
            if flag not in item["command"]:
                item["command"].append(flag)
        for key in ("wandb_run_name", "wandb_run_id"):
            item["options"][key] = name
        item = self.recovery.attempt_item(item, self.output / "stage3_joint", 0, checkpoint=self.source)
        for key in ("wandb_run_name", "wandb_run_id"):
            replace_arg(item["command"], key, name)
            item["options"][key] = name
        allowed = set(values) | {"output_ckpt_dir", "tensorboard_log_dir", "wandb_dir", "wandb_run_name",
            "wandb_run_id", "wandb_resume", "vlm_name_or_path", "action_expert_config_path",
            "resume_from_checkpoint", "action_expert_name_or_path", "verify_resume_state",
            "verify_three_stage_initialization"}
        changes = {key: [old.get(key), value] for key, value in item["options"].items() if old.get(key) != value}
        if set(changes) - allowed or not item["options"]["resume_training"] or item["options"]["init_from_checkpoint"]:
            raise ValueError(f"unexpected non-weight resume changes: {set(changes) - allowed}")
        item["resume_step"] = self.settings["resume_step"]
        plan["stages"] = [item]
        plan["formal"].update(output_root=str(self.output), experiment=self.settings["experiment"],
            wandb_project=self.settings["wandb_project"], authorization=self.settings["authorization"],
            stage3_outer_weights=self.settings["outer_weights"])
        plan["formal"]["lambda_optical_flow"]["stage3_joint"] = 1.0
        plan["resume_option_changes"] = changes
        return plan

    def entry_command(self, item):
        command = list(item["command"])
        command[command.index(str(self.runtime / "train_vla.py"))] = str(ENTRY)
        command.extend(("--loss-weight-resume-config", str(self.config)))
        return command

    def emit(self, **values):
        event = dict(time=datetime.now(timezone.utc).isoformat(), **values)
        with (self.output / "queue_events.jsonl").open("a") as stream:
            stream.write(json.dumps(event, sort_keys=True) + "\n")
        with (self.output / "experiment.md").open("a") as stream:
            stream.write("\nQueue event: `" + json.dumps(event, sort_keys=True) + "`\n")
        print(json.dumps(event, sort_keys=True), flush=True)

    def prepare(self):
        plan = self.build_plan()
        inventory = self.check_source()
        supervisor = self.recovery.process_identity(self.settings["predecessor"]["supervisor_pid"])
        if supervisor:
            args = (Path("/proc") / str(supervisor["pid"]) / "cmdline").read_bytes().split(b"\0")
            if not any(arg.endswith(b"watch_libero_finetune.py") for arg in args):
                raise ValueError("predecessor PID no longer belongs to the LIBERO supervisor")
        self.output.mkdir(parents=True, exist_ok=False)
        stage = self.output / "stage3_joint"
        stage.mkdir()
        self.write(self.output / "launch_plan.json", plan)
        policy = json.loads((self.previous / "retry_policy.json").read_text())
        if policy["max_retries_per_stage"] != 3 or policy["retry_delays_seconds"] != [60, 120, 240]:
            raise ValueError("unexpected original retry policy")
        policy.update(formal_config=str(self.config), authorization=self.settings["authorization"])
        self.write(self.output / "retry_policy.json", policy)
        self.write(stage / "expanded_options.json", plan["stages"][0]["options"])
        self.write(self.output / "source_inventory_verified.json", dict(checkpoint=str(self.source),
            step=14000, files=inventory, source_payloads_reaudited=False, diagnostic_optimizer_updates=0))
        self.write(self.output / "predecessor_identity.json", dict(supervisor=supervisor))
        template = ROOT / "docs/experiments/stage3_resume14000_after_libero/experiment.md"
        document = template.read_text() + "\nActual expanded first-attempt command:\n\n```bash\n" + shlex.join(
            self.entry_command(self.recovery.attempt_item(plan["stages"][0], stage / "retries/attempt-000", 0,
                                                        checkpoint=self.source))) + "\n```\n"
        for directory in (self.output, stage):
            (directory / "experiment.md").write_text(document)
        names = ("launch_plan.json", "retry_policy.json", "source_inventory_verified.json", "predecessor_identity.json",
                 "stage3_joint/expanded_options.json")
        self.write(self.output / "code_identity.json", dict(runtime=self.runtime_identity(), queue=self.identity(),
            prepared_files={name: hashlib.sha256((self.output / name).read_bytes()).hexdigest() for name in names}))
        self.emit(status="prepared", resume_step=14000, target=150000, stages=["stage3_joint"],
                  source=str(self.source), data_audit_reused=True)

    def wait_for_predecessor(self):
        identity = json.loads((self.output / "predecessor_identity.json").read_text())["supervisor"]
        previous_status = None
        while True:
            self.check_identity()
            events = read_events(self.predecessor / "supervisor_events.jsonl")
            alive = bool(identity and self.recovery.process_identity(identity["pid"]) == identity)
            status = predecessor_status(events, self.settings["predecessor"]["target_updates"],
                disabled=(self.predecessor / "retry_disabled").exists(), supervisor_alive=alive)
            sessions = {row["pid"] for row in events if row.get("status") == "running"}
            processes = live_processes(self.recovery.process_identity, identities=[identity], sessions=sessions,
                                       output_root=self.predecessor)
            if status == "ready" and processes:
                status = "waiting_for_process_cleanup"
            if status != previous_status:
                self.emit(status=status, predecessor=str(self.predecessor), remaining_processes=processes)
                previous_status = status
            if status == "ready":
                checkpoint = self.libero.latest_complete(self.predecessor)
                if (checkpoint is None or checkpoint.resolve() != Path(events[-1]["checkpoint"]).resolve()
                        or self.libero.validate_resume(checkpoint) != self.settings["predecessor"]["target_updates"]):
                    raise ValueError("final sealed LIBERO checkpoint failed validation")
                self.write(self.output / "predecessor_complete_verified.json", dict(
                    completion=events[-1], checkpoint=str(checkpoint), files=self.libero.inventory(checkpoint),
                    all_training_processes_exited=True, diagnostic_optimizer_updates=0))
                return
            time.sleep(self.settings["predecessor"]["poll_seconds"])

    def run_training(self):
        self.check_identity()
        self.check_source()
        queue, recovery = self, self.recovery
        native_attempt = recovery.attempt_item

        def attempt(*args, **kwargs):
            item = native_attempt(*args, **kwargs)
            if not item["options"]["resume_from_checkpoint"]:
                raise ValueError("Stage 3 queue cannot initialize fresh")
            item["command"] = queue.entry_command(item)
            return item

        class Supervisor(recovery.RecoverySupervisor):
            def check_runtime(self):
                super().check_runtime()
                queue.check_identity()

            def preserve(self, item, run):
                metric = recovery.last_metric(run)
                if metric and (metric.get("wandb_remote_enabled") != 1
                               or metric.get("wandb_remote_available") != 1
                               or metric.get("wandb_consecutive_failures", 0)):
                    raise KeyboardInterrupt("required online W&B failed")
                evidence = run / "startup_verified.json"
                if metric and not evidence.exists():
                    try:
                        result = verify_startup(run, queue.settings["outer_weights"], queue.settings["startup_verify_updates"])
                        if result:
                            queue.write(evidence, result)
                            queue.emit(status="startup_verified", output_dir=str(run), evidence=str(evidence),
                                       step=result["last_metrics"]["step"], wandb=result["wandb"])
                    except Exception as error:
                        raise KeyboardInterrupt(f"resume startup verification failed: {error}") from error
                super().preserve(item, run)

            def emit(self, **values):
                if values.get("status") == "retry_stage_complete":
                    run = Path(values["output_dir"])
                    self.preserve(self.plan["stages"][0], run)
                    step = values["completed_updates"]
                    sealed = self.archive / "stage3_joint" / f"step-{step:06d}-{run.name}" / recovery.TAG
                    if not sealed.exists():
                        raise RuntimeError("final Stage 3 checkpoint was not sealed")
                    values["immutable_checkpoint"] = str(sealed)
                super().emit(**values)
                if values.get("status") == "retry_failed" and values.get("output_dir"):
                    log = Path(values["output_dir"]) / "train.log"
                    if log.is_file() and queue.libero.wandb_failure(log):
                        raise KeyboardInterrupt("required online W&B failed; no unrecorded retry")

        plan = json.loads((self.output / "launch_plan.json").read_text())
        policy = json.loads((self.output / "retry_policy.json").read_text())
        recovery.attempt_item = attempt
        try:
            Supervisor(policy, plan).recover(recovery.read_events(self.output))
        finally:
            recovery.attempt_item = native_attempt

    def run(self):
        def stop(signum, frame):
            raise KeyboardInterrupt(f"queue received signal {signum}")
        signal.signal(signal.SIGTERM, stop)
        with (self.output / "queue.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                if (self.output / "handoff_started.json").exists():
                    raise RuntimeError("successor already dispatched; refusing duplicate training")
                self.wait_for_predecessor()
                self.check_identity()
                self.write(self.output / "handoff_started.json", dict(time=datetime.now(timezone.utc).isoformat(),
                    identity=self.recovery.process_identity(os.getpid())))
                self.emit(status="starting_stage3", resume_step=14000, target=150000)
                self.run_training()
                self.emit(status="complete", target=150000)
            except BaseException as error:
                self.emit(status="stopped", error=repr(error))
                raise

    def launch(self):
        self.check_identity()
        with (self.output / "launch.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            path = self.output / "detached_launch.json"
            prior = json.loads(path.read_text()) if path.exists() else None
            if prior and prior["identity"] and self.recovery.process_identity(prior["pid"]) == prior["identity"]:
                print(json.dumps(dict(status="already_running", **prior)), flush=True)
                return
            if (self.output / "handoff_started.json").exists():
                raise RuntimeError("successor has already been dispatched; inspect its recorded state")
            if prior:
                path.rename(self.output / f"detached_launch.previous-{time.time_ns()}.json")
            command = [sys.executable, "-u", str(Path(__file__).resolve()), "--config", str(self.config), "--mode", "run"]
            environment = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONNOUSERSITE": "1",
                "PYTHONPATH": f"{self.runtime}:{self.runtime / 'lerobot'}", "OMP_NUM_THREADS": "4",
                "MKL_NUM_THREADS": "4", "WANDB_MODE": "online", "TOKENIZERS_PARALLELISM": "false"}
            with (self.output / "queue.log").open("a") as stream:
                child = subprocess.Popen(command, cwd=self.runtime, env=environment, stdin=subprocess.DEVNULL,
                    stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            receipt = dict(pid=child.pid, identity=self.recovery.process_identity(child.pid), command=command)
            self.write(path, receipt)
            print(json.dumps(receipt), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--mode", choices=("check", "prepare", "launch", "run"), default="check")
    args = parser.parse_args()
    queue = Queue(args.config)
    try:
        if args.mode == "check":
            plan = queue.build_plan()
            inventory = queue.check_source()
            print(json.dumps(dict(status="ready", source_files=len(inventory), stages=plan["stages"],
                                  option_changes=plan["resume_option_changes"]), indent=2))
        else:
            getattr(queue, args.mode)()
    finally:
        queue.restore()


if __name__ == "__main__":
    main()
