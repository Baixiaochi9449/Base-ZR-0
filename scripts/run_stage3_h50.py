"""Prepare and supervise one explicitly authorized H50 Stage 3 experiment."""

import argparse
import copy
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

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/three_stage_formal_stage3_h50_20260911.json"
ENTRY = ROOT / "scripts/train_vla_loss_weight_resume.py"


def replace_arg(command, name, value):
    flag = "--" + name
    if flag in command:
        command[command.index(flag) + 1] = str(value)
    else:
        command.extend((flag, str(value)))


class Experiment:
    def __init__(self, config):
        self.config = config.resolve()
        self.settings = cfg = json.loads(self.config.read_text())
        expected = dict(version=1, enabled=True, allow_stage2_init=True, action_horizon=50,
            max_train_steps=150000, runtime_commit="3eefb602417d3bd4b20bef2b47660b404aefb565",
            outer_weights=dict(ar=1.0, slot=0.1, optical_flow=1.0, fm=5.0))
        if any(cfg.get(key) != value for key, value in expected.items()) or not cfg.get("authorization"):
            raise ValueError("configuration differs from the authorized H50 experiment")
        self.runtime = Path(cfg["runtime_root"]).resolve()
        self.output = Path(cfg["output_root"]).resolve()
        self.source = Path(cfg["init_from_checkpoint"]).resolve()
        self.previous = Path(cfg["source_experiment"]).resolve()
        self.resume_source = Path(cfg["resume_from_checkpoint"]).resolve() if cfg.get("resume_from_checkpoint") else None
        self.resume_origin = Path(cfg["resume_source_experiment"]).resolve() if self.resume_source else None
        if self.resume_source and (type(cfg.get("resume_step")) is not int or
                not 0 < cfg["resume_step"] < cfg["max_train_steps"] or
                not self.resume_source.is_relative_to(self.resume_origin / "recovery_checkpoints")):
            raise ValueError("resume requires a complete checkpoint from the named H50 experiment")
        protected_paths = [self.source, self.previous, self.runtime, Path("/opt/data/private/lq/models/ZR-0")]
        if self.resume_source:
            protected_paths.append(self.resume_source)
        for protected in protected_paths:
            if self.output.is_relative_to(protected) or protected.is_relative_to(self.output):
                raise ValueError("output must be independent of existing models and experiments")
        sys.path[:0] = [str(self.runtime), str(self.runtime / "lerobot")]
        from scripts import watch_three_stage_formal as recovery
        from utils.three_stage_preflight import implementation_identity
        if Path(recovery.__file__).resolve() != self.runtime / "scripts/watch_three_stage_formal.py":
            raise ValueError("recovery must use the pinned runtime")
        self.recovery, self.runtime_identity = recovery, implementation_identity
        spec = importlib.util.spec_from_file_location("zr0_h50_loss_adapter", ROOT / "utils/resume_loss_weights.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.restore = module.install_loss_weight_overrides(slot_weight=0.1, optical_flow_weight=1.0,
                                                           allow_stage2_init=True)

    def identity(self):
        paths = [self.config, Path(__file__), ENTRY, ROOT / "utils/resume_loss_weights.py",
                 ROOT / "utils/stage3_horizon_runtime.py", ROOT / "scripts/stage3_queue_checks.py",
                 self.previous / "launch_plan.json",
                 self.source.parent / "checkpoint_complete.json"]
        if self.resume_source:
            paths.extend((self.resume_origin / "code_identity.json", self.resume_origin / "launch_plan.json",
                          self.resume_source.parent / "checkpoint_complete.json"))
        return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}

    def build_plan(self):
        if self.resume_source:
            return self.build_resume_plan()
        cfg = self.settings
        plan = copy.deepcopy(json.loads((self.previous / "launch_plan.json").read_text()))
        item = next(item for item in plan["stages"] if item["stage"] == "stage3_joint")
        command = item["command"]
        for flag in ("--resume_from_checkpoint", "--resume_training", "--verify_resume_state"):
            if flag in command:
                i = command.index(flag)
                del command[i:i + (2 if flag == "--resume_from_checkpoint" else 1)]
        run = self.output / "stage3_joint"
        name = cfg["experiment"] + "-stage3_joint"
        values = dict(vlm_name_or_path=self.source, init_from_checkpoint=self.source,
            action_expert_name_or_path="/opt/data/private/lq/models/ZR-0",
            action_expert_config_path="/opt/data/private/lq/models/ZR-0/action_expert_config.json",
            action_horizon=50, vlm_loss_weight=1.0, slot_loss_weight=0.1,
            optical_flow_loss_weight=1.0, action_expert_loss_weight=5.0,
            output_ckpt_dir=run, tensorboard_log_dir=run / "tensorboard", wandb_dir=run / "wandb",
            wandb_group=cfg["experiment"], wandb_project=cfg["wandb_project"],
            wandb_run_name=name, wandb_run_id=name, wandb_resume="never")
        for key, value in values.items():
            replace_arg(command, key, value)
        from utils.cli_options import parse_train_options
        options = vars(parse_train_options(command[command.index(str(self.runtime / "train_vla.py")) + 1:]))
        if (options["resume_training"] or options["max_train_steps"] != 150000
                or options["save_step_interval"] != 2000 or options["warmup_ratio"] != 0.08
                or not options["verify_three_stage_initialization"]):
            raise ValueError("fresh Stage 3 schedule/initialization mismatch")
        item.update(output_dir=str(run), options=options)
        item.pop("resume_step", None)
        plan["stages"] = [item]
        plan["formal"].update(output_root=str(self.output), experiment=cfg["experiment"],
            stage_updates=[150000], authorization=cfg["authorization"],
            stage3_outer_weights=cfg["outer_weights"], action_horizon=50)
        plan["formal"].pop("lambda_optical_flow", None)
        return plan

    def build_resume_plan(self):
        plan = copy.deepcopy(json.loads((self.resume_origin / "launch_plan.json").read_text()))
        if len(plan["stages"]) != 1 or plan["stages"][0]["stage"] != "stage3_joint":
            raise ValueError("resume origin must be the single-stage H50 experiment")
        item = plan["stages"][0]
        old = copy.deepcopy(item["options"])
        if old["action_horizon"] != 50 or old["max_train_steps"] != 150000 or item["updates"] != 150000:
            raise ValueError("resume cannot change horizon or the original scheduler budget")
        item = self.recovery.attempt_item(item, self.output / "stage3_joint", 0, checkpoint=self.resume_source)
        command = item["command"]
        for key in ("wandb_run_name", "wandb_run_id"):
            replace_arg(command, key, self.settings["experiment"] + "-stage3_joint")
        if "--verify_resume_state" not in command:
            command.append("--verify_resume_state")
        from utils.cli_options import parse_train_options
        options = vars(parse_train_options(command[command.index(str(self.runtime / "train_vla.py")) + 1:]))
        allowed = {"vlm_name_or_path", "action_expert_name_or_path", "action_expert_config_path",
                   "resume_from_checkpoint", "resume_training", "init_from_checkpoint", "verify_resume_state",
                   "output_ckpt_dir", "tensorboard_log_dir", "wandb_dir", "wandb_run_name", "wandb_run_id"}
        changes = {key: [old.get(key), value] for key, value in options.items() if old.get(key) != value}
        if set(changes) - allowed or not options["resume_training"] or options["init_from_checkpoint"]:
            raise ValueError(f"resume changed training hyperparameters: {set(changes) - allowed}")
        item.update(options=options, resume_step=self.settings["resume_step"])
        plan["stages"] = [item]
        plan["formal"].update(output_root=str(self.output), experiment=self.settings["experiment"],
                              authorization=self.settings["authorization"])
        plan["resume_option_changes"] = changes
        return plan

    def check_resume_source(self):
        saved = json.loads((self.resume_origin / "code_identity.json").read_text())
        if saved["runtime"] != self.runtime_identity():
            raise ValueError("resume runtime differs from the original H50 experiment")
        for path, digest in {**saved["adapters"], **saved["prepared"]}.items():
            if Path(path).resolve() == Path(__file__).resolve():
                # Only this orchestration file changes to add the explicit resume entry.
                # Its prior sealed bytes must still be present in the recorded commit.
                payload = subprocess.check_output(["git", "show",
                    self.settings["resume_source_launcher_commit"] + ":scripts/run_stage3_h50.py"], cwd=ROOT)
            else:
                payload = Path(path).read_bytes()
            if hashlib.sha256(payload).hexdigest() != digest:
                raise ValueError(f"original H50 source identity changed: {path}")
        receipt = json.loads((self.resume_source.parent / "checkpoint_complete.json").read_text())
        inventory = json.loads(json.dumps(self.recovery.file_inventory(self.resume_source)))
        step = self.recovery.validate_checkpoint(self.resume_source, "stage3_joint", 150000)
        if receipt["files"] != inventory or receipt["stage"] != "stage3_joint" or step != receipt["step"] or step != self.settings["resume_step"]:
            raise ValueError("requested H50 resume checkpoint is incomplete or changed")
        return dict(checkpoint=str(self.resume_source), step=step, files=inventory,
                    source_payloads_reaudited=False, diagnostic_optimizer_updates=0,
                    original_runtime_and_training_adapters_unchanged=True)

    def command(self, item):
        command = list(item["command"])
        command[command.index(str(self.runtime / "train_vla.py"))] = str(ENTRY)
        return [*command, "--loss-weight-resume-config", str(self.config)]

    def check_sources(self, plan, *, check_preparation=True):
        from utils.three_stage_preflight import validate_preparation
        from utils.action_expert_config import load_action_expert_config
        if self.resume_source:
            return self.check_resume_source()
        if check_preparation:
            validate_preparation(plan["preparation"])
        if self.recovery.validate_checkpoint(self.source, "stage2_aux", 5000) != 5000:
            raise ValueError("Stage 2 source must be the complete step-5000 checkpoint")
        receipt = json.loads((self.source.parent / "checkpoint_complete.json").read_text())
        inventory = json.loads(json.dumps(self.recovery.file_inventory(self.source)))
        if receipt["files"] != inventory or receipt["stage"] != "stage2_aux" or receipt["step"] != 5000:
            raise ValueError("preserved Stage 2 checkpoint inventory changed")
        expert = load_action_expert_config("/opt/data/private/lq/models/ZR-0/action_expert_config.json",
            expected_action_horizon=50, expected_action_dim=64, expected_state_dim=64,
            expected_vlm_hidden_size=2048)
        return dict(checkpoint=str(self.source), step=5000, files=inventory,
            expert_config_sha256=expert.source_sha256, expert_source_horizon=expert.source_action_horizon,
            source_payloads_reaudited=False, diagnostic_optimizer_updates=0)

    def prepare(self):
        plan = self.build_plan()
        evidence = self.check_sources(plan)
        self.output.mkdir(parents=True, exist_ok=False)
        run = Path(plan["stages"][0]["output_dir"])
        run.mkdir()
        write = self.recovery.write_json
        write(self.output / "launch_plan.json", plan)
        write(self.output / "source_inventory_verified.json", evidence)
        write(run / "expanded_options.json", plan["stages"][0]["options"])
        policy = json.loads((self.previous / "retry_policy.json").read_text())
        policy.update(formal_config=str(self.config), authorization=self.settings["authorization"])
        write(self.output / "retry_policy.json", policy)
        template = (Path(self.settings["experiment_template"]) if self.resume_source else
                    ROOT / "docs/experiments/stage3_h50_20260911/experiment.md")
        document = template.read_text()
        document += "\nExpanded initial command (attempt paths are recorded before each launch):\n\n```bash\n"
        document += shlex.join(self.command(plan["stages"][0])) + "\n```\n"
        for directory in (self.output, run):
            (directory / "experiment.md").write_text(document)
        prepared = [self.output / "launch_plan.json", self.output / "source_inventory_verified.json",
                    self.output / "retry_policy.json", run / "expanded_options.json"]
        write(self.output / "code_identity.json", dict(runtime=self.runtime_identity(), adapters=self.identity(),
            prepared={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in prepared}))
        print(json.dumps(dict(status="prepared", output=str(self.output), source=evidence["checkpoint"])), flush=True)

    def check_identity(self):
        identity = json.loads((self.output / "code_identity.json").read_text())
        if identity["runtime"] != self.runtime_identity() or identity["adapters"] != self.identity():
            raise ValueError("sealed runtime or adapters changed")
        if any(hashlib.sha256(Path(p).read_bytes()).hexdigest() != h for p, h in identity["prepared"].items()):
            raise ValueError("sealed experiment artifact changed")

    def run(self):
        self.check_identity()
        plan = self.build_plan()
        if plan != json.loads((self.output / "launch_plan.json").read_text()):
            raise ValueError("prepared plan changed")
        self.check_sources(plan, check_preparation=False)
        recovery = self.recovery
        original_attempt = recovery.attempt_item

        def attempt(*args, **kwargs):
            item = original_attempt(*args, **kwargs)
            if owner.resume_source and kwargs.get("checkpoint") is None:
                raise ValueError("explicit H50 resume must never fall back to fresh initialization")
            if kwargs.get("checkpoint") is not None and "--verify_resume_state" not in item["command"]:
                item["command"].append("--verify_resume_state")
                item["options"]["verify_resume_state"] = True
            item["command"] = self.command(item)
            return item

        recovery.attempt_item = attempt
        owner = self

        class Supervisor(recovery.RecoverySupervisor):
            def check_runtime(self):
                super().check_runtime()
                owner.check_identity()

            def preserve(self, item, run):
                from stage3_queue_checks import verify_startup
                metric = recovery.last_metric(run)
                if metric and (metric.get("wandb_remote_enabled") != 1 or
                               metric.get("wandb_remote_available") != 1 or
                               metric.get("wandb_consecutive_failures", 0)):
                    raise KeyboardInterrupt("required online W&B failed")
                evidence = Path(run) / "startup_verified.json"
                if not evidence.exists() and metric is not None:
                    options = json.loads((Path(run) / "expanded_options.json").read_text())
                    sources = None if options["resume_training"] else {
                        name: str(owner.source) for name in ("vlm", "query", "slot_aux", "optical_flow_aux")}
                    if sources is not None:
                        sources["action_expert"] = "/opt/data/private/lq/models/ZR-0"
                    try:
                        result = verify_startup(Path(run), owner.settings["outer_weights"], initial_sources=sources)
                    except Exception as error:
                        raise KeyboardInterrupt(f"H50 startup verification failed: {error}") from error
                    if result is not None:
                        recovery.write_json(evidence, result)
                        self.emit(status="startup_verified", output_dir=str(run),
                            verified_updates=20, peak_reserved_gib=result["peak_reserved_gib"],
                            wandb=result["wandb"])
                super().preserve(item, run)

        policy = json.loads((self.output / "retry_policy.json").read_text())
        supervisor = Supervisor(policy, plan)
        def terminate(signum, frame):
            raise KeyboardInterrupt(f"H50 supervisor received signal {signum}")
        signal.signal(signal.SIGTERM, terminate)
        with (self.output / "retry_watch.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            recovery.write_json(self.output / "formal_started.json", dict(pid=os.getpid(),
                identity=recovery.process_identity(os.getpid()), target=150000,
                initial_step=self.settings["resume_step"] if self.resume_source else 0))
            try:
                supervisor.recover([])
                for event in recovery.read_events(self.output):
                    if event.get("status") == "retry_stage_complete":
                        supervisor.preserve(plan["stages"][0], Path(event["output_dir"]))
            except BaseException as error:
                supervisor.emit(status="stopped", error=repr(error))
                raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--mode", choices=("check", "prepare", "run"), default="check")
    args = parser.parse_args()
    experiment = Experiment(args.config)
    try:
        if args.mode == "check":
            plan = experiment.build_plan()
            print(json.dumps(experiment.check_sources(plan), indent=2))
        else:
            getattr(experiment, args.mode)()
    finally:
        experiment.restore()


if __name__ == "__main__":
    main()
