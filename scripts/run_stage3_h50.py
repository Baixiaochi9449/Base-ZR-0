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
        for protected in (self.source, self.previous, self.runtime, Path("/opt/data/private/lq/models/ZR-0")):
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
        return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}

    def build_plan(self):
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

    def command(self, item):
        command = list(item["command"])
        command[command.index(str(self.runtime / "train_vla.py"))] = str(ENTRY)
        return [*command, "--loss-weight-resume-config", str(self.config)]

    def check_sources(self, plan, *, check_preparation=True):
        from utils.three_stage_preflight import validate_preparation
        from utils.action_expert_config import load_action_expert_config
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
        document = (ROOT / "docs/experiments/stage3_h50_20260911/experiment.md").read_text()
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
            if kwargs.get("checkpoint") is not None:
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
                evidence = Path(run) / "startup_verified.json"
                if not evidence.exists() and recovery.last_metric(run) is not None:
                    options = json.loads((Path(run) / "expanded_options.json").read_text())
                    sources = None if options["resume_training"] else {
                        name: str(owner.source) for name in ("vlm", "query", "slot_aux", "optical_flow_aux")}
                    if sources is not None:
                        sources["action_expert"] = "/opt/data/private/lq/models/ZR-0"
                    result = verify_startup(Path(run), owner.settings["outer_weights"], initial_sources=sources)
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
                identity=recovery.process_identity(os.getpid()), target=150000, initial_step=0))
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
