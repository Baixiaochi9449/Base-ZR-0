#!/usr/bin/env python3
"""Generate a RoboTwin evaluation matrix without importing the simulator/model."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shlex
import shutil
import subprocess

import yaml

ROOT = Path(__file__).resolve().parents[1]


def positive_int(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def build_plan(config, root=ROOT):
    robotwin = root / "evaluation/RoboTwin"
    template_path = robotwin / "policy/ZR0/deploy_policy.yml"
    limits_path = robotwin / "task_config/_eval_step_limit.yml"
    template = yaml.safe_load(template_path.read_text())
    limits = yaml.safe_load(limits_path.read_text())
    expected = positive_int(config["expected_task_count"], "expected_task_count")
    if len(limits) != expected:
        raise ValueError(f"Expected {expected} tasks; step-limit table contains {len(limits)}")
    profiles = config["task_configs"]
    if (not profiles or len(set(profiles)) != len(profiles)
            or any(name not in {"demo_clean", "demo_randomized"} for name in profiles)):
        raise ValueError("task_configs must contain unique Clean/Random profile names")
    client = config["client"]
    allowed = set(template) - {"task_name", "task_config", "max_episode_steps"}
    if set(client) - allowed:
        raise ValueError(f"Unsupported client overrides: {sorted(set(client) - allowed)}")
    base = {**template, **client}
    episodes = positive_int(base["n_episodes"], "n_episodes")
    steps = positive_int(base["n_action_steps"], "n_action_steps")
    if type(base["seed"]) is not int or base["seed"] < 0:
        raise ValueError("seed must be a nonnegative integer")
    if positive_int(base["port"], "port") > 65535:
        raise ValueError("port must be at most 65535")
    if base["instruction_type"] not in {"seen", "unseen"}:
        raise ValueError("instruction_type must be seen or unseen")
    if base["policy_name"] != "ZR0":
        raise ValueError("This suite uses the ZR0 policy client")
    for name in (config["experiment"], base["ckpt_setting"]):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
            raise ValueError(f"Invalid experiment/result label: {name!r}")
    checkpoint = (root / config["checkpoint"]).resolve()
    expert_path = checkpoint / "action_expert_config.json"
    expert = json.loads(expert_path.read_text())
    if steps > expert["action_horizon"]:
        raise ValueError("n_action_steps exceeds checkpoint action_horizon")

    sources = [template_path, limits_path, expert_path, root / config["statistics"],
               robotwin / "script/eval_policy_client.py",
               robotwin / "policy/ZR0/deploy_policy.py"]
    jobs = []
    for profile in profiles:
        sources.append(robotwin / f"task_config/{profile}.yml")
        for task, limit in sorted(limits.items()):
            if not re.fullmatch(r"[a-z0-9_]+", task):
                raise ValueError(f"Invalid task name: {task!r}")
            for source in (robotwin / f"envs/{task}.py",
                           robotwin / f"description/task_instruction/{task}.json"):
                if not source.is_file():
                    raise FileNotFoundError(source)
            cfg = {**base, "task_name": task, "task_config": profile,
                   "max_episode_steps": positive_int(limit, f"{task} step limit")}
            jobs.append({"id": f"{profile}__{task}", "config": cfg})
    return {
        "experiment": config["experiment"], "status": "prepared_not_started",
        "checkpoint": str(checkpoint), "dataset_entry": config["dataset_entry"],
        "checkpoint_statistics_match_verified": config["checkpoint_statistics_match_verified"],
        "task_count": len(limits), "group_count": len(jobs),
        "episodes_per_group": episodes, "total_policy_rollouts": episodes * len(jobs),
        "policy_rollouts_per_profile": {name: episodes * len(limits) for name in profiles},
        "initial_candidate_seed": 100000 * (1 + base["seed"]),
        "expert_check": True, "execution": "sequential",
        "source_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in sources},
        "jobs": jobs,
    }


def render_runner(plan, robotwin, output):
    lines = [
        "#!/usr/bin/env bash", "set -euo pipefail",
        ': "${CUDA_VISIBLE_DEVICES:?Select the GPU for RoboTwin simulation}"',
        'if [[ "${CONDA_DEFAULT_ENV:-}" != "RoboTwin" ]]; then',
        '    echo "Activate the RoboTwin Conda environment before running this script." >&2',
        "    exit 2", "fi",
        f"cd {shlex.quote(str(robotwin))}",
        f"RUN_DIR={shlex.quote(str(output))}",
        '# Refuse a second run in the same directory to preserve existing logs.',
        'mkdir "$RUN_DIR/logs"',
        'printf "\\n## Runtime\\n\\nStarted (UTC): %s\\n" "$(date -u +%FT%TZ)" >> "$RUN_DIR/experiment.md"',
        'trap \'status=$?; printf "Finished (UTC): %s; exit code: %s\\n" "$(date -u +%FT%TZ)" "$status" >> "$RUN_DIR/experiment.md"\' EXIT',
    ]
    for job in plan["jobs"]:
        command = shlex.join(["python", "-u", "script/eval_policy_client.py", "--config",
                              str(output / "configs" / f"{job['id']}.yml")])
        log = shlex.quote(str(output / "logs" / f"{job['id']}.log"))
        lines.append(f"{command} 2>&1 | tee {log}")
    return "\n".join(lines) + "\n"


def assign_workers(plan, gpus):
    if (not gpus or len(set(gpus)) != len(gpus)
            or any(type(gpu) is not int or gpu < 0 for gpu in gpus)):
        raise ValueError("gpus must be unique nonnegative integer indices")
    base_port = plan["jobs"][0]["config"]["port"]
    if base_port + len(gpus) - 1 > 65535:
        raise ValueError("worker ports exceed 65535")
    workers = [{"gpu": gpu, "port": base_port + index, "jobs": []}
               for index, gpu in enumerate(gpus)]
    for index, job in enumerate(plan["jobs"]):
        worker = workers[index % len(workers)]
        job["config"]["port"] = worker["port"]
        job["config"]["ckpt_setting"] += f"_p{len(workers)}"
        worker["jobs"].append(job)
    plan["execution"] = "parallel_independent_servers"
    plan["workers"] = workers
    plan["server_seed_per_worker"] = 42


def prepare_suite(config_path, output, root=ROOT, gpus=None):
    config_path = config_path.resolve()
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}; choose a new directory")
    config = json.loads(config_path.read_text())
    plan = build_plan(config, root)
    if gpus is not None:
        assign_workers(plan, gpus)
    document = (root / config["experiment_doc"]).read_text()
    plan.update({
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "git_status": subprocess.check_output(["git", "status", "--short"], cwd=root, text=True),
        "output": str(output),
    })
    (output / "configs").mkdir(parents=True)
    for job in plan["jobs"]:
        (output / "configs" / f"{job['id']}.yml").write_text(
            yaml.safe_dump(job["config"], sort_keys=False))
    shutil.copy2(config_path, output / "requested_config.json")
    (output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    (output / "experiment.md").write_text(
        document + f"\nPrepared output: `{output}`. Actual plan: `plan.json`.\n")
    if gpus is None:
        (output / "run_all.sh").write_text(render_runner(plan, root / "evaluation/RoboTwin", output))
    else:
        commands = ["#!/usr/bin/env bash", "set -uo pipefail", "pids=()"]
        for worker in plan["workers"]:
            directory = output / "workers" / f"gpu{worker['gpu']}"
            directory.mkdir(parents=True)
            (directory / "configs").symlink_to(output / "configs", target_is_directory=True)
            (directory / "experiment.md").write_text(
                document + f"\nWorker GPU {worker['gpu']}, port {worker['port']}, "
                f"{len(worker['jobs'])} groups. Parent plan: `{output / 'plan.json'}`.\n")
            (directory / "run_all.sh").write_text(
                render_runner(worker, root / "evaluation/RoboTwin", directory))
            command = ["bash", str(root / "scripts/run_robotwin_eval_worker.sh"),
                       str(directory), str(worker["gpu"]), str(worker["port"]), plan["checkpoint"]]
            (directory / "launch.sh").write_text("#!/usr/bin/env bash\nexec " + shlex.join(command) + "\n")
            commands.extend([shlex.join(command) + f" > {shlex.quote(str(directory / 'worker.log'))} 2>&1 &",
                             'pids+=("$!")'])
        commands.extend(["status=0", 'for pid in "${pids[@]}"; do',
                         '    wait "$pid" || status=1', "done", 'exit "$status"'])
        (output / "run_parallel.sh").write_text("\n".join(commands) + "\n")
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/robotwin_eval_50x2x20.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpus", type=int, nargs="+", help="Optional independent GPU workers; default is serial")
    args = parser.parse_args()
    plan = prepare_suite(args.config, args.output, gpus=args.gpus)
    print(json.dumps({key: plan[key] for key in
                      ("status", "task_count", "group_count", "total_policy_rollouts", "output")}, indent=2))


if __name__ == "__main__":
    main()
