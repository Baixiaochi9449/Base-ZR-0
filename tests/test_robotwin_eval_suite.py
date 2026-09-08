import json
import os
import subprocess

import pytest
import yaml

from scripts.prepare_robotwin_eval_suite import ROOT, assign_workers, build_plan, prepare_suite, render_runner

CONFIG = ROOT / "configs/robotwin_eval_50x2x20.json"


@pytest.fixture
def suite_source(tmp_path):
    root = tmp_path / "source"
    config = json.loads(CONFIG.read_text())
    assert config["expected_task_count"] == 50
    config.update(expected_task_count=3, checkpoint="checkpoint", statistics="stats.json",
                  experiment_doc="experiment.md")
    limits = {"adjust_bottle": 400, "open_microwave": 1500, "put_bottles_dustbin": 1700}
    robotwin = root / "evaluation/RoboTwin"
    sources = {
        root / "checkpoint/action_expert_config.json": json.dumps({"action_horizon": 16}),
        root / "stats.json": "{}",
        root / "experiment.md": "Evaluation fixture\n",
        robotwin / "task_config/_eval_step_limit.yml": yaml.safe_dump(limits),
        robotwin / "policy/ZR0/deploy_policy.yml": (
            ROOT / "evaluation/RoboTwin/policy/ZR0/deploy_policy.yml").read_text(),
        robotwin / "policy/ZR0/deploy_policy.py": "",
        robotwin / "script/eval_policy_client.py": "",
    }
    for profile in config["task_configs"]:
        sources[robotwin / f"task_config/{profile}.yml"] = "{}"
    for task in limits:
        sources[robotwin / f"envs/{task}.py"] = ""
        sources[robotwin / f"description/task_instruction/{task}.json"] = "{}"
    for path, content in sources.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    return root, config, limits


def test_matrix_uses_all_task_limits_and_both_profiles(suite_source):
    root, config, limits = suite_source
    plan = build_plan(config, root)
    assert plan["task_count"] == 3 and plan["group_count"] == 6
    assert plan["total_policy_rollouts"] == 120
    assert plan["policy_rollouts_per_profile"] == {"demo_clean": 60, "demo_randomized": 60}
    assert plan["initial_candidate_seed"] == 100000 and plan["expert_check"]
    assert len({job["id"] for job in plan["jobs"]}) == 6
    for profile in ("demo_clean", "demo_randomized"):
        configs = [job["config"] for job in plan["jobs"] if job["config"]["task_config"] == profile]
        assert {cfg["task_name"] for cfg in configs} == set(limits)
        for cfg in configs:
            assert cfg["n_episodes"] == 20 and cfg["n_action_steps"] == 16
            assert cfg["max_episode_steps"] == limits[cfg["task_name"]]


@pytest.mark.parametrize("override, message", [
    ({"n_episodes": 0}, "positive integer"),
    ({"n_episodes": True}, "positive integer"),
    ({"n_action_steps": 17}, "action_horizon"),
    ({"seed": -1}, "nonnegative"),
    ({"max_episode_steps": 400}, "Unsupported client overrides"),
])
def test_invalid_counts_and_global_step_override_rejected(suite_source, override, message):
    root, config, _ = suite_source
    config["client"].update(override)
    with pytest.raises(ValueError, match=message):
        build_plan(config, root)


def test_task_count_drift_and_duplicate_profiles_rejected(suite_source):
    root, config, _ = suite_source
    config["expected_task_count"] = 2
    with pytest.raises(ValueError, match="contains 3"):
        build_plan(config, root)
    config["expected_task_count"] = 3
    config["task_configs"] = ["demo_clean", "demo_clean"]
    with pytest.raises(ValueError, match="unique"):
        build_plan(config, root)


def test_preparation_writes_complete_matrix_without_starting_run(tmp_path, suite_source, monkeypatch):
    root, config, _ = suite_source
    config_path = root / "suite.json"
    config_path.write_text(json.dumps(config))
    monkeypatch.setattr(subprocess, "check_output", lambda *args, **kwargs: "fixture-git-state\n")
    output = tmp_path / "suite with spaces"
    plan = prepare_suite(config_path, output, root)
    configs = list((output / "configs").glob("*.yml"))
    assert len(configs) == 6
    assert sum(yaml.safe_load(path.read_text())["n_episodes"] for path in configs) == 120
    assert json.loads((output / "plan.json").read_text()) == plan
    assert (output / "experiment.md").is_file() and not (output / "logs").exists()
    assert subprocess.run(["bash", "-n", str(output / "run_all.sh")]).returncode == 0
    before = {path.name: path.read_bytes() for path in configs}
    with pytest.raises(FileExistsError):
        prepare_suite(config_path, output, root)
    assert {path.name: path.read_bytes() for path in configs} == before


@pytest.mark.parametrize("fail_first", [False, True])
def test_runner_routes_configs_in_order_and_preserves_failures(tmp_path, fail_first):
    robotwin = tmp_path / "RoboTwin root"
    (robotwin / "script").mkdir(parents=True)
    (robotwin / "script/eval_policy_client.py").write_text(
        "import os, sys\n"
        "print(os.getcwd(), sys.argv[1:], flush=True)\n"
        "sys.exit(7 if os.environ['FAIL_FIRST'] == '1' and 'first.yml' in sys.argv[-1] else 0)\n"
    )
    output = tmp_path / "output with spaces"
    output.mkdir()
    (output / "experiment.md").write_text("Prepared\n")
    script = output / "run_all.sh"
    script.write_text(render_runner({"jobs": [{"id": "first"}, {"id": "second"}]}, robotwin, output))
    env = {**os.environ, "CONDA_DEFAULT_ENV": "RoboTwin", "CUDA_VISIBLE_DEVICES": "",
           "FAIL_FIRST": str(int(fail_first))}
    missing_gpu = subprocess.run(["bash", str(script)], env=env, capture_output=True, text=True)
    assert missing_gpu.returncode != 0 and not (output / "logs").exists()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["CONDA_DEFAULT_ENV"] = "base"
    wrong_env = subprocess.run(["bash", str(script)], env=env, capture_output=True, text=True)
    assert wrong_env.returncode == 2 and not (output / "logs").exists()
    env["CONDA_DEFAULT_ENV"] = "RoboTwin"
    result = subprocess.run(["bash", str(script)], env=env, capture_output=True, text=True, timeout=20)
    assert result.returncode == (7 if fail_first else 0), result.stderr
    assert str(robotwin) in (output / "logs/first.log").read_text()
    assert str(output / "configs/first.yml") in (output / "logs/first.log").read_text()
    assert (output / "logs/second.log").exists() is not fail_first
    assert f"exit code: {result.returncode}" in (output / "experiment.md").read_text()
    previous = (output / "logs/first.log").read_bytes()
    repeated = subprocess.run(["bash", str(script)], env=env, capture_output=True, timeout=20)
    assert repeated.returncode != 0 and (output / "logs/first.log").read_bytes() == previous


def test_parallel_workers_cover_each_group_once_with_matching_ports(suite_source):
    root, config, _ = suite_source
    plan = build_plan(config, root)
    original = {job["id"]: dict(job["config"]) for job in plan["jobs"]}
    assign_workers(plan, [0, 1, 2, 3])
    jobs = [job for worker in plan["workers"] for job in worker["jobs"]]
    assert len(jobs) == len({job["id"] for job in jobs}) == 6
    assert [worker["port"] for worker in plan["workers"]] == [8022, 8023, 8024, 8025]
    for worker in plan["workers"]:
        for job in worker["jobs"]:
            expected = {**original[job["id"]], "port": worker["port"],
                        "ckpt_setting": original[job["id"]]["ckpt_setting"] + "_p4"}
            assert job["config"] == expected


@pytest.mark.parametrize("gpus", [[], [0, 0], [-1], [True]])
def test_invalid_gpu_assignments_rejected(suite_source, gpus):
    root, config, _ = suite_source
    with pytest.raises(ValueError, match="unique nonnegative"):
        assign_workers(build_plan(config, root), gpus)


def test_parallel_preparation_is_inactive_and_workers_use_shared_configs(tmp_path, suite_source, monkeypatch):
    root, config, _ = suite_source
    source = root / "suite.json"
    source.write_text(json.dumps(config))
    monkeypatch.setattr(subprocess, "check_output", lambda *args, **kwargs: "fixture\n")
    output = tmp_path / "parallel"
    plan = prepare_suite(source, output, root, gpus=[0, 1, 2, 3])
    assert not (output / "run_all.sh").exists()
    assert plan["execution"] == "parallel_independent_servers"
    assert subprocess.run(["bash", "-n", str(output / "run_parallel.sh")]).returncode == 0
    for worker in plan["workers"]:
        directory = output / "workers" / f"gpu{worker['gpu']}"
        assert (directory / "configs").resolve() == output / "configs"
        assert not (directory / "server.log").exists()
        assert (directory / "experiment.md").exists()
        assert subprocess.run(["bash", "-n", str(directory / "run_all.sh")]).returncode == 0
