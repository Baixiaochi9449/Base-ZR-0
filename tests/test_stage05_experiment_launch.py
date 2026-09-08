import json
from types import SimpleNamespace

import pytest
from accelerate.data_loader import BatchSamplerShard

from scripts.stage05_experiment_train import ProbeBatchSampler
from scripts import run_stage05_h10_experiment as runner


@pytest.mark.parametrize("micro,gas", [(64, 1), (16, 4), (1, 64)])
def test_fixed_probe_fixture_survives_production_rank_sharding(micro, gas):
    original = SimpleNamespace(batch_size=micro, num_processes=4)
    samples = [11, 22, 33, 44, 55]
    sampler = ProbeBatchSampler(original, samples, gas)
    shards = [list(BatchSamplerShard(sampler, num_processes=4, process_index=rank,
                                    split_batches=False, even_batches=False)) for rank in range(4)]
    assert all(len(shard) == 3 * gas for shard in shards)
    assert all(len(batch) == micro for shard in shards for batch in shard)
    assert all(set(index for batch in shard for index in batch) == set(samples) for shard in shards)
    assert 4 * micro * gas == 256


@pytest.mark.parametrize("phase", ["ar", "joint"])
@pytest.mark.parametrize("processor_audits", [False, True])
def test_resume_preflight_keeps_experiment_cuda_visible(tmp_path, monkeypatch, phase, processor_audits):
    run_dir = tmp_path / "probe"
    checkpoint = run_dir / "latest-model-optimizer-lr"
    checkpoint.mkdir(parents=True)
    (run_dir / "training_metrics.jsonl").write_text(json.dumps({"step": 2, "total_loss": 1.0}) + "\n")
    for name in ("data_seen_state.npz", "resolved_dataset_manifest.json", "scheduler.pt"):
        (checkpoint / name).touch()
    snapshot = run_dir / "step-2"
    snapshot.mkdir()
    (snapshot / "zr0_checkpoint_metadata.json").touch()
    calls = []
    def gate(output, env):
        calls.append("resources")
        return {"cuda_visible_devices": "GPU-a,GPU-b,GPU-c,GPU-d"}
    monkeypatch.setattr(runner, "resource_gate", gate)
    monkeypatch.setattr(runner, "record", lambda *args, **kwargs: None)

    def preflight(command, *, cwd, env, check):
        assert calls == ["resources"]
        assert env["CUDA_VISIBLE_DEVICES"] == "GPU-a,GPU-b,GPU-c,GPU-d"
        assert env["PYTHONNOUSERSITE"] == "1"
        assert command[command.index("--purpose") + 1] == f"stage05_{phase}_resume"
        assert "--validate-resume-artifacts" in command
        assert check is True
        calls.append("preflight")

    monkeypatch.setattr(runner.subprocess, "run", preflight)
    env = {"CUDA_VISIBLE_DEVICES": "", "PYTHONNOUSERSITE": "1",
           "ZR0_ACTION_EXPERT_CONFIG_PATH": "/experiment/action_expert_config.json",
           "ZR0_ACTION_HORIZON": "10"}
    if processor_audits:
        env["ZR0_SELECT_PROCESSOR_AUDIT"] = "1"

    def select(output, processor_path, selected_env):
        assert calls == ["resources", "preflight"]
        assert processor_path == checkpoint
        calls.append("processor_audit")
        return selected_env

    monkeypatch.setattr(runner, "select_token_audit", select)
    assert runner.verify_run(tmp_path, run_dir, 2, phase, env) == checkpoint
    assert calls == ["resources", "preflight"] + (["processor_audit"] if processor_audits else [])
    assert env["CUDA_VISIBLE_DEVICES"] == ""


@pytest.mark.parametrize("resume_fails", [False, True])
@pytest.mark.parametrize("processor_audits", [False, True])
@pytest.mark.parametrize("continuation_point", ["resume", "formal"])
def test_ar_gate_continuation_skips_probes_and_stops_on_failure(tmp_path, monkeypatch, resume_fails, processor_audits, continuation_point):
    import yaml

    config = json.loads((runner.ROOT / "configs/four_dataset_dq32_h10_gbs256_seed42_20260906.json").read_text())
    config["output_root"] = str(tmp_path)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    (tmp_path / "preparation_complete.json").write_text("{}")
    (tmp_path / "data_summary.json").write_text(json.dumps({"ar_steps": 27582, "ar_warmup_steps": 1379,
                                                          "joint_steps": 95423, "joint_warmup_steps": 4771}))
    (tmp_path / "runner_started.json").write_text('{"pid": 1}')
    identity = {"train_vla.py": "unchanged", "scripts/run_stage05_h10_experiment.py": "corrected"}
    (tmp_path / "source_identity.json").write_text(json.dumps({**identity, "scripts/run_stage05_h10_experiment.py": "original"}))
    original_identity = (tmp_path / "source_identity.json").read_bytes()
    (tmp_path / "lifecycle.jsonl").write_text(json.dumps({"stage": "ar-smoke", "status": "exited", "exit_code": 0,
                                                       "output_dir": str(tmp_path / "probes/ar/mbs32_gas2")}) + "\n")
    accelerate = yaml.safe_load((runner.ROOT / "accelerate_configs/accelerate_config.yaml").read_text())
    accelerate["deepspeed_config"].update(gradient_accumulation_steps=2, train_micro_batch_size_per_gpu=32,
                                         train_batch_size=256, gradient_clipping=1.0)
    (tmp_path / "ar_mbs32_gas2.yaml").write_text(yaml.safe_dump(accelerate))
    (tmp_path / "ar_probe_samples.json").write_text("[]")
    formal_ready = continuation_point == "formal"
    argv = ["runner", "--config", str(config_path), "--continue-ar-formal" if formal_ready else "--continue-ar-resume-gate"]
    if formal_ready:
        completed = str(tmp_path / "probes/ar/completed_recovery")
        with (tmp_path / "lifecycle.jsonl").open("a") as stream:
            stream.write(json.dumps({"stage": "ar-resume", "status": "exited", "exit_code": 0, "output_dir": completed}) + "\n")
            stream.write(json.dumps({"stage": "ar", "status": "checkpoint_verified", "global_step": 3, "output_dir": completed}) + "\n")
    if processor_audits:
        argv += ["--processor-audits", "--continuation-id", "processor_audit"]
        if not formal_ready:
            failed_log = tmp_path / "previous_resume.log"
            failed_log.write_text("processor/tokenizer files identity changed")
            with (tmp_path / "lifecycle.jsonl").open("a") as stream:
                stream.write(json.dumps({"stage": "ar-resume", "status": "failed", "log": str(failed_log)}) + "\n")
    monkeypatch.setattr(runner.sys, "argv", argv)
    monkeypatch.setattr(runner, "source_identity", lambda: identity)
    monkeypatch.setattr(runner, "resource_gate", lambda *args: None)
    monkeypatch.setattr(runner, "record", lambda *args, **kwargs: None)
    monkeypatch.setattr(runner.subprocess, "check_output", lambda *args, **kwargs: "fixture\n" if kwargs.get("text") else b"fixture\n")
    fixtures = []
    monkeypatch.setattr(runner, "make_fixture", lambda output, phase, audit_report=None:
                        fixtures.append(phase) or output / "joint_samples.json")
    monkeypatch.setattr(runner, "select_token_audit", lambda output, source, env:
                        {**env, "ZR0_TOKEN_LENGTH_AUDIT": str(output / "selected_report.json")})
    checks = []
    monkeypatch.setattr(runner, "verify_run", lambda output, run_dir, step, phase, env:
                        checks.append((phase, step)) or run_dir / "latest-model-optimizer-lr")
    launches = []

    def launch(output, stage, env, **kwargs):
        launches.append((stage, env))
        assert env["CUDA_VISIBLE_DEVICES"] == "0,1,2,3"
        if stage == "ar-resume" and processor_audits:
            assert env["ZR0_INITIAL_CHECKPOINT"] == str(tmp_path / "probes/ar/mbs32_gas2/latest-model-optimizer-lr")
            assert env["ZR0_RUN_OUTPUT_DIR"] == str(tmp_path / "probes/ar/mbs32_gas2_resume_processor_audit")
        if resume_fails and stage == ("ar-formal" if formal_ready else "ar-resume"):
            raise RuntimeError("recovery failed")

    monkeypatch.setattr(runner, "launch", launch)
    if resume_fails:
        with pytest.raises(RuntimeError, match="recovery failed"):
            runner.main()
        assert [stage for stage, env in launches] == ["ar-formal" if formal_ready else "ar-resume"]
        assert checks == [("ar", 3 if formal_ready else 2)]
        assert fixtures == []
    else:
        runner.main()
        assert [stage for stage, env in launches] == ([] if formal_ready else ["ar-resume"]) + ["ar-formal", "joint-smoke", "joint-resume", "joint-formal"]
        assert checks == ([] if formal_ready else [("ar", 2)]) + [("ar", 3), ("ar", 27582), ("joint", 2), ("joint", 3), ("joint", 95423)]
        assert fixtures == ["joint"]
        ar_formal = next(env for stage, env in launches if stage == "ar-formal")
        assert ar_formal["ZR0_INITIAL_CHECKPOINT"] == config["base_model"]
        assert "ZR0_PROBE_SAMPLES" not in ar_formal
        assert (ar_formal["ZR0_PER_DEVICE_BATCH_SIZE"], ar_formal["ZR0_GRADIENT_ACCUMULATION_STEPS"]) == ("32", "2")
        assert launches[-1][1]["ZR0_INITIAL_CHECKPOINT"] == str(tmp_path / "formal/ar/latest-model-optimizer-lr")
    assert (tmp_path / "runner_started.json").read_text() == '{"pid": 1}'
    assert (tmp_path / "source_identity.json").read_bytes() == original_identity
    with pytest.raises((FileExistsError, BlockingIOError)):
        runner.main()
