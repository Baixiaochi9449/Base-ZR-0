import copy
import json
import subprocess
from types import SimpleNamespace

import pytest

from utils import gpu_resource_gate as gate


DEVICES = [{"cuda_ordinal": i, "uuid": f"GPU-{i}", "pci_bus_id": f"0000:{i + 1:02x}:00.0"} for i in range(4)]


def idle():
    return {d["uuid"]: dict(d, index=i, total_mib=81920, used_mib=11, free_mib=81226,
                           utilization=0, mig_mode="Disabled", processes=[]) for i, d in enumerate(DEVICES)}


class Clock:
    now = 0.
    def time(self):
        return self.now
    def sleep(self, seconds):
        self.now += seconds


def run(tmp_path, states, **overrides):
    clock = Clock()
    samples = iter(states)
    last = states[-1]
    def query(timeout):
        value = next(samples, last)
        if isinstance(value, Exception):
            raise value
        return copy.deepcopy(value)
    options = dict(env={}, policy=gate.GPUResourcePolicy(timeout_seconds=10),
                   log_path=tmp_path / "gate.jsonl", resolve=lambda env, timeout: DEVICES,
                   query=query, owned_state=lambda children, groups: {"children": [], "live_process_groups": []},
                   monotonic=clock.time, sleep=clock.sleep)
    options.update(overrides)
    return gate.wait_for_gpus(**options)


def events(tmp_path):
    return [json.loads(line) for line in (tmp_path / "gate.jsonl").read_text().splitlines()]


def test_exited_probe_utilization_settles_before_three_confirmations(tmp_path):
    busy = idle()
    busy["GPU-0"]["utilization"] = 100
    result = run(tmp_path, [busy, busy, idle()])
    assert result["gate"] == "GO" and result["elapsed_seconds"] == 8
    assert [e["consecutive_idle_samples"] for e in events(tmp_path)] == [0, 0, 1, 2, 3]
    assert all(e["devices"] and "policy" in e and "time" in e for e in events(tmp_path))


@pytest.mark.parametrize("cause", ["other_process", "self_process", "memory", "headroom", "utilization", "mig", "missing_gpu", "pci"])
def test_persistent_occupancy_and_unknown_states_timeout(tmp_path, cause):
    state = idle()
    gpu = state["GPU-0"]
    if cause in {"other_process", "self_process"}:
        gpu["processes"] = [{"pid": 123, "name": "python", "used_mib": 1}]
    elif cause == "memory":
        gpu["used_mib"] = 1025
    elif cause == "headroom":
        gpu["free_mib"] = 71679
    elif cause == "utilization":
        gpu["utilization"] = 100
    elif cause == "mig":
        gpu["mig_mode"] = "Enabled"
    elif cause == "missing_gpu":
        state.pop("GPU-0")
    else:
        gpu["pci_bus_id"] = "0000:ff:00.0"
    with pytest.raises(gate.GPUResourceError, match="timed out"):
        run(tmp_path, [state])
    assert events(tmp_path)[-1]["gate"] == "NO-GO"
    assert not any(e["gate"] == "GO" for e in events(tmp_path))


def test_query_error_resets_consecutive_samples(tmp_path):
    run(tmp_path, [idle(), RuntimeError("process query unavailable"), idle()])
    assert [e["consecutive_idle_samples"] for e in events(tmp_path)] == [1, 0, 1, 2, 3]


def test_query_failure_never_means_empty(tmp_path):
    with pytest.raises(gate.GPUResourceError, match="query unavailable"):
        run(tmp_path, [RuntimeError("query unavailable")])


@pytest.mark.parametrize("owned", [{"children": [{"pid": 10, "exit_code": None}], "live_process_groups": []},
                                   {"children": [{"pid": 10, "exit_code": 0}], "live_process_groups": [10]}])
def test_own_launcher_and_descendants_must_exit_even_without_cuda_context(tmp_path, owned):
    with pytest.raises(gate.GPUResourceError, match="owned child"):
        run(tmp_path, [idle()], owned_state=lambda *args: owned)


def test_device_order_is_pinned_to_cuda_uuid_not_smi_index(tmp_path):
    reordered = [DEVICES[i] for i in (2, 0, 3, 1)]
    def resolve(env, timeout):
        assert env["CUDA_VISIBLE_DEVICES"] == "2,0,3,1"
        return reordered
    result = run(tmp_path, [idle()], env={"CUDA_VISIBLE_DEVICES": "2,0,3,1"}, resolve=resolve)
    assert result["cuda_visible_devices"] == "GPU-2,GPU-0,GPU-3,GPU-1"


@pytest.mark.parametrize("devices", [[], DEVICES[:3], [DEVICES[0]] * 4])
def test_ambiguous_device_selection_rejected(tmp_path, devices):
    with pytest.raises(gate.GPUResourceError, match="device identity unavailable"):
        run(tmp_path, [idle()], resolve=lambda *args: devices)


def test_mps_environment_rejected(tmp_path):
    with pytest.raises(gate.GPUResourceError, match="MPS"):
        run(tmp_path, [idle()], env={"CUDA_MPS_PIPE_DIRECTORY": "/tmp/mps"})


@pytest.mark.parametrize("response", ["N/A", "GPU-0, 12, python, N/A", "GPU-unknown, 12, python, 10"])
def test_malformed_process_query_rejected(monkeypatch, response):
    gpu = "0,GPU-0,0000:01:00.0,81920,11,81226,0,Disabled\n"
    def command(cmd, **kwargs):
        assert kwargs["check"] and kwargs["timeout"] > 0
        return SimpleNamespace(stdout=response if "compute-apps" in cmd[1] else gpu)
    monkeypatch.setattr(gate.subprocess, "run", command)
    with pytest.raises((gate.GPUResourceError, ValueError)):
        gate.query_gpu_snapshot(5)


def test_failed_nvidia_smi_propagates(monkeypatch):
    def failed(*args, **kwargs):
        raise subprocess.CalledProcessError(1, args[0], stderr="driver failed")
    monkeypatch.setattr(gate.subprocess, "run", failed)
    with pytest.raises(subprocess.CalledProcessError):
        gate.query_gpu_snapshot(5)


def test_policy_environment_and_validation():
    policy = gate.GPUResourcePolicy.from_env({"ZR0_GPU_GATE_POLL_SECONDS": "1", "ZR0_GPU_GATE_CONSECUTIVE_SAMPLES": "4"})
    assert policy.poll_seconds == 1 and policy.consecutive_samples == 4
    with pytest.raises(ValueError):
        gate.GPUResourcePolicy.from_env({"ZR0_GPU_GATE_POLL_SECONDS": "nan"})


def test_h10_launch_uses_gate_uuid_and_waits_for_child(tmp_path, monkeypatch):
    from scripts import run_stage05_h10_experiment as runner
    identity = tmp_path / "source_identity.json"
    identity.write_text("{}")
    monkeypatch.setattr(runner, "source_identity", lambda: {})
    monkeypatch.setattr(runner, "record", lambda *args, **kwargs: None)
    monkeypatch.setattr(runner, "OWNED_CHILDREN", [])
    checked = []
    def check(output, env):
        checked.append(output)
        return {"cuda_visible_devices": "GPU-2,GPU-0,GPU-3,GPU-1"}
    monkeypatch.setattr(runner, "resource_gate", check)
    child = SimpleNamespace(pid=123, wait=lambda: checked.append("wait") or 0)
    def popen(command, **kwargs):
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == kwargs["env"]["ZR0_CUDA_VISIBLE_DEVICES"] == "GPU-2,GPU-0,GPU-3,GPU-1"
        assert kwargs["start_new_session"]
        return child
    monkeypatch.setattr(runner.subprocess, "Popen", popen)
    runner.launch(tmp_path, "ar-smoke", {"ZR0_RUN_OUTPUT_DIR": str(tmp_path / "probe")})
    assert checked == [tmp_path, "wait"] and runner.OWNED_CHILDREN == [child]
