"""Bounded, fail-closed GPU checks. A passed check is not a device reservation."""

import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time


@dataclass(frozen=True)
class GPUResourcePolicy:
    poll_seconds: float = 2.0
    consecutive_samples: int = 3
    timeout_seconds: float = 60.0
    min_free_mib: int = 71680
    max_used_mib: int = 1024
    max_utilization: int = 0
    query_timeout_seconds: float = 5.0

    @classmethod
    def from_env(cls, env=None):
        env = os.environ if env is None else env
        defaults = cls()
        return cls(**{name: type(value)(env.get("ZR0_GPU_GATE_" + name.upper(), value))
                      for name, value in asdict(defaults).items()}).validate()

    def validate(self):
        for name, value in asdict(self).items():
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"invalid GPU gate {name}")
        if min(self.poll_seconds, self.timeout_seconds, self.query_timeout_seconds, self.consecutive_samples) <= 0:
            raise ValueError("GPU gate intervals and consecutive_samples must be positive")
        if type(self.consecutive_samples) is not int or self.max_utilization > 100:
            raise ValueError("invalid GPU gate sample count or utilization")
        return self


class GPUResourceError(RuntimeError):
    pass


def cuda_device_identities():
    """Driver enumeration only, in a fresh process with the launch environment."""
    import ctypes
    import uuid

    driver = ctypes.CDLL("libcuda.so.1")

    def checked(name, *args):
        status = getattr(driver, name)(*args)
        if status:
            raise GPUResourceError(f"CUDA identity query {name} failed: {status}")

    checked("cuInit", 0)
    count = ctypes.c_int()
    checked("cuDeviceGetCount", ctypes.byref(count))
    devices = []
    for ordinal in range(count.value):
        device = ctypes.c_int()
        checked("cuDeviceGet", ctypes.byref(device), ordinal)
        identity = (ctypes.c_ubyte * 16)()
        checked("cuDeviceGetUuid_v2" if hasattr(driver, "cuDeviceGetUuid_v2") else "cuDeviceGetUuid",
                ctypes.byref(identity), device)
        bus = ctypes.create_string_buffer(64)
        checked("cuDeviceGetPCIBusId", bus, len(bus), device)
        devices.append({"cuda_ordinal": ordinal, "uuid": "GPU-" + str(uuid.UUID(bytes=bytes(identity))),
                        "pci_bus_id": bus.value.decode()})
    return devices


def resolve_visible_devices(env, timeout):
    result = subprocess.run(
        [sys.executable, "-c", "import json; from utils.gpu_resource_gate import cuda_device_identities; "
         "print(json.dumps(cuda_device_identities()))"],
        env=env, cwd=Path(__file__).resolve().parents[1], text=True, capture_output=True, check=True, timeout=timeout,
    )
    return json.loads(result.stdout)


def query_gpu_snapshot(timeout):
    deadline = time.monotonic() + timeout

    def query(option, fields):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("GPU query deadline exceeded")
        result = subprocess.run(
            ["nvidia-smi", f"--query-{option}={fields}", "--format=csv,noheader,nounits"],
            text=True, capture_output=True, check=True, timeout=remaining,
        )
        return [[value.strip() for value in row] for row in csv.reader(result.stdout.splitlines()) if row]

    rows = query("gpu", "index,uuid,pci.bus_id,memory.total,memory.used,memory.free,utilization.gpu,mig.mode.current")
    gpus = {}
    for row in rows:
        if len(row) != 8:
            raise GPUResourceError(f"malformed GPU query row: {row}")
        index, identity, bus, total, used, free, utilization, mig = row
        if not identity.startswith("GPU-") or identity in gpus:
            raise GPUResourceError("unknown or duplicate GPU identity")
        gpus[identity] = dict(index=int(index), uuid=identity, pci_bus_id=bus,
                             total_mib=int(total), used_mib=int(used), free_mib=int(free),
                             utilization=int(utilization), mig_mode=mig, processes=[])
        numeric = [gpus[identity][key] for key in ("total_mib", "used_mib", "free_mib", "utilization")]
        if min(numeric) < 0 or not 0 <= int(utilization) <= 100 or int(total) <= 0:
            raise GPUResourceError("invalid GPU query values")
    if not gpus:
        raise GPUResourceError("GPU inventory query returned no devices")
    for row in query("compute-apps", "gpu_uuid,pid,process_name,used_memory"):
        if len(row) != 4 or row[0] not in gpus:
            raise GPUResourceError(f"unresolved compute process query row: {row}")
        identity, pid, name, memory = row
        process = dict(pid=int(pid), name=name, used_mib=int(memory))
        if process["pid"] <= 0 or process["used_mib"] < 0:
            raise GPUResourceError(f"invalid compute process: {row}")
        gpus[identity]["processes"].append(process)
    return gpus


def owned_process_state(children, process_groups):
    states = [{"pid": child.pid, "exit_code": child.poll()} for child in children]
    groups = []
    for group in process_groups:
        try:
            os.killpg(group, 0)
        except ProcessLookupError:
            continue
        # Permission errors propagate: an unobservable group is not idle.
        groups.append(group)
    return {"children": states, "live_process_groups": groups}


def wait_for_gpus(*, env=None, expected_count=4, policy=None, log_path=None,
                  children=(), process_groups=(), resolve=resolve_visible_devices,
                  query=query_gpu_snapshot, owned_state=owned_process_state,
                  monotonic=time.monotonic, sleep=time.sleep):
    env = dict(os.environ if env is None else env)
    policy = (policy or GPUResourcePolicy.from_env(env)).validate()
    started = monotonic()
    deadline = started + policy.timeout_seconds
    gate_id = datetime.now(timezone.utc).isoformat()

    def emit(event):
        event.update(time=datetime.now(timezone.utc).isoformat(), gate_id=gate_id,
                     elapsed_seconds=monotonic() - started, policy=asdict(policy))
        if log_path is not None:
            path = Path(log_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a") as stream:
                stream.write(json.dumps(event, sort_keys=True) + "\n")
        return event

    try:
        if any(key.startswith("CUDA_MPS_") for key in env):
            raise GPUResourceError("CUDA MPS environment cannot be reliably gated")
        devices = resolve(env, min(policy.query_timeout_seconds, policy.timeout_seconds))
        identities = [device["uuid"] for device in devices]
        if len(identities) != expected_count or len(set(identities)) != expected_count:
            raise GPUResourceError(f"expected {expected_count} distinct visible CUDA devices, got {devices}")
    except Exception as error:
        event = emit(dict(gate="NO-GO", devices=[], reasons=[f"device identity unavailable: {error}"],
                          requested_cuda_visible_devices=env.get("CUDA_VISIBLE_DEVICES")))
        raise GPUResourceError(json.dumps(event)) from error

    consecutive = 0
    last_event = dict(devices=devices, gpus=[], ownership={}, reasons=[], consecutive_idle_samples=0)
    while True:
        if monotonic() >= deadline:
            event = emit({**last_event, "gate": "NO-GO", "sample_reused_on_timeout": True,
                          "last_sample_time": last_event.get("time"),
                          "reasons": last_event["reasons"] + ["GPU gate deadline reached"]})
            raise GPUResourceError("GPU resource wait timed out: " + json.dumps(event))
        reasons, selected, ownership = [], [], {}
        try:
            remaining = deadline - monotonic()
            if remaining <= 0:
                raise TimeoutError("GPU gate deadline reached")
            snapshot = query(min(policy.query_timeout_seconds, remaining))
            for device in devices:
                gpu = snapshot[device["uuid"]]
                selected.append(gpu)
                # UUID is authoritative; PCI identity independently catches mismatched inventories.
                if int(gpu["pci_bus_id"].replace(":", "").replace(".", ""), 16) != int(
                        device["pci_bus_id"].replace(":", "").replace(".", ""), 16):
                    reasons.append(f"{gpu['uuid']}: PCI identity mismatch")
                if gpu["mig_mode"] != "Disabled":
                    reasons.append(f"{gpu['uuid']}: MIG state is unsupported or unknown: {gpu['mig_mode']}")
                if gpu["processes"]:
                    reasons.append(f"{gpu['uuid']}: active compute processes")
                if gpu["free_mib"] < policy.min_free_mib or gpu["used_mib"] > policy.max_used_mib:
                    reasons.append(f"{gpu['uuid']}: memory not released or below required headroom")
                if gpu["utilization"] > policy.max_utilization:
                    reasons.append(f"{gpu['uuid']}: utilization has not settled")
            ownership = owned_state(children, process_groups)
            if ownership["live_process_groups"] or any(child["exit_code"] is None for child in ownership["children"]):
                reasons.append("owned child or process group has not exited")
        except Exception as error:
            reasons.append(f"resource query failed: {type(error).__name__}: {error}")
        consecutive = 0 if reasons else consecutive + 1
        expired = monotonic() >= deadline
        passed = consecutive >= policy.consecutive_samples and not expired
        event = emit(dict(gate="GO" if passed else "NO-GO" if expired else "WAIT",
                          devices=devices, gpus=selected, ownership=ownership, reasons=reasons,
                          consecutive_idle_samples=consecutive))
        last_event = event
        if passed:
            return {**event, "cuda_visible_devices": ",".join(identities)}
        if expired:
            raise GPUResourceError("GPU resource wait timed out: " + json.dumps(event))
        sleep(min(policy.poll_seconds, max(0, deadline - monotonic())))


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--visible-devices", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3"))
    parser.add_argument("--expected-count", type=int, default=4)
    args = parser.parse_args()
    result = wait_for_gpus(env={**os.environ, "CUDA_VISIBLE_DEVICES": args.visible_devices},
                          expected_count=args.expected_count, log_path=args.log)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
