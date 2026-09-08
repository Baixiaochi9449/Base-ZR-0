#!/usr/bin/env python3
"""CPU-only real AV1 resource and production-loader diagnostics; no training."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import shlex
import subprocess
import sys
import time
from unittest.mock import patch

import av
import psutil
import torch

from lerobot.common.datasets import video_utils

ROOT = Path(__file__).resolve().parents[1]
BASELINE = "29796d94059153e332449563657b1155acce7f56"
VIDEO_SOURCE = "lerobot/lerobot/common/datasets/video_utils.py"
PARTIAL = Path("/opt/data/private/lq/datasets/droid_1.0.1_stage05_partial_9465_20260816")
FULL = Path("/opt/data/private/lq/datasets/droid_1.0.1_stage05_full_95658_20260831")


def process_snapshot(process=None):
    process = process or psutil.Process()
    return dict(pid=process.pid, rss_mib=process.memory_info().rss / 2**20,
                threads=process.num_threads(), fds=process.num_fds(), time=time.time())


def system_snapshot():
    limits = {}
    for name in ("RLIMIT_AS", "RLIMIT_RSS", "RLIMIT_NPROC", "RLIMIT_NOFILE", "RLIMIT_STACK"):
        limits[name] = resource.getrlimit(getattr(resource, name))
    cgroups = {}
    for name in ("memory/memory.limit_in_bytes", "memory/memory.usage_in_bytes", "memory/memory.failcnt",
                 "memory/memory.oom_control", "pids/pids.max", "pids/pids.current",
                 "memory.max", "memory.current", "memory.events", "pids.max", "pids.current"):
        path = Path("/sys/fs/cgroup") / name
        if path.is_file():
            cgroups[name] = path.read_text().strip()
    return dict(memory=psutil.virtual_memory()._asdict(), cpu_count=psutil.cpu_count(),
                cpu_affinity=psutil.Process().cpu_affinity(), limits=limits, cgroups=cgroups,
                cgroup_membership=Path("/proc/self/cgroup").read_text())


def baseline_decoder():
    source = subprocess.check_output(["git", "show", f"{BASELINE}:{VIDEO_SOURCE}"], cwd=ROOT, text=True)
    namespace = {"__name__": "historical_video_utils"}
    exec(compile(source, f"{BASELINE}:{VIDEO_SOURCE}", "exec"), namespace)
    return namespace["decode_video_frames"]


def video_cases():
    import pyarrow.parquet as pq

    cases = []
    for name in ("file-149.mp4", "file-274.mp4", "file-276.mp4"):
        path = PARTIAL / "videos/observation.images.exterior_1_left/chunk-000" / name
        cases.append(dict(path=str(path), timestamps=[0.], tolerance_s=.100001, case="partial_" + name))
    for path in sorted((FULL / "meta/episodes").rglob("*.parquet")):
        table = pq.read_table(path, filters=[("dataset_from_index", "<=", 2041670), ("dataset_to_index", ">", 2041670)])
        if not table.num_rows:
            continue
        row = table.to_pylist()[0]
        camera = "observation.images.exterior_1_left"
        prefix = "videos/" + camera
        video = FULL / "videos" / camera / f"chunk-{row[prefix + '/chunk_index']:03d}" / f"file-{row[prefix + '/file_index']:03d}.mp4"
        # Read the actual source timestamp, rather than deriving it from row offset.
        from utils.future_difference_audit import _load_episode_mapping
        mapping = _load_episode_mapping(FULL)[row["episode_index"]]
        source = pq.read_table(FULL / mapping["source_data_uri"], columns=["index", "episode_index", "frame_index", "timestamp"],
                               filters=[("index", "=", 2041670)]).to_pylist()
        if len(source) != 1:
            raise ValueError("historical sample identity is ambiguous")
        timestamp = float(row[prefix + "/from_timestamp"]) + source[0]["timestamp"]
        cases.append(dict(path=str(video), timestamps=[timestamp], tolerance_s=.100001,
                          case="historical_global_index_2041670", identity=source[0]))
        break
    if len(cases) != 4:
        raise ValueError("historical DROID sample could not be located")
    return cases


def write_event(output, event):
    with (output / "resources.jsonl").open("a") as stream:
        stream.write(json.dumps(event, sort_keys=True) + "\n")


def raw_audit(args, output):
    decode = baseline_decoder() if args.mode == "baseline" else video_utils.decode_video_frames
    cases = video_cases()
    metadata = []
    for case in cases:
        with av.open(case["path"], options={"threads": "1"}) as container:
            stream = container.streams.video[0]
            metadata.append(dict(case=case, codec=stream.codec_context.name, width=stream.width,
                                 height=stream.height, fps=float(stream.average_rate)))
    (output / "videos.json").write_text(json.dumps(metadata, indent=2) + "\n")
    start = process_snapshot()
    write_event(output, dict(phase="start", **start))
    reference, hashes, timings = [], [], []
    for i, case in enumerate(cases):
        queries = case["timestamps"]
        # Include non-monotonic and duplicate timestamp requests.
        for times in (queries, [queries[0] + 2 / 15, queries[0], queries[0] + 1 / 15, queries[0]]):
            before = time.monotonic()
            tensor = decode(case["path"], times, case["tolerance_s"], "pyav")
            timings.append(time.monotonic() - before)
            reference.append(tensor)
            hashes.append(hashlib.sha256(tensor.numpy().tobytes()).hexdigest())
            write_event(output, dict(phase="comparison", case=i, **process_snapshot()))
    torch.save(reference, output / "frames.pt")
    result = dict(cases=cases, frame_hashes=hashes, comparison_seconds=timings,
                  configured_threads=("legacy auto (0); environment ignored" if args.mode == "baseline"
                                      else video_utils.resolve_pyav_threads()), start=start)
    if args.reference:
        previous = torch.load(args.reference / "frames.pt", map_location="cpu", weights_only=True)
        for before, after in zip(previous, reference, strict=True):
            torch.testing.assert_close(before, after, rtol=0, atol=0)
        result["frames_exact"] = True
    if args.mode == "baseline":
        result.update(end=process_snapshot(), iterations=len(reference),
                      scope="bounded baseline; automatic GC remains enabled, no forced GC")
        return result
    del reference
    phases = {}
    for phase in ("same_video", "switch_video", "exceptions", "decode_exceptions"):
        snapshots, started = [], time.monotonic()
        retained_errors = []
        for index in range(args.iterations):
            case = cases[index % len(cases)] if phase == "switch_video" else cases[0]
            if phase == "decode_exceptions":
                real_open = av.open
                class FailingContainer:
                    def __init__(self, actual):
                        self.actual, self.streams = actual, actual.streams
                    def seek(self, *a, **kw):
                        return self.actual.seek(*a, **kw)
                    def close(self):
                        self.actual.close()
                    def decode(self, **kw):
                        iterator = self.actual.decode(**kw)
                        try:
                            yield next(iterator)
                            raise MemoryError("injected failure after a real AV1 frame")
                        finally:
                            iterator.close()
                with patch.object(av, "open", lambda *a, **kw: FailingContainer(real_open(*a, **kw))):
                    try:
                        decode(case["path"], [.5], .100001, "pyav")
                    except MemoryError as error:
                        retained_errors.append(error)
                        del retained_errors[:-4]
                    else:
                        raise AssertionError("expected injected decode error")
            elif phase == "exceptions":
                try:
                    # A real decoder is opened and advanced, then tolerance validation fails.
                    decode(case["path"], [.031], .00001, "pyav")
                except AssertionError as error:
                    retained_errors.append(error)
                    del retained_errors[:-4]
                else:
                    raise AssertionError("expected timestamp validation error")
            else:
                decode(case["path"], case["timestamps"], case["tolerance_s"], "pyav")
            state = process_snapshot()
            snapshots.append(state)
            write_event(output, dict(phase=phase, iteration=index, **state))
            if state["threads"] > start["threads"] + 64 or state["rss_mib"] > start["rss_mib"] + 2048:
                raise RuntimeError("decoder resource safety ceiling exceeded")
        warmed = snapshots[len(snapshots) // 4:]
        phases[phase] = dict(seconds=time.monotonic() - started, iterations=args.iterations,
                             retained_exceptions=len(retained_errors), first=warmed[0], last=warmed[-1],
                             ranges={key: [min(s[key] for s in warmed), max(s[key] for s in warmed)]
                                     for key in ("rss_mib", "threads", "fds")})
    result.update(phases=phases, end=process_snapshot())
    return result


def tensor_digest(batch):
    digest = hashlib.sha256()
    for key, value in sorted(batch.items()):
        if isinstance(value, torch.Tensor):
            digest.update(key.encode())
            digest.update(str((value.dtype, tuple(value.shape))).encode())
            digest.update(value.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def production_audit(args, output):
    from utils.load_training_dataset import build_concat_streaming_dataset, create_dataloader_for_concat, set_dataloader_epoch

    dataset = build_concat_streaming_dataset(
        ["stage05_droid_partial_mixed", "stage05_rh20t_mixed"],
        "/opt/data/private/lq/models/Qwen3-VL-2B-Instruct", None, 1, 32, None,
        loss_type="vlm_and_action", max_length=1024, aux_dataset_config="configs/aux_four_dataset_v1.json")
    # Check all sample tensors and identities, not just decoded pixel arrays.
    if args.mode == "production-baseline":
        saved = []
        with patch("utils.stage05_dataset.decode_video_frames", baseline_decoder()):
            for ds in dataset.datasets:
                for index in (0, len(ds) // 2):
                    saved.append(ds[index])
        torch.save(saved, output / "samples.pt")
        return dict(samples=len(saved), end=process_snapshot(), scope="isolated baseline process exits before any worker fork")
    if args.reference is None:
        raise ValueError("production audit requires an isolated production-baseline reference")
    previous = iter(torch.load(args.reference / "samples.pt", map_location="cpu", weights_only=True))
    comparisons = []
    for ds in dataset.datasets:
        for index in (0, len(ds) // 2):
            before = next(previous)
            after = ds[index]
            for key, value in before.items():
                if isinstance(value, torch.Tensor):
                    torch.testing.assert_close(value, after[key], rtol=0, atol=0)
                else:
                    assert value == after[key], key
            comparisons.append(dict(dataset=ds.spec.dataset_entry, index=index, digest=tensor_digest(after)))
    loader = create_dataloader_for_concat(dataset, batch_size_per_device=2, num_processes=1,
                                         num_workers=args.workers, prefetch_factor=3, seed=42)
    parent = psutil.Process()
    start = time.monotonic()
    records = []
    for epoch in range(2):
        set_dataloader_epoch(loader, loader.batch_sampler, epoch)
        iterator = iter(loader)
        worker_pids = []
        try:
            for index in range(args.iterations):
                batch = next(iterator)
                workers = []
                for child in parent.children():
                    if child.name().startswith("pt_data_worker"):
                        workers.append(process_snapshot(child))
                        worker_pids.append(child.pid)
                event = dict(epoch=epoch, batch=index, digest=tensor_digest(batch), parent=process_snapshot(),
                             workers=workers, dataset_id=batch["dataset_id"].tolist(),
                             episode_id=batch["episode_id"].tolist(), frame_id=batch["frame_id"].tolist())
                write_event(output, event)
                records.append(event)
        finally:
            # Dropping the owned iterator invokes PyTorch's documented iterator
            # lifetime shutdown. There is no persistent worker or shared decoder.
            del iterator
        gone, alive = psutil.wait_procs([psutil.Process(pid) for pid in set(worker_pids) if psutil.pid_exists(pid)], timeout=10)
        if alive:
            raise RuntimeError(f"DataLoader workers did not exit: {[p.pid for p in alive]}")
        write_event(output, dict(phase="workers_exited", epoch=epoch, pids=sorted(set(worker_pids)), **process_snapshot()))
    summaries = {}
    for record in records:
        if record["batch"] < args.iterations // 4:
            continue
        for worker in record["workers"]:
            summaries.setdefault(str(worker["pid"]), []).append(worker)
    summaries = {pid: dict(first=states[0], last=states[-1],
                          ranges={key: [min(s[key] for s in states), max(s[key] for s in states)]
                                  for key in ("rss_mib", "threads", "fds")}) for pid, states in summaries.items()}
    return dict(sample_comparisons=comparisons, workers=args.workers, prefetch_factor=3, persistent_workers=False,
                batch_size=2, epochs=2, batches_per_epoch=args.iterations, seconds=time.monotonic() - start,
                worker_resources=summaries, exact_sample_fields=True,
                cache_bounds={"parquet_files_per_dataset": 1, "episode_rows_per_dataset": 2, "active_decoder_cache": 0})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("baseline", "fixed", "production", "production-baseline"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--iterations", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("CPU diagnostics require CUDA_VISIBLE_DEVICES empty")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "experiment.md").write_text(
        (ROOT / "experiments/resource_preflight_20260907/experiment.md").read_text()
        + "\nCommand: `" + shlex.join([sys.executable, *sys.argv]) + "`\n")
    report = dict(command=shlex.join([sys.executable, *sys.argv]), mode=args.mode,
                  started=time.time(), before_system=system_snapshot(), pyav=av.__version__,
                  libraries=av.library_versions, video_module=video_utils.__file__, baseline_commit=BASELINE,
                  source_sha256=hashlib.sha256((ROOT / VIDEO_SOURCE).read_bytes()).hexdigest())
    try:
        report["result"] = production_audit(args, args.output) if args.mode.startswith("production") else raw_audit(args, args.output)
        report["status"] = "passed"
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report.update(ended=time.time(), after_system=system_snapshot())
        (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
        with (args.output / "experiment.md").open("a") as stream:
            stream.write("\nResult: `" + json.dumps({key: report[key] for key in ("status", "started", "ended")}) + "`\n")
    print(json.dumps({"status": report["status"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
