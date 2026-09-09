"""Read-only next-production-window diagnostics for the bounded validation run."""

import hashlib
from pathlib import Path

import torch
import numpy as np

from utils.training_checkpoint import capture_rng_state, restore_rng_state


def fingerprint(value):
    if isinstance(value, np.ndarray):
        return {"shape": list(value.shape), "dtype": str(value.dtype),
                "sha256": hashlib.sha256(value.tobytes()).hexdigest()}
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        return {"shape": list(tensor.shape), "dtype": str(tensor.dtype),
                "sha256": hashlib.sha256(tensor.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()}
    if isinstance(value, dict):
        return {str(key): fingerprint(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [fingerprint(item) for item in value]
    return value


def state_identity(model, scheduler, bare):
    tracker = bare.validation_seen_tracker
    return {"model": fingerprint(bare.state_dict()),
            "optimizer": fingerprint(model.optimizer.optimizer.state_dict()),
            "master_parameters": fingerprint(model.optimizer.single_partition_of_fp32_groups),
            "scheduler": fingerprint(scheduler.state_dict()),
            "cursor": dict(bare.training_data_cursor), "rng": fingerprint(capture_rng_state()),
            "sampler_contract": dict(bare.training_sampler_contract),
            "exposure": {key: fingerprint(getattr(tracker, key)) for key in
                ("entries", "seen", "unique", "duplicates", "ar_eligible", "fm_eligible", "aux_counts", "bitsets")}}


def verify_saved_training_state(model, scheduler, accelerator, directory):
    """Once per resume, compare loaded state before consuming the first new window."""
    from utils.training_checkpoint import register_deepspeed_checkpoint_safe_globals
    register_deepspeed_checkpoint_safe_globals()
    path = Path(directory)
    rank = accelerator.process_index
    bare = accelerator.unwrap_model(model)
    zero = model.optimizer
    if not getattr(zero, "partition_gradients", False):
        raise ValueError("saved-state verification requires ZeRO-2")
    rng_before = fingerprint(capture_rng_state())
    runtime = torch.load(path / f"training_runtime_rank{rank}.pt", map_location="cpu", weights_only=True)
    expected = torch.load(path / f"bf16_zero_pp_rank_{rank}_mp_rank_00_optim_states.pt",
                          map_location="cpu", weights_only=False, mmap=True)["optimizer_state_dict"]
    if fingerprint(zero.optimizer.state_dict()) != fingerprint(expected["base_optimizer_state"]):
        raise RuntimeError("restored native optimizer state differs from checkpoint")
    masters = zero.single_partition_of_fp32_groups
    saved_masters, padding = expected["single_partition_of_fp32_groups"], expected["group_paddings"]
    if len(masters) != len(saved_masters) or len(masters) != len(padding):
        raise RuntimeError("restored FP32 partition count differs from checkpoint")
    for actual, saved, pad in zip(masters, saved_masters, padding):
        if (actual.numel() != saved.numel() + pad or actual.dtype != saved.dtype or
                not torch.equal(actual.detach().reshape(-1)[:saved.numel()].cpu(), saved.reshape(-1)) or
                (pad and bool(actual.detach().reshape(-1)[saved.numel():].ne(0).any()))):
            raise RuntimeError("restored FP32 master partition differs from checkpoint")
    saved_scheduler = torch.load(path / "scheduler.pt", map_location="cpu", weights_only=True)
    if fingerprint(scheduler.state_dict()) != fingerprint(saved_scheduler):
        raise RuntimeError("restored scheduler differs from checkpoint")
    if (runtime["world_size"] != accelerator.num_processes or
            runtime["global_step"] != scheduler.state_dict()["last_epoch"] or
            runtime["cursor"] != bare.training_data_cursor or
            runtime["sampler_contract"] != bare.training_sampler_contract):
        raise RuntimeError("restored sampler/cursor differs from checkpoint")
    if rng_before != fingerprint(runtime["rng"]):
        raise RuntimeError("restored RNG differs from checkpoint")
    if accelerator.is_main_process:
        tracker = bare.validation_seen_tracker
        with np.load(path / "data_seen_state.npz", allow_pickle=False) as arrays:
            for name in ("seen", "unique", "duplicates", "ar_eligible", "fm_eligible", "aux_counts"):
                if fingerprint(getattr(tracker, name)) != fingerprint(arrays[name]):
                    raise RuntimeError(f"restored exposure differs from checkpoint: {name}")
            for i, bits in enumerate(tracker.bitsets):
                if not np.array_equal(bits, np.unpackbits(arrays[f"bitset_{i}"])[:len(bits)].astype(bool)):
                    raise RuntimeError("restored exposure bitset differs from checkpoint")
    if fingerprint(capture_rng_state()) != rng_before:
        raise RuntimeError("saved-state verification changed RNG")
    return dict(source=str(path.resolve()), rank=rank, global_step=runtime["global_step"],
                optimizer_exact=True, master_parameters_exact=True, scheduler_exact=True,
                rng_exact=True, sampler_cursor_exact=True,
                exposure_exact=True if accelerator.is_main_process else None, optimizer_updates=0)


def diagnostic_forward(model, batches, options):
    rng = capture_rng_state()
    was_training = model.training
    try:
        model.eval()
        with torch.no_grad():
            return [{key: value.detach().cpu() for key, value in model(batch, 0.,
                vlm_loss_weight=options.vlm_loss_weight,
                action_expert_loss_weight=options.action_expert_loss_weight).items()
                if isinstance(value, torch.Tensor)} for batch in batches]
    finally:
        model.train(was_training)
        restore_rng_state(rng)


def save_next_window_evidence(model, scheduler, accelerator, indexed_window, options, directory):
    bare = accelerator.unwrap_model(model)
    path = Path(directory)
    runtime = torch.load(path / f"training_runtime_rank{accelerator.process_index}.pt", map_location="cpu", weights_only=True)
    rng = capture_rng_state()
    try:
        restore_rng_state(runtime["rng"])
        before = state_identity(model, scheduler, bare)
        batches = [batch for _, batch in indexed_window]
        outputs = diagnostic_forward(model, batches, options)
        if before != state_identity(model, scheduler, bare):
            raise RuntimeError("resume diagnostic mutated training state")
        evidence = {"state": before, "next_window": fingerprint(indexed_window), "outputs": outputs,
                    "rtol": 1e-3, "atol": 1e-4, "optimizer_updates": 0}
        with (path / f"validation_resume_rank{accelerator.process_index}.pt").open("xb") as stream:
            torch.save(evidence, stream)
    finally:
        restore_rng_state(rng)


def verify_next_window_evidence(model, scheduler, accelerator, indexed_window, options):
    bare = accelerator.unwrap_model(model)
    path = Path(options.vlm_name_or_path) / f"validation_resume_rank{accelerator.process_index}.pt"
    evidence = torch.load(path, map_location="cpu", weights_only=True)
    before = state_identity(model, scheduler, bare)
    if before != evidence["state"]:
        mismatches = [key for key in before if before[key] != evidence["state"][key]]
        raise RuntimeError(f"exact resume state comparison failed: {mismatches}")
    if fingerprint(indexed_window) != evidence["next_window"]:
        raise RuntimeError("next production accumulation window identity/tensor hashes differ")
    actual = diagnostic_forward(model, [batch for _, batch in indexed_window], options)
    if evidence["rtol"] != 1e-3 or evidence["atol"] != 1e-4 or len(actual) != len(evidence["outputs"]):
        raise RuntimeError("invalid resume diagnostic contract")
    for expected, observed in zip(evidence["outputs"], actual):
        if expected.keys() != observed.keys():
            raise RuntimeError("resume diagnostic output keys differ")
        for key in expected:
            torch.testing.assert_close(observed[key], expected[key], rtol=1e-3, atol=1e-4)
    if before != state_identity(model, scheduler, bare):
        raise RuntimeError("resume diagnostic mutated training state")
    return {"state_exact": True, "next_window_exact": True, "forward_close": True, "optimizer_updates": 0}
