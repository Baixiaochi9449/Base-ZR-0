"""Exact startup comparison must detect loaded-state corruption without updates."""

import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from utils.training_checkpoint import capture_rng_state
from utils.validation_resume import fingerprint, verify_saved_training_state


@pytest.fixture
def restored(tmp_path, monkeypatch):
    monkeypatch.setattr("utils.training_checkpoint.register_deepspeed_checkpoint_safe_globals", lambda: ())
    p = torch.nn.Parameter(torch.tensor([1., 2., 0.]))
    optimizer = torch.optim.AdamW([dict(params=[p], component="vlm")], lr=.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1)
    p.grad = torch.tensor([.2, .3, 0.])
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
    tracker = SimpleNamespace(**{name: np.array([2], dtype=np.int64) for name in
                                ("seen", "unique", "duplicates", "ar_eligible", "fm_eligible")},
                              aux_counts=np.zeros((1, 16), dtype=np.int64),
                              bitsets=[np.array([True, False, True], dtype=bool)])
    bare = SimpleNamespace(training_data_cursor={"epoch": 0, "batch_idx": 2},
                           training_sampler_contract={"seed": 42}, validation_seen_tracker=tracker)
    model = SimpleNamespace(optimizer=SimpleNamespace(optimizer=optimizer, partition_gradients=True,
                            single_partition_of_fp32_groups=[p]))
    accelerator = SimpleNamespace(process_index=0, num_processes=4, is_main_process=True,
                                  unwrap_model=lambda model: bare)
    native = copy.deepcopy(optimizer.state_dict())
    torch.save(dict(optimizer_state_dict=dict(base_optimizer_state=native,
               single_partition_of_fp32_groups=[p.detach()[:2].clone()], group_paddings=[1])),
               tmp_path / "bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt")
    torch.save(scheduler.state_dict(), tmp_path / "scheduler.pt")
    torch.save(dict(rng=capture_rng_state(), global_step=1, world_size=4,
               cursor=bare.training_data_cursor, sampler_contract=bare.training_sampler_contract),
               tmp_path / "training_runtime_rank0.pt")
    np.savez_compressed(tmp_path / "data_seen_state.npz", **{name: getattr(tracker, name) for name in
                        ("seen", "unique", "duplicates", "ar_eligible", "fm_eligible", "aux_counts")},
                        bitset_0=np.packbits(tracker.bitsets[0]))
    return model, scheduler, accelerator, tmp_path, bare


def test_saved_resume_verification_is_exact_and_read_only(restored):
    model, scheduler, accelerator, path, _ = restored
    before = fingerprint(dict(optimizer=model.optimizer.optimizer.state_dict(),
                               master=model.optimizer.single_partition_of_fp32_groups,
                               scheduler=scheduler.state_dict(), rng=capture_rng_state()))
    result = verify_saved_training_state(model, scheduler, accelerator, path)
    assert result["optimizer_updates"] == 0 and result["optimizer_exact"] and result["exposure_exact"]
    assert result["master_parameters_exact"] and result["rng_exact"] and result["scheduler_exact"]
    assert before == fingerprint(dict(optimizer=model.optimizer.optimizer.state_dict(),
                                      master=model.optimizer.single_partition_of_fp32_groups,
                                      scheduler=scheduler.state_dict(), rng=capture_rng_state()))


@pytest.mark.parametrize("corruption", ["moment", "master", "padding", "scheduler", "cursor", "rng", "exposure", "bitset"])
def test_saved_resume_verification_rejects_corruption(restored, corruption):
    model, scheduler, accelerator, path, bare = restored
    master = model.optimizer.single_partition_of_fp32_groups[0]
    if corruption == "moment":
        model.optimizer.optimizer.state[master]["exp_avg"].add_(.1)
    elif corruption in {"master", "padding"}:
        with torch.no_grad():
            master[-1 if corruption == "padding" else 0].add_(.1)
    elif corruption == "scheduler":
        scheduler.last_epoch += 1
    elif corruption == "cursor":
        bare.training_data_cursor["batch_idx"] += 1
    elif corruption == "rng":
        torch.rand(1)
    elif corruption == "exposure":
        bare.validation_seen_tracker.seen[0] += 1
    else:
        bare.validation_seen_tracker.bitsets[0][0] = False
    with pytest.raises(RuntimeError, match="differs from checkpoint"):
        verify_saved_training_state(model, scheduler, accelerator, path)
