import json
import os
import tempfile
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn

from model.difference_query import (
    DifferenceQuery,
    resolve_difference_query_config,
    save_difference_query_artifacts,
)
from utils.training_checkpoint import (
    checkpoint_model_optimizer_scheduler,
    resume_model_optimizer_scheduler,
)


class TinyZeroQueryModel(nn.Module):
    def __init__(self, checkpoint_directory: str | None = None):
        super().__init__()
        checkpoint = None
        if checkpoint_directory is not None:
            checkpoint = resolve_difference_query_config(
                checkpoint_directory,
                None,
            ).checkpoint_tensor
        self.difference_query = DifferenceQuery(8, 16, initializer_std=0.02)
        if checkpoint is not None:
            with torch.no_grad():
                self.difference_query.weight.copy_(checkpoint)
        self.projection = nn.Linear(16, 1)

    def forward(self, inputs):
        query = self.difference_query.for_batch(inputs.shape[0], inputs)
        return self.projection(query + inputs.unsqueeze(1)).square().mean()

    def save_pretrained(self, directory):
        save_difference_query_artifacts(
            directory,
            enabled=True,
            hidden_size=16,
            difference_query=self.difference_query.weight,
        )


class WorkerAccelerator:
    def __init__(self, rank: int):
        self.is_main_process = rank == 0

    def print(self, *args, **kwargs):
        if self.is_main_process:
            print(*args, **kwargs)

    def save(self, state, path):
        if self.is_main_process:
            torch.save(state, path)

    @staticmethod
    def unwrap_model(model):
        return model.module

    @staticmethod
    def wait_for_everyone():
        dist.barrier()


def optimizer_step(engine) -> int:
    optimizer = getattr(engine.optimizer, "optimizer", engine.optimizer)
    return max(
        int(state["step"].item())
        for state in optimizer.state.values()
        if "step" in state
    )


def assert_rank_parameters_match(parameter: torch.Tensor) -> None:
    gathered = [torch.empty_like(parameter) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, parameter)
    for other_rank in gathered[1:]:
        torch.testing.assert_close(gathered[0], other_rank)


def main() -> None:
    required_environment = (
        "LOCAL_RANK",
        "RANK",
        "WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
    )
    missing = [name for name in required_environment if name not in os.environ]
    if missing:
        raise RuntimeError(f"torchrun did not provide required environment: {missing}")

    import deepspeed

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    deepspeed.init_distributed(dist_backend="nccl")
    accelerator = WorkerAccelerator(rank)
    config = {
        "train_batch_size": world_size,
        "train_micro_batch_size_per_gpu": 1,
        "gradient_accumulation_steps": 1,
        "bf16": {"enabled": True},
        "zero_optimization": {"stage": 2},
        "zero_allow_untested_optimizer": True,
    }

    directory_context = tempfile.TemporaryDirectory() if rank == 0 else nullcontext(None)
    try:
        with directory_context as local_directory:
            directory_holder = [local_directory]
            dist.broadcast_object_list(directory_holder, src=0)
            checkpoint_root = directory_holder[0]

            torch.manual_seed(41)
            model = TinyZeroQueryModel().cuda().to(torch.bfloat16)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            first_scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=1, gamma=0.5
            )
            first_engine, _, _, _ = deepspeed.initialize(
                model=model,
                optimizer=optimizer,
                config=config,
                dist_init_required=False,
            )
            inputs = torch.full(
                (1, 16), 0.25, device="cuda", dtype=torch.bfloat16
            )
            first_loss = first_engine(inputs)
            first_engine.backward(first_loss)
            first_engine.step()
            first_scheduler.step()
            expected_query = (
                first_engine.module.difference_query.weight.detach().clone()
            )
            expected_optimizer_step = optimizer_step(first_engine)
            checkpoint_model_optimizer_scheduler(
                first_engine,
                checkpoint_root,
                global_completed_steps=1,
                lr_scheduler=first_scheduler,
                accelerator=accelerator,
            )
            accelerator.wait_for_everyone()

            sidecar_directory = Path(checkpoint_root) / "latest-model-optimizer-lr"
            if rank == 0:
                config_data = json.loads(
                    (sidecar_directory / "difference_query_config.json").read_text(
                        encoding="utf-8"
                    )
                )
                assert config_data["enabled"] is True
                assert (sidecar_directory / "difference_query.safetensors").is_file()
            accelerator.wait_for_everyone()

            torch.manual_seed(99)
            resumed_model = TinyZeroQueryModel(str(sidecar_directory)).cuda().to(
                torch.bfloat16
            )
            resumed_optimizer = torch.optim.AdamW(
                resumed_model.parameters(), lr=1e-3
            )
            resumed_scheduler = torch.optim.lr_scheduler.StepLR(
                resumed_optimizer, step_size=1, gamma=0.5
            )
            resumed_engine, _, _, _ = deepspeed.initialize(
                model=resumed_model,
                optimizer=resumed_optimizer,
                config=config,
                dist_init_required=False,
            )
            completed_steps = resume_model_optimizer_scheduler(
                resumed_engine,
                checkpoint_root,
                resumed_scheduler,
            )
            assert completed_steps == 1
            assert resumed_scheduler.state_dict() == first_scheduler.state_dict()
            assert optimizer_step(resumed_engine) == expected_optimizer_step
            torch.testing.assert_close(
                resumed_engine.module.difference_query.weight,
                expected_query,
            )
            assert_rank_parameters_match(
                resumed_engine.module.difference_query.weight.detach()
            )

            resumed_loss = resumed_engine(inputs)
            resumed_engine.backward(resumed_loss)
            resumed_engine.step()
            resumed_scheduler.step()
            assert torch.isfinite(resumed_loss)
            assert all(
                torch.isfinite(parameter).all()
                for parameter in resumed_engine.module.parameters()
            )
            assert_rank_parameters_match(
                resumed_engine.module.difference_query.weight.detach()
            )
            accelerator.wait_for_everyone()
            if rank == 0:
                print(f"ZERO2_SMOKE_OK world_size={world_size}")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
