import argparse
import json

import torch
import torch.distributed as dist
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.state import AcceleratorState
from torch import nn
from transformers import BatchFeature

from train_vla import run_optimizer_step_window, json_scalar_metrics


class TinyObjectiveModel(nn.Module):
    def __init__(self, loss_type: str):
        super().__init__()
        self.ar_weight = nn.Parameter(torch.tensor(0.5))
        self.fm_weight = nn.Parameter(torch.tensor(0.5))
        self.loss_type = loss_type

    def forward(
        self,
        batch,
        training_progress,
        vlm_loss_weight=1.0,
        action_expert_loss_weight=1.0,
    ):
        del training_progress
        data = {
            "vlm_loss_weight": float(vlm_loss_weight),
            "action_expert_loss_weight": float(action_expert_loss_weight),
        }
        total = (self.ar_weight + self.fm_weight) * 0.0
        if self.loss_type in {"vlm", "vlm_and_action"}:
            mask = batch["labels"][:, 1:].ne(-100)
            elements = (self.ar_weight * batch["ar_x"] - batch["ar_y"]).square()
            ar_sum = (elements * mask).sum()
            ar_count = mask.sum()
            ar_mean = ar_sum / ar_count.clamp_min(1)
            data.update(
                ar_loss=ar_mean,
                vlm_loss=ar_mean,
                ar_loss_sum=ar_sum,
                ar_loss_count=ar_count,
            )
            total = total + vlm_loss_weight * ar_mean
        if self.loss_type in {"action", "vlm_and_action"}:
            mask = batch["action_mask"].bool()
            elements = (self.fm_weight * batch["fm_x"] - batch["fm_y"]).square()
            fm_sum = (elements * mask).sum()
            fm_count = mask.sum()
            fm_mean = fm_sum / fm_count.clamp_min(1)
            data.update(
                flow_matching_loss=fm_mean,
                action_expert_loss=fm_mean,
                flow_matching_loss_sum=fm_sum,
                flow_matching_loss_count=fm_count,
            )
            total = total + action_expert_loss_weight * fm_mean
        data["loss"] = total
        data["total_loss"] = total
        return BatchFeature(data)


def _batch(ar_values, fm_values, *, fm_valid=True):
    ar_x = torch.tensor([ar_values], dtype=torch.float32)
    ar_y = torch.zeros_like(ar_x)
    labels = torch.full((1, ar_x.shape[1] + 1), -100, dtype=torch.long)
    labels[:, 1:] = torch.arange(1, ar_x.shape[1] + 1)
    fm_x = torch.tensor([[fm_values]], dtype=torch.float32)
    fm_y = torch.zeros_like(fm_x)
    action_mask = torch.full_like(fm_x, bool(fm_valid), dtype=torch.bool)
    return {
        "input_ids": torch.zeros_like(labels),
        "labels": labels,
        "ar_x": ar_x,
        "ar_y": ar_y,
        "action": torch.zeros_like(fm_x),
        "action_mask": action_mask,
        "fm_x": fm_x,
        "fm_y": fm_y,
        "context_token_count": torch.tensor([10 + len(ar_values)]),
        "context_token_count_valid": torch.tensor([True]),
        "padding_token_count": torch.tensor([2]),
        "padding_token_count_valid": torch.tensor([True]),
    }


def _concat(batches):
    result = {}
    for key in batches[0]:
        values = [batch[key] for batch in batches]
        target_shape = [max(value.shape[axis] for value in values) for axis in range(1, values[0].ndim)]
        padded = []
        for value in values:
            padding = []
            for axis in range(value.ndim - 1, 0, -1):
                padding.extend((0, target_shape[axis - 1] - value.shape[axis]))
            fill = -100 if key == "labels" else 0
            padded.append(F.pad(value, padding, value=fill))
        result[key] = torch.cat(padded, dim=0)
    return result


def _scenario(name, rank, distributed, distributed_partition="original"):
    if distributed:
        if distributed_partition == "redistributed":
            if rank == 0:
                return [
                    _batch([1.0, 3.0], [1.0]),
                    _batch([5.0], [4.0], fm_valid=False),
                ]
            return [
                _batch([2.0], [2.0]),
                _batch([4.0], [3.0]),
            ]
        if rank == 0:
            return [
                _batch([1.0], [1.0, 2.0], fm_valid=False),
                _batch([2.0], [3.0], fm_valid=False),
            ]
        return [
            _batch([3.0, 4.0], [1.0]),
            _batch([5.0], [2.0, 3.0]),
        ]

    first = _batch(
        [1.0], [1.0], fm_valid=name not in {"joint_vqa", "joint_all_vqa"}
    )
    second = _batch(
        [2.0, 3.0, 4.0],
        [2.0, 3.0, 4.0],
        fm_valid=name != "joint_all_vqa",
    )
    return [first, second]


def _execute(
    *, loss_type, gas, combined, distributed, scenario, distributed_partition="original"
):
    accelerator = Accelerator(cpu=True, gradient_accumulation_steps=gas)
    rank = accelerator.process_index
    batches = _scenario(scenario, rank, distributed, distributed_partition)
    if combined:
        batches = [_concat(batches)]

    model = TinyObjectiveModel(loss_type)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)

    metrics = run_optimizer_step_window(
        model=model,
        batches=batches,
        accelerator=accelerator,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        training_progress=0.0,
        loss_type=loss_type,
        vlm_loss_weight=2.0,
        action_expert_loss_weight=5.0,
        next_global_step=1,
    )
    accelerator.wait_for_everyone()
    unwrapped = accelerator.unwrap_model(model)
    payload = None
    if accelerator.is_main_process:
        payload = {
            "ar_parameter": float(unwrapped.ar_weight.detach()),
            "fm_parameter": float(unwrapped.fm_weight.detach()),
            "metrics": json_scalar_metrics(metrics),
        }
    return payload, accelerator


def _reset_accelerate():
    AcceleratorState._reset_state(reset_partial_state=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--loss-type", choices=("vlm", "action", "vlm_and_action"))
    parser.add_argument("--gas", type=int)
    parser.add_argument("--combined", action="store_true")
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument(
        "--distributed-partition", choices=("original", "redistributed"), default="original"
    )
    parser.add_argument("--scenario", default="standard")
    parser.add_argument("--comparison-suite", action="store_true")
    args = parser.parse_args()

    if args.comparison_suite:
        comparisons = {}
        for loss_type, scenario in (
            ("vlm", "standard"),
            ("action", "standard"),
            ("vlm_and_action", "standard"),
            ("vlm_and_action", "joint_vqa"),
            ("vlm_and_action", "joint_all_vqa"),
        ):
            reference, _ = _execute(
                loss_type=loss_type,
                gas=1,
                combined=True,
                distributed=False,
                scenario=scenario,
            )
            _reset_accelerate()
            candidate, _ = _execute(
                loss_type=loss_type,
                gas=2,
                combined=False,
                distributed=False,
                scenario=scenario,
            )
            _reset_accelerate()
            comparisons[f"{loss_type}:{scenario}"] = {
                "gas1": reference,
                "gas2": candidate,
            }
        reference, _ = _execute(
            loss_type="vlm_and_action",
            gas=1,
            combined=True,
            distributed=False,
            scenario="standard",
        )
        _reset_accelerate()
        candidate, _ = _execute(
            loss_type="vlm_and_action",
            gas=3,
            combined=False,
            distributed=False,
            scenario="standard",
        )
        comparisons["partial"] = {"gas1": reference, "gas3": candidate}
        print("RESULT " + json.dumps({"comparisons": comparisons}, sort_keys=True))
        return

    if args.loss_type is None or args.gas is None:
        parser.error("--loss-type and --gas are required without --comparison-suite")
    payload, accelerator = _execute(
        loss_type=args.loss_type,
        gas=args.gas,
        combined=args.combined,
        distributed=args.distributed,
        scenario=args.scenario,
        distributed_partition=args.distributed_partition,
    )
    if accelerator.is_main_process:
        print("RESULT " + json.dumps(payload, sort_keys=True))
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


if __name__ == "__main__":
    main()
