import os
import math
import time
import json
import torch
import torch.distributed as dist

from utils.load_training_dataset import build_concat_streaming_dataset, create_dataloader_for_concat
from model import FlowmatchingActionHeadConfig
from model import ZR0Model
from accelerate import Accelerator
from torch.optim import AdamW
from accelerate.utils import set_seed
from transformers import get_cosine_with_min_lr_schedule_with_warmup_lr_rate, get_constant_schedule_with_warmup
from torch.utils.tensorboard import SummaryWriter
from utils.training_numerics import assert_all_finite
from utils.wandb_training_logger import WandbTrainingLogger
from utils.cli_options import parse_train_options
from utils.training_checkpoint import (
    checkpoint_model_optimizer_scheduler,
    resume_model_optimizer_scheduler,
)

def parse_option(args=None):
    return parse_train_options(args)


def should_skip_resumed_batch(
    *, epoch: int, batch_idx: int, resume_epoch: int, resume_batch_idx: int
) -> bool:
    return epoch == resume_epoch and batch_idx < resume_batch_idx


def synchronize_global_step(global_completed_steps: int, accelerator) -> int:
    if accelerator.num_processes == 1:
        return global_completed_steps
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError(
            "multi-process global-step synchronization requires an initialized "
            "torch.distributed process group"
        )
    step_tensor = torch.tensor(
        global_completed_steps,
        device=accelerator.device,
        dtype=torch.long,
    )
    dist.broadcast(step_tensor, src=0)
    return int(step_tensor.item())

def get_absolute_path(path):
    if os.path.isabs(path):
        return path
    else:
        current_path = os.getcwd()
        return os.path.join(current_path, path)

def print_model_parameters(model):
    for name, p in model.named_parameters():
        print(f"{name}: {p.numel()}")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"# all params: {total_params}")
    print(f"# trainable params: {trainable_params}")

def integrity_check(batch_data, processor):
    for input_id, label, attn_mask in zip(
        batch_data["input_ids"][0], batch_data["labels"][0], batch_data["attention_mask"][0]
    ):
        print(f"{input_id} -> {repr(processor.tokenizer.decode(input_id))} -> {label} -> {attn_mask}")
    print("integrity check ends.")

def save_model(accelerator: Accelerator, model, output_ckpt_dir, tag):
    accelerator.print(f"save model at {tag}")
    # accelerator.wait_for_everyone()
    # save checkpoint
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(os.path.join(output_ckpt_dir, tag))
    # accelerator.wait_for_everyone()

def train(opt):
    set_seed(opt.seed)
    
    writer = SummaryWriter(opt.tensorboard_log_dir)
    accelerator = Accelerator()
    accelerator.print(opt)

    total_batch_size = opt.per_device_train_batch_size * accelerator.num_processes * accelerator.gradient_accumulation_steps
    accelerator.print("data items per batch:", total_batch_size)

    concat_dataset = build_concat_streaming_dataset(
        dataset_entries=opt.dataset_entries,
        model_name_or_path=opt.vlm_name_or_path,
        fast_tokenizer_path=opt.FAST_tokenizer_path,
        window_size=opt.window_size,
        action_horizon=opt.action_horizon,
        accelerator=accelerator,
        process_mode="train",
        max_pad_state_and_action_length=opt.max_pad_state_and_action_length
    )
    accelerator.print("len(concat_dataset):", len(concat_dataset))

    dataloader = create_dataloader_for_concat(
        concat_dataset,
        batch_size_per_device=opt.per_device_train_batch_size,
        num_workers=24,
        prefetch_factor=3
    )
    accelerator.wait_for_everyone()
    accelerator.print("len(dataloader):", len(dataloader))

    if opt.action_expert_name_or_path:
        accelerator.print(f"using a pre-trained action expert from {opt.action_expert_name_or_path}, thus we will load its configurations.")
        # load action expert's configurations
        with open(
            os.path.join(opt.action_expert_name_or_path, "action_expert_config.json"),
            encoding="utf-8",
        ) as config_file:
            action_expert_config_json = json.load(config_file)
        # overwrite some args
        action_expert_config_json["action_dim"] = opt.max_pad_state_and_action_length
        action_expert_config_json["state_dim"] = opt.max_pad_state_and_action_length
        action_expert_config_json["action_horizon"] = opt.action_horizon
        action_expert_config = FlowmatchingActionHeadConfig(**action_expert_config_json)
    else:
        accelerator.print("using a randomly initialized action expert.")
        # manually set some important configurations because the action expert is randomly initialized
        action_expert_config = FlowmatchingActionHeadConfig(
            action_dim = opt.max_pad_state_and_action_length, # 64 is the max padding length of action and state vectors
            state_dim = opt.max_pad_state_and_action_length,
            action_horizon = opt.action_horizon
        )

    if opt.use_lora:
        lora_args = {
            "use_lora": opt.use_lora,
            "target_modules": opt.target_modules,
            "r": opt.r,
            "lora_alpha": opt.lora_alpha,
            "lora_dropout": opt.lora_dropout
        }
    else:
        lora_args = None
    
    # initialize ZR-0 model
    model = ZR0Model(
        vlm_name_or_path = opt.vlm_name_or_path,
        action_expert_name_or_path = opt.action_expert_name_or_path,
        action_expert_config = action_expert_config,
        tune_vlm = opt.tune_vlm,
        tune_action_expert = opt.tune_action_expert,
        detach_vlm_outputs_for_action_expert = opt.detach_vlm_outputs_for_action_expert,
        lora_args = lora_args,
        use_difference_query = opt.use_difference_query,
        num_difference_queries = opt.num_difference_queries,
        vlm_attention_backend = opt.vlm_attention_backend,
    )
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters in model: {total_params}")

        total_backbone_params = sum(p.numel() for p in model.backbone.parameters())
        print(f"Total parameters in model.backbone: {total_backbone_params}")

        total_action_expert_params = sum(p.numel() for p in model.action_expert.parameters())
        print(f"Total parameters in model.action_expert: {total_action_expert_params}")

        total_dit_params = sum(p.numel() for p in model.action_expert.dit.parameters())
        print(f"Total parameters in model.action_expert.DiT: {total_dit_params}")

        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        trainable_ratio = trainable_params / total_params

        print(f"# Trainable parameters: {trainable_params}")
        print(f"Trainable ratio: {trainable_ratio:.2%}")

    if accelerator.is_main_process:
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f"Trainable parameter: {name} | shape: {param.shape}")

    accelerator.wait_for_everyone()
    # '''grouped lr (start) '''
    # param_groups = []
    # if opt.tune_vlm:
    #     param_groups.append({"params": [p for p in model.backbone.parameters()], "lr": opt.vlm_peak_learning_rate, "weight_decay": 1e-3, "name": "backbone"})
    # if opt.tune_action_expert:
    #     param_groups.append({"params": [p for p in model.action_expert.parameters()], "lr": opt.action_expert_peak_learning_rate, "weight_decay": 1e-5, "name": "action_expert"})
    # # load AdamW optimizer
    # optimizer = AdamW(
    #     # model.parameters(),
    #     param_groups,
    #     # lr = opt.vlm_peak_learning_rate, # not working
    #     betas = (0.95, 0.9995), # (0.9, 0.95),
    #     eps = 1e-6
    # )
    # for i, group in enumerate(optimizer.param_groups):
    #     accelerator.print(f"Group {i} ({group.get('name', 'unnamed')}): lr = {group['lr']}")
    # '''grouped lr (end) '''

    '''single lr (start)'''
    optimizer = AdamW(
        model.parameters(),
        lr = opt.peak_learning_rate,
        betas = (0.9, 0.95), # (0.95, 0.9995),
        eps = 1e-6, # 1e-8,
        weight_decay = 0.01, # 1e-5
    )
    '''single lr (end)'''
    
    num_total_batches = math.ceil(opt.epochs * math.ceil(len(concat_dataset) / total_batch_size))
    warmup_steps = min(20000 * accelerator.num_processes, int(num_total_batches * 0.08) * accelerator.num_processes)

    wandb_logger = WandbTrainingLogger(
        accelerator,
        project=opt.wandb_project,
        run_name=opt.wandb_run_name,
        run_id=opt.wandb_run_id,
        resume=opt.wandb_resume,
        log_dir=opt.wandb_dir,
        group=opt.wandb_group,
        tags=opt.wandb_tags,
        config={
            **vars(opt),
            "global_batch_size": total_batch_size,
            "num_processes": accelerator.num_processes,
            "num_total_steps": num_total_batches,
            "warmup_steps": warmup_steps // accelerator.num_processes,
        },
    )
    accelerator.wait_for_everyone()

    if opt.lr_scheduler == "cosine":
        # learning rate scheduler (linear warm up and cosine decay)
        lr_scheduler = get_cosine_with_min_lr_schedule_with_warmup_lr_rate(
            optimizer = optimizer,
            num_warmup_steps = warmup_steps,
            num_training_steps = num_total_batches * accelerator.num_processes,
            min_lr_rate = opt.min_lr_rate
        )
    elif opt.lr_scheduler == "constant":
        # learning rate scheduler (linear warm up and keeping constant)
        lr_scheduler = get_constant_schedule_with_warmup(
            optimizer, 
            num_warmup_steps = warmup_steps
        )
    else:
        raise ValueError("lr_scheduler should be in [cosine, constant].")

    accelerator.wait_for_everyone()
    optimizer, model, dataloader, lr_scheduler = accelerator.prepare(
        optimizer, model, dataloader, lr_scheduler
    )

    global_completed_steps = 0
    # set model to the train() mode
    model.train()
    
    if opt.resume_training:
        assert opt.vlm_name_or_path == opt.action_expert_name_or_path

        old_output_ckpt_dir = os.path.dirname(opt.vlm_name_or_path)
        global_completed_steps = resume_model_optimizer_scheduler(
            model,
            old_output_ckpt_dir,
            lr_scheduler,
        )
        
        resume_epoch = global_completed_steps * accelerator.gradient_accumulation_steps // len(dataloader)
        resume_batch_idx = global_completed_steps * accelerator.gradient_accumulation_steps % len(dataloader)

        accelerator.print("resume epoch:", resume_epoch)
        accelerator.print("resume batch index:", resume_batch_idx)
        accelerator.print("resume training from {} steps.".format(global_completed_steps))

        accelerator.print("resumed lr scheduler state dict:", lr_scheduler.state_dict())

    accelerator.wait_for_everyone()
    st = time.time()
    reached_max_train_steps = False
    for epoch in range(opt.epochs):
        # set the epoch into each dataset
        for ds in concat_dataset.datasets:
            ds.set_epoch(epoch)

        if opt.resume_training and resume_epoch > epoch:
            accelerator.print("skip {}-th epoch".format(epoch))
            accelerator.wait_for_everyone()
            continue

        accelerator.print(f"{epoch=}")

        for batch_idx, batch in enumerate(dataloader):
            if opt.resume_training and should_skip_resumed_batch(
                epoch=epoch,
                batch_idx=batch_idx,
                resume_epoch=resume_epoch,
                resume_batch_idx=resume_batch_idx,
            ):
                accelerator.print("skip {}-th batch".format(batch_idx))
                continue

            if accelerator.is_main_process and batch_idx==0 and epoch==0:
                integrity_check(batch, model.backbone.processor)
            with accelerator.accumulate(model):
                training_progress = global_completed_steps/num_total_batches
                outputs = model(batch, training_progress, opt.loss_type, opt.vlm_loss_weight, opt.action_expert_loss_weight)
                loss = outputs.loss
                next_global_step = global_completed_steps + 1
                assert_all_finite(accelerator, loss, "loss", next_global_step)

                # when deepspeed is enabled, `accelerator.backward(loss)` will perform optimizer.step(), optimizer.zero_grad(), and grad accumulation automatically. 
                # see `if self.is_gradient_accumulation_boundary():` line in path-to-env/site-packages/deepspeed/runtime/engine.py
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                global_grad_norm = model.get_global_grad_norm()
                if global_grad_norm is not None:
                    assert_all_finite(
                        accelerator,
                        global_grad_norm,
                        "global gradient norm",
                        next_global_step,
                    )

            # 'accelerator.sync_gradients' checks if the accelerator has performed an optimization step on the `total_batch_size` data samples, 
            # however, sync_gradients is not guaranteed to be consistent in all ranks
            # 1. only rank 0 updates the `global_completed_steps` variable during sync_gradients
            if accelerator.sync_gradients and accelerator.is_main_process:
                global_completed_steps += 1

            # 2. synchronize from rank 0 when distributed training is active.
            global_completed_steps = synchronize_global_step(
                global_completed_steps, accelerator
            )

            reached_max_train_steps = (
                opt.max_train_steps is not None
                and global_completed_steps >= opt.max_train_steps
            )

            # check whether to save model
            do_save = (
                global_completed_steps > 0
                and (
                    global_completed_steps % opt.save_step_interval == 0
                    or reached_max_train_steps
                )
            )

            if do_save:
                accelerator.wait_for_everyone()
                save_model(accelerator, model, opt.output_ckpt_dir, f"step-{global_completed_steps}")
                accelerator.wait_for_everyone()
                if opt.save_optimizer_and_lr_states:
                    checkpoint_model_optimizer_scheduler(model, opt.output_ckpt_dir, global_completed_steps, lr_scheduler, accelerator)
                accelerator.wait_for_everyone()
            
            if accelerator.sync_gradients and global_completed_steps % 10 == 0:
                # accelerator.print(f"GPU 0, step {global_completed_steps}, lr state dict:", lr_scheduler.state_dict())
                loss_detach = outputs.loss.detach().float()
                vlm_loss_detach = outputs.vlm_loss.detach().float() if "vlm_loss" in outputs else 0
                action_expert_loss_detach = outputs.action_expert_loss.detach().float() if "action_expert_loss" in outputs else 0

                # accelerator.print("loss:", loss_detach, "vlm loss:", vlm_loss_detach, "action expert loss:", action_expert_loss_detach)

                if accelerator.is_main_process:
                    writer.add_scalar('learning-rate', lr_scheduler.get_last_lr()[0], global_completed_steps)
                    if global_grad_norm is not None:
                        writer.add_scalar('grad-norm', global_grad_norm, global_completed_steps)
                    writer.add_scalar('training progress', training_progress, global_completed_steps)

                writer.add_scalar('train-loss/gpu-{}'.format(accelerator.process_index), loss_detach, global_completed_steps)
                if "vlm_loss" in outputs:
                    writer.add_scalar('train-vlm-loss/gpu-{}'.format(accelerator.process_index), vlm_loss_detach, global_completed_steps)
                if "action_expert_loss" in outputs:
                    writer.add_scalar('train-action-expert-loss/gpu-{}'.format(accelerator.process_index), action_expert_loss_detach, global_completed_steps)

                mean_metrics = {"train/loss": loss_detach}
                if "vlm_loss" in outputs:
                    mean_metrics["train/vlm_loss"] = vlm_loss_detach
                if "action_expert_loss" in outputs:
                    mean_metrics["train/action_expert_loss"] = action_expert_loss_detach
                scalar_metrics = {
                    "train/learning_rate": lr_scheduler.get_last_lr()[0],
                    "train/progress": global_completed_steps / num_total_batches,
                    "train/epoch": epoch,
                }
                if global_grad_norm is not None:
                    scalar_metrics["train/grad_norm"] = float(global_grad_norm)

                wandb_logger.log(
                    step=global_completed_steps,
                    mean_metrics=mean_metrics,
                    scalar_metrics=scalar_metrics,
                )

            if reached_max_train_steps:
                break

        if reached_max_train_steps:
            accelerator.wait_for_everyone()
            accelerator.print(
                f"Reached max training steps: {opt.max_train_steps}"
            )
            break
        
        accelerator.wait_for_everyone()

        if accelerator.is_main_process:
            accelerator.print(f"Epoch {epoch} finished, saving checkpoint")

        do_epoch_save = (
            (epoch + 1) % opt.save_ckpt_interval == 0
            or epoch == opt.epochs - 1
        )

        if do_epoch_save:
            accelerator.wait_for_everyone()
            save_model(accelerator, model, opt.output_ckpt_dir, f"step-{global_completed_steps}")
            accelerator.wait_for_everyone()
            if opt.save_optimizer_and_lr_states:
                checkpoint_model_optimizer_scheduler(model, opt.output_ckpt_dir, global_completed_steps, lr_scheduler, accelerator)
            accelerator.wait_for_everyone()

    writer.close()
    wandb_logger.finish(exit_code=0)

if __name__ == "__main__":
    opt = parse_option()
    train(opt)
