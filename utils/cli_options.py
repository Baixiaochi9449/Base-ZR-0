import argparse


def _add_difference_query_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--use_difference_query",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable learnable Difference Query conditioning for the action expert.",
    )
    parser.add_argument(
        "--num_difference_queries",
        type=int,
        default=None,
        help=(
            "Number of Difference Query tokens; requires --use_difference_query "
            "for random initialization."
        ),
    )
    parser.add_argument(
        "--vlm_attention_backend",
        choices=["eager", "flash_attention_2", "sdpa"],
        default=None,
        help="Optional Qwen attention backend override; Difference Query requires sdpa.",
    )


def build_train_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vlm_name_or_path", type=str, help="file path of pretrained VLM"
    )
    parser.add_argument(
        "--action_expert_name_or_path",
        type=str,
        help=(
            "path of the pretrained action expert. Unset means that we will use "
            "a randomly initialized action expert."
        ),
    )
    parser.add_argument(
        "--FAST_tokenizer_path",
        type=str,
        help="file path of pretrained FAST action tokenizer",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="batch size per gpu device.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--save_ckpt_interval", type=int, default=1)
    parser.add_argument("--save_step_interval", type=int, default=20000)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Optional early-stop step for smoke validation; scheduler horizon remains unchanged.",
    )
    parser.add_argument(
        "--peak_learning_rate", type=float, default=1e-5, help="peak learning rate"
    )
    parser.add_argument(
        "--min_lr_rate",
        type=float,
        default=0.1,
        help="the minimal learning rate in the end of training (percent of peak LR)",
    )
    parser.add_argument(
        "--tensorboard_log_dir", type=str, default="./outputs/train_logs/ZR-0"
    )
    parser.add_argument(
        "--output_ckpt_dir", type=str, default="./outputs/ckpts/ZR-0"
    )
    parser.add_argument("--wandb_project", type=str)
    parser.add_argument("--wandb_run_name", type=str)
    parser.add_argument("--wandb_run_id", type=str)
    parser.add_argument(
        "--wandb_resume", choices=["never", "must"], default="never"
    )
    parser.add_argument("--wandb_dir", type=str, default="./outputs/wandb")
    parser.add_argument("--wandb_group", type=str)
    parser.add_argument("--wandb_tags", type=str, nargs="*")
    parser.add_argument(
        "--tune_vlm", action="store_true", help="Whether to fine-tune the VLM"
    )
    parser.add_argument(
        "--tune_action_expert",
        action="store_true",
        help="Whether to fine-tune the projectors in the action expert",
    )
    parser.add_argument(
        "--detach_vlm_outputs_for_action_expert",
        action="store_true",
        help="whether to stop gradient from the action expert to VLM",
    )
    parser.add_argument(
        "--loss_type",
        type=str,
        default="vlm_and_action",
        help="support [vlm_and_action, vlm, action]",
    )
    parser.add_argument(
        "--vlm_loss_weight",
        type=float,
        default=1.0,
        help=(
            "when setting loss type to vlm_and_action, we can control the weight "
            "of the VLM's loss"
        ),
    )
    parser.add_argument(
        "--action_expert_loss_weight",
        type=float,
        default=1.0,
        help=(
            "when setting loss type to vlm_and_action, we can control the weight "
            "of the action expert's loss"
        ),
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="cosine",
        help="the type of the LR Scheduler. Avaliable: [cosine, constant]",
    )
    parser.add_argument(
        "--resume_training", action="store_true", help="whether to resume training"
    )
    parser.add_argument(
        "--save_optimizer_and_lr_states",
        action="store_true",
        help="Whether to save states of the optimizer and the LR scheduler",
    )
    _add_difference_query_options(parser)
    parser.add_argument(
        "--use_lora",
        action="store_true",
        help="Whether to use LoRA to fine-tune the model",
    )
    parser.add_argument(
        "--target_modules",
        type=str,
        default="gate_proj, up_proj, down_proj",
        help="The names of the modules to apply the adapter to",
    )
    parser.add_argument(
        "--r", type=int, default=16, help="LoRA attention dimension (the `rank`)"
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=32,
        help="The alpha parameter for LoRA scaling. Typically setting to the double of `r`.",
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.0,
        help="The dropout probability for LoRA layers",
    )
    parser.add_argument(
        "--dataset_entries",
        type=str,
        nargs="+",
        help=(
            "List of training dataset entries, e.g., bridge_orig_lerobot "
            "fractal20220817_data_lerobot libero_v21"
        ),
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=1,
        help="size of the sliding window for historical image observations",
    )
    parser.add_argument(
        "--action_horizon", type=int, default=32, help="size of the action chunk"
    )
    parser.add_argument(
        "--max_pad_state_and_action_length",
        type=int,
        default=64,
        help="dim size of the max padded state and action",
    )
    return parser


def parse_train_options(args=None) -> argparse.Namespace:
    parser = build_train_parser()
    options = parser.parse_args(args)
    if options.max_train_steps is not None and options.max_train_steps < 1:
        parser.error("--max_train_steps must be at least 1")
    return options


def build_server_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_entry",
        type=str,
        default="libero_v21",
        help="the pre-registration dataset entry in dataset2feature.yaml",
    )
    parser.add_argument("--ckpt_dir", type=str, help="checkpoint directory")
    parser.add_argument(
        "--inference_mode",
        type=str,
        help="specify the inference mode, in `direct_action` or `subtask_then_action`",
    )
    parser.add_argument("--window_size", type=int, default=1, help="window size")
    parser.add_argument(
        "--num_denoised_steps",
        type=int,
        default=5,
        help="number of denoised steps",
    )
    parser.add_argument(
        "--max_pad_state_and_action_length",
        type=int,
        default=64,
        help="max padding length",
    )
    parser.add_argument("--port", type=int, default=8000, help="server port")
    _add_difference_query_options(parser)
    return parser


def parse_server_options(args=None) -> argparse.Namespace:
    return build_server_parser().parse_args(args)
