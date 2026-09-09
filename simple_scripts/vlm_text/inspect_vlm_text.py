#!/usr/bin/env python3
"""Inspect the text-only language capability of a local Qwen3-VL checkpoint."""

from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Local directory containing full Qwen3-VL safetensors and processor.")
    questions = parser.add_mutually_exclusive_group(required=True)
    questions.add_argument("--question", action="append", help="Repeat for independent questions.")
    questions.add_argument("--questions-file", type=Path, help="UTF-8 JSONL: question, optional id/reference.")
    parser.add_argument("--system-prompt", default=None)
    parser.add_argument("--device", default="cpu", help="cpu, cuda, or cuda:N (default: cpu).")
    parser.add_argument("--dtype", choices=("auto", "float32", "bfloat16", "float16"), default="auto")
    parser.add_argument("--attention-backend", choices=("sdpa", "eager"), default="sdpa")
    parser.add_argument("--max-new-tokens", type=positive_int, default=256)
    parser.add_argument("--cpu-threads", type=positive_int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, help="New directory; defaults to outputs/vlm_text/<timestamp>.")
    parser.add_argument("--training-experiment", help="Optional source training experiment path or name.")
    return parser


def read_questions(args: argparse.Namespace) -> list[dict]:
    if args.question is not None:
        records = [{"question": question} for question in args.question]
    else:
        records = []
        for line_number, line in enumerate(args.questions_file.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on line {line_number}: {exc.msg}") from exc
    if not records:
        raise ValueError("at least one question is required")
    normalized, seen = [], set()
    for index, record in enumerate(records, 1):
        if not isinstance(record, dict) or not isinstance(record.get("question"), str) or not record["question"].strip():
            raise ValueError(f"question {index} must contain a nonempty question string")
        identity = record.get("id", str(index))
        if not isinstance(identity, str) or not identity.strip() or identity in seen:
            raise ValueError(f"question {index} must have a unique nonempty string id")
        if "reference" in record and not isinstance(record["reference"], str):
            raise ValueError(f"question {index} reference must be a string")
        seen.add(identity)
        normalized.append({"id": identity, "question": record["question"],
                           **({"reference": record["reference"]} if "reference" in record else {})})
    return normalized


def inspect_checkpoint(checkpoint: Path) -> dict:
    if not checkpoint.is_dir():
        raise ValueError(f"checkpoint must be a local directory: {checkpoint}")
    if (checkpoint / "adapter_config.json").exists():
        raise ValueError("LoRA/PEFT adapters are not supported; supply an explicitly merged full VLM checkpoint")
    config_path = checkpoint / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("model_type") != "qwen3_vl":
        raise ValueError("only full Qwen3-VL checkpoints (model_type=qwen3_vl) are supported")
    index_path = checkpoint / "model.safetensors.index.json"
    if (checkpoint / "model.safetensors").is_file():
        names = ["model.safetensors"]
    else:
        index = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"invalid safetensors index: {index_path}")
        if not all(isinstance(name, str) and name for name in weight_map.values()):
            raise ValueError(f"invalid shard names: {index_path}")
        names = sorted(set(weight_map.values()))
    inventory = []
    for name in names:
        path = checkpoint / name
        if Path(name).is_absolute() or ".." in Path(name).parts or not path.is_file():
            raise ValueError(f"missing or invalid checkpoint weight shard: {name}")
        stat = path.stat()
        inventory.append({"file": name, "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    metadata = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(checkpoint.iterdir())
                if path.is_file() and path.suffix in {".json", ".jinja"}}
    return {"path": str(checkpoint), "metadata_sha256": metadata, "weight_inventory": inventory,
            "weight_identity_method": "filename/size/mtime only; weight contents are not hashed"}


def load_model(args: argparse.Namespace):
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    device = torch.device(args.device)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("--device must be cpu or cuda[:N]")
    if device.type == "cuda":
        if not torch.cuda.is_available() or (device.index is not None and device.index >= torch.cuda.device_count()):
            raise ValueError(f"requested CUDA device is unavailable: {device}")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        torch.cuda.set_device(device)
    dtype = args.dtype
    if dtype == "auto":
        dtype = "bfloat16" if device.type == "cuda" and torch.cuda.is_bf16_supported() else "float32"
    if device.type == "cpu" and dtype == "float16":
        raise ValueError("use float32 or bfloat16 on CPU")
    if device.type == "cuda" and dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise ValueError("requested CUDA device does not support bfloat16")
    processor = AutoProcessor.from_pretrained(args.checkpoint, local_files_only=True, trust_remote_code=False)
    model, loading_info = Qwen3VLForConditionalGeneration.from_pretrained(
        args.checkpoint, local_files_only=True, trust_remote_code=False,
        use_safetensors=True, dtype=getattr(torch, dtype),
        attn_implementation=args.attention_backend, output_loading_info=True,
    )
    problems = {key: loading_info[key] for key in
                ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs") if loading_info.get(key)}
    if problems:
        raise ValueError(f"VLM checkpoint did not load exactly: {problems}")
    model.requires_grad_(False)
    model.eval().to(device)
    return model, processor


def generate_answer(model, processor, question: str, args: argparse.Namespace, generation_config) -> dict:
    import torch

    messages = []
    if args.system_prompt is not None:
        messages.append({"role": "system", "content": [{"type": "text", "text": args.system_prompt}]})
    messages.append({"role": "user", "content": [{"type": "text", "text": question}]})
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt",
    ).to(next(model.parameters()).device)
    prompt_length = inputs["input_ids"].shape[1]
    context_limit = model.config.text_config.max_position_embeddings
    if prompt_length + args.max_new_tokens > context_limit:
        raise ValueError(f"prompt ({prompt_length}) + max-new-tokens ({args.max_new_tokens}) exceeds context ({context_limit})")
    started = time.perf_counter()
    with torch.inference_mode():
        # Transformers >=4.50 otherwise replaces greedy defaults with checkpoint sampling settings.
        generated = model.generate(**inputs, generation_config=generation_config,
                                   use_model_defaults=False, return_dict_in_generate=False)
    tokens = generated[0, prompt_length:].tolist()
    eos_ids = generation_config.eos_token_id
    eos_ids = [eos_ids] if isinstance(eos_ids, int) else (eos_ids or [])
    stopped_on_eos = bool(tokens and tokens[-1] in eos_ids)
    hit_limit = len(tokens) >= args.max_new_tokens
    return {
        "messages": messages, "prompt_token_ids": inputs["input_ids"][0].tolist(),
        "prompt_token_count": prompt_length, "generated_token_ids": tokens,
        "generated_token_count": len(tokens), "hit_max_new_tokens": hit_limit,
        "stopped_on_eos": stopped_on_eos, "truncated": hit_limit and not stopped_on_eos,
        "generated_text": processor.tokenizer.decode(tokens, skip_special_tokens=True,
                                                       clean_up_tokenization_spaces=False).strip(),
        "generation_seconds": time.perf_counter() - started,
    }


def write_report(output: Path, report: dict) -> None:
    temporary = output / "result.json.tmp"
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output / "result.json")
    lines = [
        "# VLM text capability inspection", "",
        f"- Status: {report['status']}", f"- Created: {report['started_at']}",
        f"- Finished: {report.get('finished_at', 'pending')}", f"- Owner: {report['owner']}",
        "- Purpose: manual inspection of independent text-only answers; no benchmark score.",
        f"- Checkpoint / initialization: {report['checkpoint']['path']}",
        f"- Source training experiment: {report['training_experiment'] or 'unspecified'}",
        f"- Code commit: {report['git_commit']}",
        "- Working tree changes at launch are recorded in result.json: git_status.",
        f"- Output: {output}",
        f"- Questions: {len(report['questions'])}; completed: {len(report['answers'])}; seed: {report['seed']}",
        "- Inputs: explicit questions; references are for review and never enter the prompt.",
        "- Dataset split / sample selection: user-selected prompts; no train/validation/test dataset.",
        "- Inference only: all VLM parameters frozen; no optimizer, loss, scheduler, or training steps.",
        "- Vision tower and merger weights load as part of Qwen3-VL but receive no images.",
        "- Action Expert, Difference Query, Slot and Flow modules are not loaded.",
        "- Images / resize / crop / padding / normalization / cameras: not applicable (text only).",
        "- State / actions / horizon / execution / control frequency / rollouts: not applicable.",
        "- Batch size: 1; one independent greedy decode per question; no conversation history.",
        "- W&B / checkpoints saved / resume / early stopping: not applicable (standalone inference).",
        f"- Runtime: {json.dumps(report.get('runtime', {}), ensure_ascii=False)}",
        "- Complete generation settings and exact input/output token IDs are in result.json.",
        "- Weight identity uses names, sizes and mtimes, not full weight hashes.",
        "", "## Command", "", "```bash", report["command"], "```", "",
    ]
    if "error" in report:
        lines.extend(["## Error", "", "```text", report["error"], "```", ""])
    (output / "experiment.md").write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> Path:
    args.checkpoint = args.checkpoint.expanduser().resolve()
    questions = read_questions(args)
    checkpoint = inspect_checkpoint(args.checkpoint)
    output = (args.output_dir or ROOT / "outputs/vlm_text" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")).expanduser().resolve()
    if output.is_relative_to(args.checkpoint):
        raise ValueError("output directory must be outside the checkpoint directory")
    output.mkdir(parents=True, exist_ok=False)

    def git_output(*arguments):
        result = subprocess.run(["git", *arguments], cwd=ROOT, text=True, capture_output=True, check=False)
        return result.stdout.strip() if result.returncode == 0 else "unavailable"

    report = {
        "format_version": 1, "status": "running", "mode": "question-chat",
        "started_at": datetime.now().astimezone().isoformat(), "owner": getpass.getuser(),
        "checkpoint": checkpoint, "training_experiment": args.training_experiment,
        "git_commit": git_output("rev-parse", "HEAD"), "git_status": git_output("status", "--short"),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "command": shlex.join([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]]),
        "working_directory": str(Path.cwd()), "seed": args.seed,
        "options": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "questions": questions, "answers": [],
    }
    write_report(output, report)
    print(f"Results: {output}", flush=True)
    try:
        import torch
        import transformers
        from transformers import GenerationConfig, set_seed

        torch.set_num_threads(args.cpu_threads)
        set_seed(args.seed)
        model, processor = load_model(args)
        parameter = next(model.parameters())
        report["runtime"] = {
            "python": sys.version.split()[0], "torch": torch.__version__, "transformers": transformers.__version__,
            "device": str(parameter.device), "dtype": str(parameter.dtype),
            "attention_backend": model.config._attn_implementation, "cpu_threads": torch.get_num_threads(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "device_name": torch.cuda.get_device_name(parameter.device) if parameter.is_cuda else "CPU",
        }
        # A fresh config avoids inheriting checkpoint-specific sampling/beam settings.
        generation_config = GenerationConfig(
            max_new_tokens=args.max_new_tokens, do_sample=False, num_beams=1, use_cache=True,
            bos_token_id=model.generation_config.bos_token_id,
            eos_token_id=(model.generation_config.eos_token_id
                          if model.generation_config.eos_token_id is not None else processor.tokenizer.eos_token_id),
            pad_token_id=processor.tokenizer.pad_token_id,
        )
        report["generation_config"] = generation_config.to_dict()
        report["use_model_defaults"] = False
        for question in questions:
            answer = generate_answer(model, processor, question["question"], args, generation_config)
            report["answers"].append({**question, **answer})
            write_report(output, report)
            print(f"\n[{question['id']}] {question['question']}\n{answer['generated_text']}", flush=True)
            if answer["truncated"]:
                print("[truncated: reached --max-new-tokens without an EOS token]", flush=True)
        report["status"] = "complete"
    except BaseException as exc:
        report["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["finished_at"] = datetime.now().astimezone().isoformat()
        write_report(output, report)
    return output


def main() -> int:
    args = build_parser().parse_args()
    try:
        run(args)
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
