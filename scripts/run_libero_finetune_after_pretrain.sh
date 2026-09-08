#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CONDA_SH=/opt/data/private/lq/miniconda3/etc/profile.d/conda.sh
PYTHON_BIN=/opt/data/private/lq/miniconda3/envs/ZR-0/bin/python
PRETRAIN_OUTPUT="$ROOT_DIR/outputs/pretrain/tabletop_v3_dq32_joint_gbs128_seed42_mbs16_gas2"
FINAL_CHECKPOINT="$PRETRAIN_OUTPUT/step-19424"
FINAL_METRICS="$PRETRAIN_OUTPUT/training_metrics.jsonl"
FORMAL_OUTPUT="$ROOT_DIR/outputs/ckpts/Qwen3-VL-2B-Instruct-LIBERO-wo-ECoT-PT-FinalRMSNorm-difference-query-nq32-tabletop-v3-joint-init"
FULL_LOG_DIR="$ROOT_DIR/outputs/full_logs/libero-dq32-tabletop-v3-joint-init"
POLL_SECONDS=${ZR0_WAIT_POLL_SECONDS:-60}
SOURCE_ACTION_HORIZON=${ZR0_SOURCE_ACTION_HORIZON:-32}
FINETUNE_ACTION_HORIZON=${ZR0_LIBERO_ACTION_HORIZON:-10}

usage() {
    echo "Usage: $0 {run|dry-run}" >&2
}

mode=${1:-run}
case "$mode" in
    run|dry-run)
        ;;
    *)
        usage
        exit 2
        ;;
esac

[[ "$SOURCE_ACTION_HORIZON" =~ ^[1-9][0-9]*$ ]] || {
    echo "ZR0_SOURCE_ACTION_HORIZON must be a positive integer" >&2; exit 2;
}
[[ "$FINETUNE_ACTION_HORIZON" =~ ^[1-9][0-9]*$ ]] || {
    echo "ZR0_LIBERO_ACTION_HORIZON must be a positive integer" >&2; exit 2;
}

if [[ "$mode" == "dry-run" ]]; then
    printf 'final_checkpoint=%s\n' "$FINAL_CHECKPOINT"
    printf 'smoke=train:2,resume:3\n'
    printf 'formal_output=%s\n' "$FORMAL_OUTPUT"
    printf 'launcher=%s\n' "$ROOT_DIR/scripts/run_libero_wo_ecot_pt.sh"
    exit 0
fi

if [[ ! "$POLL_SECONDS" =~ ^[1-9][0-9]*$ ]]; then
    echo "ZR0_WAIT_POLL_SECONDS must be a positive integer" >&2
    exit 2
fi
if [[ ! -f "$CONDA_SH" || ! -x "$PYTHON_BIN" ]]; then
    echo "ZR-0 Conda environment is unavailable" >&2
    exit 1
fi
if [[ -z "${WANDB_API_KEY:-}" ]]; then
    echo "WANDB_API_KEY is not set; refusing to queue formal training" >&2
    exit 1
fi

# shellcheck source=/dev/null
source "$CONDA_SH"
conda activate ZR-0
export PYTHONNOUSERSITE=1
mkdir -p "$FULL_LOG_DIR"

final_checkpoint_ready() {
    [[ -f "$FINAL_CHECKPOINT/zr0_checkpoint_metadata.json" ]] || return 1
    [[ -f "$FINAL_CHECKPOINT/difference_query.safetensors" ]] || return 1
    [[ -f "$FINAL_CHECKPOINT/action_expert.safetensors" ]] || return 1
    [[ -f "$FINAL_METRICS" ]] || return 1
    "$PYTHON_BIN" - "$FINAL_METRICS" <<'PY'
import json
import math
import sys

with open(sys.argv[1], encoding="utf-8") as source:
    records = [json.loads(line) for line in source if line.strip()]
if not records or records[-1].get("step") != 19424:
    raise SystemExit(1)
for key in (
    "loss",
    "total_loss",
    "ar_loss",
    "flow_matching_loss",
    "vlm_grad_norm",
    "difference_query_grad_norm",
    "action_expert_grad_norm",
):
    value = records[-1].get(key)
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        raise SystemExit(1)
PY
}

gpus_are_free() {
    local gate_result
    gate_result=$(PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" -m utils.gpu_resource_gate \
        --visible-devices "${ZR0_CUDA_VISIBLE_DEVICES:-0,1,2,3}" --log "$FULL_LOG_DIR/gpu_gate.jsonl") || return
    printf '%s\n' "$gate_result"
    ZR0_CUDA_VISIBLE_DEVICES=$("$PYTHON_BIN" -c \
        'import json,sys; print(json.loads(sys.argv[1])["cuda_visible_devices"])' "$gate_result") || return
    export ZR0_CUDA_VISIBLE_DEVICES
    export CUDA_VISIBLE_DEVICES="$ZR0_CUDA_VISIBLE_DEVICES"
}

wait_count=0
until final_checkpoint_ready; do
    if (( wait_count % 10 == 0 )); then
        printf '[%s] waiting for complete Joint step-19424\n' "$(date --iso-8601=seconds)"
    fi
    wait_count=$((wait_count + 1))
    sleep "$POLL_SECONDS"
done

gpus_are_free

printf '[%s] final checkpoint and GPU gates passed; starting LIBERO preflight\n' "$(date --iso-8601=seconds)"
    ZR0_LIBERO_ACTION_HORIZON="$FINETUNE_ACTION_HORIZON" \
        bash "$ROOT_DIR/scripts/run_libero_wo_ecot_pt.sh" preflight difference_query_pretrained

"$PYTHON_BIN" -m pytest -q \
    "$ROOT_DIR/tests/test_libero_wo_ecot_pt_launcher.py" \
    "$ROOT_DIR/tests/test_train_resume.py" \
    "$ROOT_DIR/tests/test_optimizer_step_loss.py" \
    "$ROOT_DIR/tests/test_action_expert_config.py" \
    "$ROOT_DIR/tests/test_query_ar_joint_checkpoint.py"

timestamp=$(date +%Y%m%d-%H%M%S)
smoke_output="$ROOT_DIR/outputs/smoke/libero-dq32-tabletop-v3-joint-init-$timestamp"
smoke_run_name="libero-dq32-tabletop-v3-joint-init-smoke-$timestamp"
smoke_log="$FULL_LOG_DIR/$smoke_run_name.log"

printf '[%s] starting isolated real smoke: %s\n' "$(date --iso-8601=seconds)" "$smoke_output"
gpus_are_free
ZR0_OUTPUT_DIR="$smoke_output" \
ZR0_MAX_TRAIN_STEPS=2 \
ZR0_SAVE_STEP_INTERVAL=1 \
ZR0_RUN_NAME="$smoke_run_name" \
    ZR0_LIBERO_ACTION_HORIZON="$FINETUNE_ACTION_HORIZON" \
    bash "$ROOT_DIR/scripts/run_libero_wo_ecot_pt.sh" train difference_query_pretrained \
    2>&1 | tee -a "$smoke_log"

gpus_are_free
ZR0_OUTPUT_DIR="$smoke_output" \
ZR0_MAX_TRAIN_STEPS=3 \
ZR0_SAVE_STEP_INTERVAL=1 \
    ZR0_LIBERO_ACTION_HORIZON="$FINETUNE_ACTION_HORIZON" \
    bash "$ROOT_DIR/scripts/run_libero_wo_ecot_pt.sh" resume difference_query_pretrained \
    2>&1 | tee -a "$smoke_log"

"$PYTHON_BIN" - "$smoke_output" "$smoke_log" "$FINAL_CHECKPOINT" \
    "$SOURCE_ACTION_HORIZON" "$FINETUNE_ACTION_HORIZON" <<'PY'
import json
import math
import sys
from pathlib import Path

output = Path(sys.argv[1])
log = Path(sys.argv[2]).read_text(encoding="utf-8")
source_checkpoint = Path(sys.argv[3])
all_records = [
    json.loads(line)
    for line in (output / "training_metrics.jsonl").read_text(encoding="utf-8").splitlines()
    if line.strip()
]
records = [record for record in all_records if "total_loss" in record]
if [record.get("step") for record in records] != [1, 2, 3]:
    raise SystemExit(f"unexpected smoke steps: {[record.get('step') for record in records]}")
for record in records:
    for key in (
        "loss",
        "total_loss",
        "flow_matching_loss",
        "vlm_grad_norm",
        "difference_query_grad_norm",
        "action_expert_grad_norm",
    ):
        value = record.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise SystemExit(f"non-finite smoke metric at step {record.get('step')}: {key}={value}")
for filename in (
    "initialization_manifest_fresh.json",
    "initialization_manifest_resume.json",
    "step-3/zr0_checkpoint_metadata.json",
    "latest-model-optimizer-lr/scheduler.pt",
):
    if not (output / filename).is_file():
        raise SystemExit(f"missing smoke artifact: {filename}")
if log.count(
    "prepared training scale: world_size=4, micro_batch=16, "
    "gradient_accumulation_steps=1, nominal_global_batch=64"
) < 2:
    raise SystemExit("fresh/resume runtime batch assertions were not both observed")
fresh = json.loads(
    (output / "initialization_manifest_fresh.json").read_text(encoding="utf-8")
)
resume = json.loads(
    (output / "initialization_manifest_resume.json").read_text(encoding="utf-8")
)
if fresh["difference_query"]["source"] != "checkpoint":
    raise SystemExit("smoke did not load Difference Query from checkpoint")
if fresh["action_expert"]["source"] != "checkpoint":
    raise SystemExit("smoke did not load Action Expert from checkpoint")
source_config = json.loads(
    (source_checkpoint / "action_expert_config.json").read_text(encoding="utf-8")
)
saved_config = json.loads(
    (output / "step-3/action_expert_config.json").read_text(encoding="utf-8")
)
fresh_config = fresh["action_expert"]["config"]
resume_config = resume["action_expert"]["config"]
source_horizon = int(sys.argv[4])
target_horizon = int(sys.argv[5])
if source_config.get("action_horizon") != source_horizon:
    raise SystemExit(f"smoke source Action Expert horizon is not {source_horizon}")
if saved_config.get("action_horizon") != target_horizon:
    raise SystemExit(f"smoke checkpoint did not save resolved horizon {target_horizon}")
if (
    fresh_config.get("source_action_horizon") != source_horizon
    or fresh_config.get("resolved_action_horizon") != target_horizon
    or fresh_config.get("action_horizon_overridden") is not True
):
    raise SystemExit(f"unexpected fresh horizon resolution: {fresh_config}")
if (
    resume_config.get("source_action_horizon") != target_horizon
    or resume_config.get("resolved_action_horizon") != target_horizon
    or resume_config.get("action_horizon_overridden") is not False
):
    raise SystemExit(f"unexpected resume horizon resolution: {resume_config}")
PY

if [[ -e "$FORMAL_OUTPUT" ]]; then
    echo "Refusing to overwrite existing formal output: $FORMAL_OUTPUT" >&2
    exit 1
fi

formal_timestamp=$(date +%Y%m%d-%H%M%S)
formal_run_name="qwen3vl2b-libero-wo-ecot-dq32-tabletop-v3-joint-init-seed42-$formal_timestamp"
formal_log="$FULL_LOG_DIR/$formal_run_name.log"
printf '[%s] smoke/resume passed; starting formal LIBERO fine-tuning\n' "$(date --iso-8601=seconds)"
gpus_are_free
ZR0_RUN_NAME="$formal_run_name" \
    bash "$ROOT_DIR/scripts/run_libero_wo_ecot_pt.sh" train difference_query_pretrained \
    2>&1 | tee -a "$formal_log"
