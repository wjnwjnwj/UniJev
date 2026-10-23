#!/usr/bin/env bash
# One-command evaluation of a UniJev checkpoint on the trimodal validation set.
#
# Usage:
#   bash test/run_eval.sh --model <2b|9b|/path/to/Qwen3.5-*> --ckpt <name|dir> --gpus <ids> [options]
#
# Required
#   --model          2b | 9b | absolute path of the Qwen3.5 base model
#   --ckpt           checkpoint name under --ckpt-root (e.g. cora-2b, sft_2b_hard/step-00600),
#                    or an absolute directory containing checkpoint.pt (a run dir uses best/ or latest/)
#   --gpus           GPU ids to use, comma separated, e.g. 0 | 0,1 | 4,5,6,7
# Optional
#   --val            validation JSONL           (default: JeV-Data/trimodal_val.jsonl)
#   --output-dir     where run folders go       (default: test/output)
#   --procs-per-gpu  model replicas per GPU     (default: 2b -> 3, 9b -> 1)
#   --cpu-workers    media-decoding workers per replica (default: 4)
#   --threads        torch CPU threads per replica      (default: 2)
#   --ckpt-root      checkpoint root            (default: ../cora-weights-hf)
#   --media-root     base for relative media paths (default: ../JeV-Data)
#   --max-frames / --max-pixels   override the checkpoint's video/image budget (0 = keep)
#   --limit N        only the first N records (quick check)
#   --attn / --dtype sdpa|eager / bf16|fp32
# Every flag also works as an env var (MODEL, CKPT, GPUS, VAL, OUTPUT_DIR, PROCS_PER_GPU, ...).
#
# Output: OUTPUT_DIR/<model>_<ckpt>_<yyyymmdd_hhmmss>/
#   eval_config.json   full evaluation config (model, ckpt + sha256, data + sha256, inference, launch, env)
#   predictions.jsonl  one row per question: logits and probability of EVERY candidate, prediction,
#                      confidence, target, gold, correct, nll, brier
#   metrics.json       overall / by modality / task / modality+task / domain:
#                      acc, mean confidence, ECE(15 bins), NLL, Brier, AURC, reliability bins
#   summary.txt        human-readable table
#   errors.jsonl       records that failed (empty when everything ran)
#   logs/ shards/      per-replica logs and partial outputs
set -euo pipefail

TEST_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd -- "$TEST_DIR/.." && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "$REPO/.." && pwd)}"   # holds models/, cora-weights-hf/, JeV-Data/

MODEL="${MODEL:-}"
CKPT="${CKPT:-}"
GPUS="${GPUS:-}"
CKPT_ROOT="${CKPT_ROOT:-$PROJECT_ROOT/cora-weights-hf}"
VAL="${VAL:-$PROJECT_ROOT/JeV-Data/trimodal_val.jsonl}"
MEDIA_ROOT="${MEDIA_ROOT:-$PROJECT_ROOT/JeV-Data}"
OUTPUT_DIR="${OUTPUT_DIR:-$TEST_DIR/output}"
PROCS_PER_GPU="${PROCS_PER_GPU:-}"
CPU_WORKERS="${CPU_WORKERS:-4}"
THREADS="${THREADS:-2}"
MAX_FRAMES="${MAX_FRAMES:-0}"
MAX_PIXELS="${MAX_PIXELS:-0}"
LIMIT="${LIMIT:-0}"
ATTN="${ATTN:-sdpa}"
DTYPE="${DTYPE:-bf16}"
if [[ -z "${PYTHON:-}" ]]; then
    PYTHON=python
    [[ -x "$PROJECT_ROOT/../valen-env/bin/python" ]] && PYTHON="$PROJECT_ROOT/../valen-env/bin/python"
fi

usage() { sed -n '2,36p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }
ARGS_LINE="$0 $*"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) MODEL="$2"; shift 2 ;;
        --ckpt) CKPT="$2"; shift 2 ;;
        --gpus) GPUS="$2"; shift 2 ;;
        --ckpt-root) CKPT_ROOT="$2"; shift 2 ;;
        --val) VAL="$2"; shift 2 ;;
        --media-root) MEDIA_ROOT="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --procs-per-gpu) PROCS_PER_GPU="$2"; shift 2 ;;
        --cpu-workers) CPU_WORKERS="$2"; shift 2 ;;
        --threads) THREADS="$2"; shift 2 ;;
        --max-frames) MAX_FRAMES="$2"; shift 2 ;;
        --max-pixels) MAX_PIXELS="$2"; shift 2 ;;
        --limit) LIMIT="$2"; shift 2 ;;
        --attn) ATTN="$2"; shift 2 ;;
        --dtype) DTYPE="$2"; shift 2 ;;
        -h|--help) usage 0 ;;
        *) echo "Unknown argument: $1" >&2; usage 1 ;;
    esac
done
missing=()
[[ -n "$MODEL" ]] || missing+=(--model)
[[ -n "$CKPT" ]] || missing+=(--ckpt)
[[ -n "$GPUS" ]] || missing+=(--gpus)
if [[ ${#missing[@]} -gt 0 ]]; then
    echo "Missing required: ${missing[*]}" >&2; usage 1
fi
[[ "$GPUS" =~ ^[0-9]+(,[0-9]+)*$ ]] || { echo "--gpus must look like 0 or 0,1,3" >&2; exit 1; }
if [[ -z "$PROCS_PER_GPU" ]]; then
    case "${MODEL,,}" in *9b*) PROCS_PER_GPU=1 ;; *) PROCS_PER_GPU=3 ;; esac
fi

IFS=',' read -r -a GPU_LIST <<< "$GPUS"
NUM_SHARDS=$(( ${#GPU_LIST[@]} * PROCS_PER_GPU ))

cd "$REPO"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$("$PYTHON" "$TEST_DIR/eval.py" config \
    --model "$MODEL" --ckpt "$CKPT" --ckpt-root "$CKPT_ROOT" --project-root "$PROJECT_ROOT" \
    --val "$VAL" --media-root "$MEDIA_ROOT" --output-dir "$OUTPUT_DIR" --timestamp "$TIMESTAMP" \
    --gpus "$GPUS" --procs-per-gpu "$PROCS_PER_GPU" --cpu-workers "$CPU_WORKERS" --threads "$THREADS" \
    --dtype "$DTYPE" --attn "$ATTN" --max-frames "$MAX_FRAMES" --max-pixels "$MAX_PIXELS" \
    --limit "$LIMIT" --command-line "$ARGS_LINE" | tail -n 1)"
echo "run dir: $RUN_DIR"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false
pids=()
shard=0
for gpu in "${GPU_LIST[@]}"; do
    for (( r = 0; r < PROCS_PER_GPU; r++ )); do
        log="$RUN_DIR/logs/shard_$(printf %02d "$shard")_gpu${gpu}.log"
        CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS="$THREADS" "$PYTHON" "$TEST_DIR/eval.py" predict \
            --run-dir "$RUN_DIR" --shard "$shard" --num-shards "$NUM_SHARDS" \
            --cpu-workers "$CPU_WORKERS" --threads "$THREADS" > "$log" 2>&1 &
        pids+=("$!")
        shard=$(( shard + 1 ))
    done
done
trap 'kill "${pids[@]}" 2>/dev/null || true' INT TERM
echo "launched $NUM_SHARDS replica(s): GPU(s) $GPUS x $PROCS_PER_GPU per GPU, $CPU_WORKERS CPU workers each"
echo "progress: tail -f $RUN_DIR/logs/shard_*.log"

failed=0
for i in "${!pids[@]}"; do
    if ! wait "${pids[$i]}"; then
        echo "shard $i failed:" >&2
        tail -n 20 "$RUN_DIR"/logs/shard_"$(printf %02d "$i")"_gpu*.log >&2 || true
        failed=1
    fi
done

"$PYTHON" "$TEST_DIR/eval.py" merge --run-dir "$RUN_DIR"
echo "results: $RUN_DIR"
exit "$failed"
