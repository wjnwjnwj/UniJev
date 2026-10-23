#!/usr/bin/env bash
# Stage 1: soft-label SFT (decision head + LoRA + vision merger).
#
#   bash train/run_sft.sh --model <model_dir> --gpus <ids> [--data train.jsonl] [options]
#
# Required
#   --model            base model directory (any size; architecture read from its config.json)
#   --gpus             GPU ids, e.g. 0,1,2,3,4,5,6,7 (one rank per GPU)
# Common options (defaults in brackets; `--help` lists every option)
#   --data             training JSONL (default: ../JeV-Data/trimodal_100k_train.jsonl)
#   --init-ckpt        optional LoRA/head checkpoint to start from (dir with checkpoint.pt,
#                      a run dir -> latest/, or a name under ../cora-weights-hf/)
#   --epochs N         passes over the data (default 1)
#   --max-steps N      stop after N updates (default 0 = run all epochs)
#   --save-every N     checkpoint every N updates (default 100)
#   --batch-size N     questions per update, summed over all GPUs (default 0 = use --tokens-per-step)
#   --tokens-per-step  global compute tokens per update when --batch-size is 0 (default 262144)
#   --head-lr --lora-lr --merger-lr --weight-decay --grad-clip (2e-4 / 5e-5 / 1e-5 / 0.01 / 1.0)
#   --lr-schedule constant|cosine|linear  --warmup-steps N  --min-lr-ratio R
#   --lora-rank 32 --lora-alpha 64 --projection-dim 256 --soft-label-prob 0.9 --seed 42
#   --grad-reduce question_mean|rank_mean (default question_mean; rank_mean = original 2B recipe)
#   --max-frames 128 --min-pixels 3136 --max-pixels 65536 --max-length 32768
#   --attn sdpa|eager [sdpa]  --grad-ckpt 0|1 [1]  --cpu-offload 0|1 [0]  --skip-nonfinite-update 0|1 [1]
#   --cpu-workers N    DataLoader workers per GPU (default 6)   --prefetch N   --threads N
#   --output-dir DIR   runs go to DIR/<run> (default: output/)   --name TAG (added to the run name)
#   --resume RUN_DIR   continue RUN_DIR from latest/ with its original settings (same #GPUs)
#   --check-data       validate labels and media only (CPU, no GPU needed)
#   --progress 0|1     progress bar on the console [1]   --log-every N  also print the JSON line every N updates [10]
#
# Output: output/sft_<model>_<data>[_<name>]_<yyyymmdd_hhmmss>/
#   run_config.json, init_checkpoint.json, logs/{train.log,train.jsonl,launch_args.json,skipped.*},
#   step-XXXXX/ (checkpoint.pt, config.json, training_state.pt, manifest.json), latest -> newest, finished.json
set -euo pipefail

STAGE=sft
MODULE=train.sft
SCRIPT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/$(basename -- "${BASH_SOURCE[0]}")"
REPO="$(cd -- "$(dirname -- "$SCRIPT")/.." && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "$REPO/.." && pwd)}"
GPUS="${GPUS:-}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO/output}"
NAME="${NAME:-}"
RESUME_DIR=""
if [[ -z "${PYTHON:-}" ]]; then
    PYTHON=python
    [[ -x "$PROJECT_ROOT/../valen-env/bin/python" ]] && PYTHON="$PROJECT_ROOT/../valen-env/bin/python"
fi
# Relative paths stay relative to the caller's working directory.
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

# --gpus/--output-dir/--name/--resume/--python are handled here; everything else goes to Python.
forward=() model="" data="" init="" check=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpus) GPUS="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --name) NAME="$2"; shift 2 ;;
        --resume) RESUME_DIR="$2"; shift 2 ;;
        --python) PYTHON="$2"; shift 2 ;;
        --model) model="$2"; forward+=("$1" "$2"); shift 2 ;;
        --data) data="$2"; forward+=("$1" "$2"); shift 2 ;;
        --init-ckpt) init="$2"; forward+=("$1" "$2"); shift 2 ;;
        --check-data) check=1; forward+=("$1"); shift ;;
        -h|--help)
            sed -n '2,/^set -euo/p' "$SCRIPT" | sed '$d' | sed 's/^# \{0,1\}//'
            echo; echo "All Python-level options:"
            exec "$PYTHON" -m "$MODULE" --help ;;
        *) forward+=("$1"); shift ;;
    esac
done

if [[ -n "$RESUME_DIR" ]]; then
    # Replay the original arguments from the original working directory;
    # the trainer refuses to resume if any setting changed.
    RUN_DIR="$(cd -- "$RESUME_DIR" && pwd)"
    ARGS_FILE="$RUN_DIR/logs/launch_args.json"
    mapfile -t forward < <("$PYTHON" -c 'import json, sys; print("\n".join(json.load(open(sys.argv[1]))["forwarded"]))' "$ARGS_FILE")
    cd -- "$("$PYTHON" -c 'import json, sys; print(json.load(open(sys.argv[1]))["cwd"])' "$ARGS_FILE")"
    forward+=(--resume)
else
    [[ -n "$model" ]] || { echo "--model is required (base model directory)" >&2; exit 1; }
    if [[ -z "$data" ]]; then
        data="$PROJECT_ROOT/JeV-Data/trimodal_100k_train.jsonl"
        forward+=(--data "$data")
    fi
fi

if [[ $check == 1 ]]; then
    exec "$PYTHON" -m "$MODULE" "${forward[@]}" --output "${RUN_DIR:-/tmp/unijev_check_data}"
fi

[[ -n "$GPUS" ]] || { echo "--gpus is required, e.g. --gpus 0,1,2,3" >&2; exit 1; }
[[ "$GPUS" =~ ^[0-9]+(,[0-9]+)*$ ]] || { echo "--gpus must look like 0 or 0,1,3" >&2; exit 1; }
IFS=',' read -r -a GPU_LIST <<< "$GPUS"
NPROC="${#GPU_LIST[@]}"

if [[ -z "$RESUME_DIR" ]]; then
    RUN_DIR="$OUTPUT_DIR/${STAGE}_$(basename -- "${model%/}")_$(basename -- "$data" .jsonl)${NAME:+_$NAME}_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$RUN_DIR/logs"
    "$PYTHON" -c 'import json, os, sys; json.dump({"gpus": sys.argv[2], "cwd": os.getcwd(), "forwarded": sys.argv[3:]}, open(sys.argv[1], "w"), indent=1)' \
        "$RUN_DIR/logs/launch_args.json" "$GPUS" "${forward[@]}"
fi

export CUDA_VISIBLE_DEVICES="$GPUS"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
echo "stage=$STAGE gpus=$GPUS (nproc=$NPROC)"
echo "run dir: $RUN_DIR"
echo "log:     $RUN_DIR/logs/train.log   (per-step JSON: $RUN_DIR/logs/train.jsonl)"
"$PYTHON" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="$NPROC" \
    -m "$MODULE" "${forward[@]}" --output "$RUN_DIR" --command-line "$(basename -- "$SCRIPT") ${forward[*]}" \
    2>&1 | tee -a "$RUN_DIR/logs/train.log"
