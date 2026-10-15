#!/usr/bin/env bash
set -euo pipefail
stage="$1"
size="$2"
shift 2
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
python_bin="${PYTHON:-python}"
args=(--config "train/configs/${stage}_${size}.json")
[[ -z "${MODEL_PATH:-}" ]] || args+=(--model-path "$MODEL_PATH")
[[ -z "${TRAIN_DATA:-}" ]] || args+=(--data "$TRAIN_DATA")
[[ -z "${MEDIA_ROOT:-}" ]] || args+=(--media-root "$MEDIA_ROOT")
[[ -z "${OUTPUT:-}" ]] || args+=(--output "$OUTPUT")
if [[ "$stage" == cora && -n "${SFT_CHECKPOINT:-}" ]]; then
    args+=(--init-checkpoint "$SFT_CHECKPOINT")
fi
module="train.sft"
[[ "$stage" != cora ]] || module="train.cora.train"
workers=4
[[ "$stage" != sft || "$size" != 2b ]] || workers=8
for arg in "$@"; do
    if [[ "$arg" == --check-data || "$arg" == --smoke ]]; then
        exec "$python_bin" -m "$module" "${args[@]}" "$@"
    fi
done
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
exec "$python_bin" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="$workers" \
    -m "$module" "${args[@]}" "$@"
