# CORA

**Confidence calibration for decisions across text, images, and video.**

[🤗 Model Weights](https://huggingface.co/wjnwjnwj/UniJev) · [Quick Start](#quick-start) · [Training](#training) · [Data](#data)

CORA trains multimodal decision models to express confidence that reflects how often they are correct. Given text, an image, or a video and a set of candidate answers, the model scores the candidates directly and returns an explicit probability distribution. This gives applications a decision and a confidence score to use for routing, filtering, or requesting human review.

CORA grounds confidence in **the accuracy of the model's own current predictions**. Predictions with similar confidence share a calibration target: their observed accuracy. These targets are refreshed as the model learns, keeping confidence training tied to its evolving decision behavior.

- **Three modalities, one interface.** Text, image, and video tasks share a model and training pipeline, with support for multiple-choice questions, binary judgments, and ordinal scores.
- **Direct candidate scoring.** A shared decision head produces answer probabilities without generating and parsing a text response.
- **Two-stage training.** Soft-label SFT builds the decision capability; CORA then trains confidence against observed accuracy.
- **2B and 9B models.** Both sizes use Qwen3.5 backbones, with LoRA adaptation, a trainable vision merger, and a lightweight decision head.

This repository includes the SFT and CORA training code, configurations for both model sizes, launch scripts, and 10 training examples per modality.

## Models

Model weights: **[UniJev on Hugging Face](https://huggingface.co/wjnwjnwj/UniJev)**.

| Backbone   | Modalities         | SFT config                               | CORA config                                |
| ---------- | ------------------ | ---------------------------------------- | ------------------------------------------ |
| Qwen3.5-2B | Text, image, video | [sft_2b.json](train/configs/sft_2b.json) | [cora_2b.json](train/configs/cora_2b.json) |
| Qwen3.5-9B | Text, image, video | [sft_9b.json](train/configs/sft_9b.json) | [cora_9b.json](train/configs/cora_9b.json) |

Training uses a local Qwen3.5 base model directory containing its weights, tokenizer, and processor. Set `MODEL_PATH` to the matching 2B or 9B directory. Base models and trained checkpoints are stored separately from this code repository.

## Quick Start

### Install

Use Python 3.10 and a CUDA GPU with BF16 support. Run the following commands from the repository root:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.6.0 torchvision==0.21.0 \
  --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
```

### Run the examples

Try the complete SFT → CORA workflow on a single GPU:

```bash
export MODEL_PATH=/absolute/path/to/Qwen3.5-2B
CUDA_VISIBLE_DEVICES=0 bash train/run_sft_2b.sh --smoke
CUDA_VISIBLE_DEVICES=0 bash train/run_cora_2b.sh --smoke
```

Each stage runs one training update on the bundled examples. CORA automatically loads the checkpoint from the preceding SFT step. For 9B, change the model path and use `run_sft_9b.sh` and `run_cora_9b.sh`.

To check the example data without loading a model or using a GPU:

```bash
bash train/run_cora_2b.sh --smoke --check-data
```

## Training

Both model sizes follow the same two stages:

1. **Soft-label SFT** trains the candidate distribution on labeled text, image, and video examples.
2. **CORA** samples questions across modalities, groups predictions by confidence, and trains each confidence score toward its group's current accuracy.

Set the paths to your full training JSONL file and media directory:

```bash
export TRAIN_DATA=/absolute/path/to/trimodal_full_train.jsonl
export MEDIA_ROOT=/absolute/path/to/media/root
```

### Qwen3.5-2B

```bash
export MODEL_PATH=/absolute/path/to/Qwen3.5-2B

# Stage 1: soft-label SFT on 8 GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash train/run_sft_2b.sh

# Stage 2: CORA on 4 GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3 bash train/run_cora_2b.sh
```

### Qwen3.5-9B

```bash
export MODEL_PATH=/absolute/path/to/Qwen3.5-9B

# Stage 1: soft-label SFT on 4 GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3 bash train/run_sft_9b.sh

# Stage 2: CORA on 4 GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3 bash train/run_cora_9b.sh
```

The launch scripts select the corresponding configuration and SFT checkpoint automatically. Full settings are in [`train/configs/`](train/configs). For 9B training, we recommend GPUs with 80 GB of memory and sufficient host memory for SFT activation offload.

To start CORA from an existing SFT checkpoint, set `SFT_CHECKPOINT`:

```bash
SFT_CHECKPOINT=/absolute/path/to/sft/checkpoint-directory \
  CUDA_VISIBLE_DEVICES=0,1,2,3 bash train/run_cora_2b.sh
```

Outputs are saved under `outputs/` by default. Use `OUTPUT` to choose another directory, or append `--resume` to continue an interrupted run with the same configuration. Explicit path settings also apply to smoke runs; unset them to return to the bundled examples.

## Data

Training data uses a shared JSONL schema for all three modalities. Each record contains a context, one or more questions with candidate answers, and target probabilities. Supported question types are `choice` (multiple choice), `noul` (true/false), and `score` (ordered descriptions).

The [`data/`](data) directory includes **10 text, 10 image, and 10 video records**, with small images and video frame caches ready for the example workflow. See [`examples.jsonl`](data/examples.jsonl) for all three modalities and [`manifest.json`](data/manifest.json) for sample sources.

<details>
<summary>Example record and media paths</summary>

```json
{
  "group_id": "unique-source-group",
  "request": {
    "state": "The store closes at 6 pm.",
    "questions": {
      "q1": {
        "type": "choice",
        "instructions": "When does the store close?",
        "criteria": {"A": "6 pm", "B": "8 pm"}
      }
    }
  },
  "targets": {"q1": {"probabilities": {"A": 1.0, "B": 0.0}}}
}
```

For image and video inputs, `state` can be an object containing a `messages` list with `text`, `image_url`, and `video_url` content. Relative media paths are resolved against `MEDIA_ROOT`; absolute paths are also supported.

</details>

## Repository Structure

```text
data/                  # Example records and media
train/
  common/              # Shared model and data utilities
  sft.py               # Soft-label SFT
  cora/                # CORA training
  configs/             # 2B and 9B training recipes
  run_*.sh             # Launch scripts for both stages
requirements.txt       # Python dependencies
```
