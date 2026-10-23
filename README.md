# CORA

**Confidence calibration for decisions across text, images, and video.**

[🌐 Live Demo](https://unijev-website.3323422990.workers.dev/) · [🤗 Model Weights](https://huggingface.co/wjnwjnwj/UniJev) · [Results](#results) · [Case Gallery](#case-gallery) · [Quick Start](#quick-start) · [Training](#training) · [Data](#data)

CORA trains multimodal decision models to express confidence that reflects how often they are correct. Given text, an image, or a video and a set of candidate answers, the model scores the candidates directly and returns an explicit probability distribution. This gives applications a decision and a confidence score to use for routing, filtering, or requesting human review.

CORA grounds confidence in **the accuracy of the model's own current predictions**. Predictions with similar confidence share a calibration target: their observed accuracy. These targets are refreshed as the model learns, keeping confidence training tied to its evolving decision behavior.

- **Three modalities, one interface.** Text, image, and video tasks share a model and training pipeline, with support for multiple-choice questions, binary judgments, and ordinal scores.
- **Direct candidate scoring.** A shared decision head produces answer probabilities without generating and parsing a text response.
- **Two-stage training.** Soft-label SFT builds the decision capability; CORA then trains confidence against observed accuracy.
- **2B and 9B models.** Both sizes use Qwen3.5 backbones, with LoRA adaptation, a trainable vision merger, and a lightweight decision head.

This repository includes the SFT and CORA training code, configurations for both model sizes, launch scripts, and 10 training examples per modality.

## Results

**At 2B, CORA reduces image calibration error by 50% and video calibration error by 32.5%, with per-modality accuracy changing by at most 0.3 percentage points.** These are relative reductions in expected calibration error (ECE) from the soft-label SFT initialization.

The paper evaluates **3,124 held-out questions**: 764 text, 1,858 image, and 502 video questions, with training and test data separated by source group. Both CORA scales are evaluated after **100 supervised calibration updates**. Accuracy is reported in percent; ECE uses **15 equal-width evaluation bins**, and lower is better.

### Soft-label SFT → CORA

<img src="docs/results/calibration.png" alt="SFT versus CORA ECE at 2B and 9B: image and video improve at both scales; text changes from 0.057 to 0.055 at 2B and from 0.040 to 0.048 at 9B." width="960">

| Scale | Modality | SFT accuracy ↑ | CORA accuracy ↑ | SFT ECE ↓ | CORA ECE ↓ |
| --- | --- | ---: | ---: | ---: | ---: |
| 2B | Text | 72.0 | 71.9 | 0.057 | **0.055** |
| 2B | Image | 83.3 | 83.6 | 0.060 | **0.030** |
| 2B | Video | 74.7 | 74.5 | 0.077 | **0.052** |
| 9B | Text | 80.8 | 81.0 | **0.040** | 0.048 |
| 9B | Image | 82.8 | 82.6 | 0.061 | **0.031** |
| 9B | Video | 77.5 | 77.5 | 0.077 | **0.063** |

Bold marks the lower ECE within each SFT/CORA pair. The largest gains are on image and video. At 9B, these gains persist with accuracy changes of at most 0.2 percentage points, while **text ECE increases from 0.040 to 0.048**. At 2B, text ECE is nearly unchanged.

Candidate-distribution negative log-likelihood (NLL, lower is better) also improves at 2B: **0.693 → 0.643** on text, **0.482 → 0.449** on image, and **0.648 → 0.620** on video.

### Alignment ablations

The following 2B ablations share the same soft-label SFT initialization, training mixture, and 100-update budget. Each variant changes one factor from the default: **10 training bins, 256 questions per group, and minimum bin support of 20**. Training bins are separate from the 15 bins used to evaluate ECE.

| Variant | Text ECE ↓ | Image ECE ↓ | Video ECE ↓ |
| --- | ---: | ---: | ---: |
| **CORA (default)** | **0.055** | 0.030 | 0.052 |
| Replace bin accuracy with bin mean confidence | 0.274 | 0.162 | 0.240 |
| Replace confidence bins with one global target | 0.158 | 0.127 | 0.090 |
| Update the decision head only | 0.083 | 0.032 | 0.061 |
| Reduce minimum bin support from 20 to 1 | 0.095 | **0.026** | **0.045** |

Removing **correctness supervision** or **confidence-conditioned targets** raises ECE in all three modalities. Updating only the decision head also trails the default. Bin support exposes a tradeoff: support 1 improves image and video ECE but worsens text ECE.

<details>
<summary><strong>Comparison with five local RLCD continuations (2B)</strong></summary>

All methods below start from the same soft-label SFT checkpoint. CORA uses 100 supervised calibration updates; the five RLCD recipes each use a 500-step budget. The training budgets therefore differ.

| Method | Continuation updates | Text ECE ↓ | Image ECE ↓ | Video ECE ↓ |
| --- | ---: | ---: | ---: | ---: |
| Shaped RLCD | 500 | 0.220 | 0.099 | 0.173 |
| GRPO | 500 | 0.234 | 0.114 | 0.183 |
| CtR (OpenJev-RLCD) | 500 | 0.095 | **0.025** | 0.054 |
| RLCD-lite | 500 | 0.250 | 0.143 | 0.220 |
| Rewarding Doubt | 500 | 0.230 | 0.107 | 0.178 |
| **CORA** | **100** | **0.055** | 0.030 | **0.052** |

CORA has the lowest text and video ECE among these local continuations. CtR has the lowest image ECE, while its text ECE rises from the SFT initialization's 0.057 to 0.095. These results support direct confidence-bin supervision as an effective calibration approach without a reinforcement stage.

</details>

Values come from the latest manuscript's main results and ablation tables. [Metric snapshot and figure source](docs/results/README.md).

## Case Gallery

These examples show recorded predictions from the paper's held-out evaluation. **SFT** is the 2B soft-label initialization; **CORA** is the 2B model after 100 calibration updates. Candidate probabilities are shown as percentages, rounded to one decimal place. The original probabilities, labels, and case identifiers are available in [the case snapshot](docs/cases/predictions.json).

### Text · Answering a science question

> What is the term for the force that brings objects toward the earth?

| Candidate answer | SFT | CORA |
| --- | ---: | ---: |
| centrifugal | 1.4% | 0.2% |
| light | 0.8% | 0.1% |
| weight | 23.0% | 14.8% |
| **gravity — ground truth** | **74.8%** | **84.9%** |

Both models select **gravity**. The shared decision head returns the candidate distribution directly.

### Image · Counting objects

<img src="docs/cases/image-counting.jpg" alt="A cat beside one patterned bowl, from the paper's image example I3." width="480">

**Question:** How many bowls does the cat have?

| Candidate answer | SFT | CORA |
| --- | ---: | ---: |
| 3 | 1.7% | 0.4% |
| 2 | 17.0% | 12.4% |
| **1 — ground truth** | **77.9%** | **86.5%** |
| 0 | 3.4% | 0.8% |

Both models choose **1 bowl**, with different confidence distributions over the alternatives.

### Video · Recognizing a scene

<img src="docs/cases/video-scene.gif" alt="Animated preview of six timestamped frames showing people on a tourist lift, from paper example V6." width="640">

The animation replays six sampled frames from the paper, with their source timestamps. It has no audio. [Open or download the MP4 preview](docs/cases/video-scene.mp4).

**Question:** Where is this happening?

| Candidate answer | SFT | CORA |
| --- | ---: | ---: |
| **tourist lift. — ground truth** | 42.3% | **56.0%** |
| park. | **50.3%** | 41.9% |
| forest. | 2.9% | 0.9% |
| bowling alley. | 1.8% | 0.5% |
| stage. | 2.7% | 0.8% |

SFT selects **park**; CORA selects the correct answer, **tourist lift**. Calibration training updates the model, so it can change the selected answer as well as its confidence.

<details>
<summary><strong>A remaining error · Reading a chart</strong></summary>

<img src="docs/cases/image-chart-error.png" alt="A bar chart of the Columbus metropolitan area's population, from paper example I7." width="640">

**Question:** How many people lived in the Columbus metropolitan area in 2019?

**Ground truth:** 2,104,194.

| Model | Selected answer | Confidence | Correct? |
| --- | --- | ---: | --- |
| SFT | 2,104,195 | 62.1% | No |
| CORA | 2,104,195 | 61.3% | No |

This prediction remains incorrect after calibration. Calibration describes groups of predictions; individual answers can still be wrong.

</details>

These selected cases illustrate model behavior. The [aggregate results](#results) measure accuracy and calibration across the full evaluation suite. [Case sources and preview details](docs/cases/README.md).

## Models

Model weights: **[UniJev on Hugging Face](https://huggingface.co/wjnwjnwj/UniJev)**.

| Backbone | Modalities |
| --- | --- |
| Qwen3.5-2B | Text, image, video |
| Qwen3.5-9B | Text, image, video |

Training is not tied to these sizes: `--model` takes any Qwen-VL-style base model directory (Qwen2-VL, Qwen2.5-VL, Qwen3-VL, Qwen3.5), and the decision head is sized from that model's own `config.json`. Base models and trained checkpoints are stored separately from this code repository.

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

Check the bundled examples (CPU only, no model weights or GPU needed):

```bash
bash train/run_sft.sh --model /abs/path/Qwen3.5-2B --data data/examples.jsonl --check-data
```

Short SFT → CORA run on the bundled examples on one GPU:

```bash
bash train/run_sft.sh  --model /abs/path/Qwen3.5-2B --data data/examples.jsonl --gpus 0 \
    --max-steps 2 --save-every 1 --batch-size 4
bash train/run_cora.sh --model /abs/path/Qwen3.5-2B --data data/examples.jsonl --gpus 0 \
    --init-ckpt output/<sft run folder> --max-steps 1 --group-questions 24 --min-bin-count 1
```

## Training

Training has two stages, each launched by its own script:

1. **Soft-label SFT** (`train/run_sft.sh`) trains the candidate distribution on labeled text, image and video examples.
2. **CORA** (`train/run_cora.sh`) samples questions across modalities, groups predictions into 15 equal-width confidence bins (the same bins as `test/eval.py`), and trains each confidence toward its bin's current accuracy.

No validation runs during or after training; evaluate saved checkpoints with `test/run_eval.sh`.

```bash
# Stage 1: SFT on GPUs 0-7, checkpoint every 50 updates
bash train/run_sft.sh \
  --model /pfs/siqingyi/code/token_credit/wjn/unijev/models/Qwen3.5-2B \
  --gpus 0,1,2,3,4,5,6,7 \
  --epochs 1 \
  --tokens-per-step 1048576 \
  --grad-reduce rank_mean \
  --lora-rank 32 --lora-alpha 64 --projection-dim 256 \
  --head-lr 2e-4 --lora-lr 5e-5 --merger-lr 1e-5 \
  --weight-decay 0.01 --grad-clip 1.0 \
  --lr-schedule cosine --warmup-steps 40 --min-lr-ratio 0.1 \
  --max-steps 1000 \
  --soft-label-prob 0.9 \
  --max-frames 128 --min-pixels 3136 --max-pixels 65536 --max-length 32768 \
  --attn sdpa --grad-ckpt 0 --skip-nonfinite-update 1 \
  --cpu-workers 16 \
  --save-every 50 

bash train/run_sft.sh \
  --model /pfs/siqingyi/code/token_credit/wjn/unijev/models/Qwen3.5-9B \
  --gpus 0,1,2,3,4,5,6,7 \
  --epochs 1 \
  --tokens-per-step 524288 \
  --grad-reduce question_mean \
  --lora-rank 64 --lora-alpha 128 --projection-dim 256 \
  --head-lr 2e-4 --lora-lr 5e-5 --merger-lr 1e-5 \
  --weight-decay 0.01 --grad-clip 1.0 \
  --lr-schedule cosine --warmup-steps 80 --min-lr-ratio 0.1 \
  --soft-label-prob 0.9 \
  --max-frames 128 --min-pixels 3136 --max-pixels 65536 --max-length 32768 \
  --attn sdpa --grad-ckpt 1 --skip-nonfinite-update 1 \
  --cpu-workers 16 \
  --save-every 100

# Stage 2: CORA on GPUs 0-3 from an SFT checkpoint (a run folder resolves to its latest/)
bash train/run_cora.sh --model /abs/path/Qwen3.5-2B --gpus 0,1,2,3 \
    --init-ckpt output/sft_Qwen3.5-2B_trimodal_100k_train_<timestamp>/step-01150 \
    --data /abs/path/trimodal_100k_train.jsonl --max-steps 100 --save-every 10
```

- Batch size: SFT `--batch-size N` = questions per update summed over all GPUs (0 = fixed token budget `--tokens-per-step`); CORA `--group-questions N` = questions binned and updated together.
- `--init-ckpt` is optional for SFT (continue from a LoRA/head checkpoint) and required for CORA.
- All settings are command-line flags with defaults; `bash train/run_sft.sh --help` lists them.
- Each run writes `output/<stage>_<model>_<data>[_<name>]_<yyyymmdd_hhmmss>/` with `run_config.json`, `logs/` (`train.log`, per-update `train.jsonl`, CORA `bins.jsonl`, skipped records) and `step-XXXXX/` checkpoints every `--save-every` updates.
- `--resume <run folder>` continues an interrupted run with its original settings on the same number of GPUs.

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
docs/cases/            # README gallery, saved predictions, and preview assets
docs/results/          # Paper metrics, calibration comparison, and figure source
train/
  common/              # Shared model, data and launch utilities
  sft.py               # Soft-label SFT
  cora/                # CORA training
  run_sft.sh           # Stage 1 launch script
  run_cora.sh          # Stage 2 launch script
test/
  eval.py, run_eval.sh # Validation-set evaluation
requirements.txt       # Python dependencies
```

## Cite us

```bib
@misc{wang2026unijev,
  title         = {{UniJev}: Separating Accuracy from Calibration in Trimodal System One Decision Models},
  author        = {Wang, Jianing and Jian, Youwen and Wang, Derong and Yu, Jinyang and Si, Qingyi and Qin, Chuanyu and Xiao, Yanghua and Guo, Hongcheng},
  year          = {2026},
  eprint        = {2610.25784},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  doi           = {10.48550/arXiv.2610.25784},
  url           = {https://arxiv.org/abs/2610.25784},
  note          = {21 pages, 8 figures}
}
```