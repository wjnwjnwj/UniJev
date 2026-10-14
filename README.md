# CORA: Confidence–Outcome Reliability Alignment

Training code for **UniJev + CORA** — trimodal (text/image/video) soft-label SFT followed by
**supervised calibration training**: no RL, no action sampling, no post-hoc lookup at inference.

Each CORA optimizer step samples one modality (weighted by its share of the training set), draws a
group of 256 distinct questions from it, forwards the group, bins the questions by their current
maximum candidate probability (10 equal-width bins, minimum support 20), takes the detached per-bin
accuracy of this very pass as the target, and applies a squared loss aligning each supported
question's confidence to its bin target. The next group re-forwards and re-bins. Trainable scope is
the SFT delta (language LoRA + vision merger + decision head); the pretrained backbone stays frozen.

## Repository layout

| Path | Contents |
|---|---|
| `cora_train_v2/` | CORA training, evaluation and unit tests (the method) |
| `campaign_train_v2/` | Model building, compiler, distributed runtime used by CORA |
| `campaign_train_v1/` | Shared schema/report utilities, plus the 9B soft-label SFT trainer (`objective: sft_soft`) |
| `videojev_soft_label/` | 2B soft-label SFT trainer (self-contained) |
| `confidence_group_v1/` | Binning statistics and confidence metrics reused by CORA |
| `configs/` | Exact configurations of the released runs |
| `data/` | 10 stripped-down example records per modality, with their media under `data/media/` |
| `versions/release_v1.json` | SHA-256 pins of every source file in this release (verified at startup) |
| `weights/README.md` | Released checkpoints (HuggingFace links + SHA-256) |

This is a minimal release: ablation harnesses, evaluation queues, result-sync tooling and all
baseline methods were removed. `campaign_train_v1/objectives.py` therefore accepts only
`sft_soft`; the config vocabulary still names the excluded campaign recipes, and selecting one
raises immediately.

## Setup

```bash
pip install -r requirements.txt
```

You also need the frozen backbones (`Qwen3.5-2B`, `Qwen3.5-9B`) and the released checkpoints — see
`weights/README.md`. Configs reference the backbone via `model_path`.

## Data

`data/examples_{text,image,video}_10.jsonl` hold 10 training records per modality in the exact
schema the trainers consume (`request.state.messages` + `questions` + one-hot `targets`; media
paths are relative to the JSONL's directory / the config's `media_root`). They demonstrate the
format only — the full training mixture (157,656 records / 160,608 questions) and the test suite
(3,124 questions) are not distributed here. The configs pin their SHA-256 (`input_sha256`), so a
reconstructed dataset can be verified byte-for-byte.

## Training

All commands run from the repository root.

**1. Soft-label SFT** (correct option 0.9, remainder split evenly):

```bash
# 2B (videojev_soft_label trainer; one epoch, checkpoint used: step-01150)
torchrun --standalone --nproc_per_node=8 -m videojev_soft_label.train \
    --config configs/sft_trimodal_full_2b_softlabel.json

# 9B (campaign trainer, objective sft_soft; fixed 500-step budget)
torchrun --standalone --nproc_per_node=8 -m campaign_train_v1.train \
    --config configs/campaign_train_v1/sft_softlabel_9b.json
```

**2. CORA** (initialized from the SFT checkpoints above; 100 steps, final checkpoint reported):

```bash
# CPU-only input validation first
python -m cora_train_v2.train --config configs/cora_train_v2/soft_sft_2b.json --check-data

# 2B / 9B
torchrun --standalone --nproc_per_node=4 -m cora_train_v2.train --config configs/cora_train_v2/soft_sft_2b.json
torchrun --standalone --nproc_per_node=4 -m cora_train_v2.train --config configs/cora_train_v2/soft_sft_9b.json
```

**3. Evaluation of the final CORA checkpoint (step-00100):**

```bash
torchrun --standalone --nproc_per_node=4 -m cora_train_v2.evaluate --config configs/cora_train_v2/soft_sft_2b.json
```

Reporting protocol: every trained row reports the fixed-budget final checkpoint; the internal
1% source-group holdout is training-time monitoring only.

## Tests

```bash
python -m unittest cora_train_v2.test_training confidence_group_v1.test_grouping -v
```

See `cora_train_v2/README.md` for a detailed method description (in Chinese).
