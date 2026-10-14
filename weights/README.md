# Model weights

Weights are hosted on HuggingFace Hub (links below), not in this repository.
Every checkpoint is the *trainable delta* over the frozen Qwen3.5 backbone
(language LoRA r=32/alpha=64 + vision merger + decision head), stored as a single
`checkpoint.pt` next to the `config.json` of the run that produced it. Download the
backbone separately (`Qwen3.5-2B` / `Qwen3.5-9B`) and point `model_path` at it.

| Checkpoint | Role | File | SHA-256 |
|---|---|---|---|
| `sft-soft-2b-step01150` | Soft-label SFT, 2B (CORA-2B init) | `checkpoint.pt` (189 MB) | `7b783310d733fa5da471f6dbec4f4e1fdd532a0821eacbf8bd2772b0cf40ceca` |
| `sft-soft-9b-step00500` | Soft-label SFT, 9B (CORA-9B init) | `checkpoint.pt` (435 MB) | `ed3973f0319b6d550756c13d0f83c35dd81ce75bf050358a022d1c46d84d8eee` |
| `cora-modality-2b-step00100` | CORA (LoRA+merger+head delta), 2B | pending — training in progress | — |
| `cora-modality-9b-step00100` | CORA (LoRA+merger+head delta), 9B | pending — training in progress | — |

The SHA-256 values above are exactly the `checkpoint_sha256` pins inside
`configs/cora_train_v2/*.json`, so a downloaded checkpoint can be verified before CORA
training starts.

HuggingFace links: **TBD — fill in after upload.**

To use a downloaded SFT checkpoint as the CORA initialization, place it at the
`init_checkpoint` path named in the base config (or edit that path), e.g.:

    save-ckpt/sft/sft_2b_soft/step-01150/            <- sft-soft-2b-step01150
    save-ckpt/experiments/campaign_20261010_v1/runs/sft_softlabel_9b/step-00500/  <- sft-soft-9b-step00500
