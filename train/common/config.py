"""Portable recipe loading and explicit command-line overrides."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parser(description):
    result = argparse.ArgumentParser(description=description)
    result.add_argument("--config", required=True)
    for name in ("model-path", "data", "media-root", "output", "init-checkpoint"):
        result.add_argument("--" + name)
    result.add_argument("--check-data", action="store_true", help="Validate labels and assets without loading weights")
    result.add_argument("--smoke", action="store_true", help="One update on the bundled examples; not the full recipe")
    result.add_argument("--resume", action="store_true")
    result.add_argument("--strict-determinism", action="store_true",
                        help="Use deterministic PyTorch kernels for numerical comparisons; changes the original kernel execution mode")
    return result


def load_config(args, stage):
    path = Path(args.config).resolve()
    config = json.loads(path.read_text())
    if args.strict_determinism:
        config["strict_determinism"] = True
    if type(config.get("strict_determinism", False)) is not bool:
        raise ValueError("strict_determinism must be a boolean")
    if config["stage"] != stage:
        raise ValueError(f"Expected a {stage} recipe")
    if config["model_size"] not in {"2b", "9b"}:
        raise ValueError("model_size must be 2b or 9b")
    for key in ("model_path", "data", "media_root", "output", "init_checkpoint"):
        override = getattr(args, key, None)
        if override is not None:
            config[key] = str(Path(override).expanduser().resolve())
        elif config.get(key):
            value = Path(config[key]).expanduser()
            config[key] = str((value if value.is_absolute() else REPO / value).resolve())
    if args.smoke:
        config.update(smoke=True, world_size=1, max_steps=1, save_every=1)
        if args.data is None:
            config["data"] = str(REPO / "data/examples.jsonl")
        if args.media_root is None:
            config["media_root"] = str(REPO / "data")
        if args.output is None:
            config["output"] = str(REPO / "outputs" / f"smoke_{stage}_{config['model_size']}")
        if stage == "cora":
            config.update(group_questions=24, min_bin_count=1, eval_every=1)
            if args.init_checkpoint is None:
                config["init_checkpoint"] = str(REPO / "outputs" / f"smoke_sft_{config['model_size']}" / "step-00001")
        elif config["model_size"] == "2b":
            config["tokens_per_step"] = 1024
        else:
            config["global_tokens_per_step"] = 1024
    for key in ("world_size", "max_steps", "save_every", "max_length", "projection_dim", "lora_rank", "lora_alpha"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError("Invalid positive integer: " + key)
    for key in ("head_lr", "lora_lr", "merger_lr", "grad_clip"):
        if not math.isfinite(config[key]) or config[key] <= 0:
            raise ValueError("Invalid optimizer setting: " + key)
    if config["weight_decay"] < 0 or not math.isfinite(config["weight_decay"]):
        raise ValueError("Invalid weight decay")
    if stage == "cora":
        for key in ("bins", "min_bin_count", "group_questions", "eval_every"):
            if type(config[key]) is not int or config[key] < 1:
                raise ValueError("Invalid positive integer: " + key)
        if config["group_questions"] < config["world_size"]:
            raise ValueError("Group smaller than world size")
        if not 0 < config["internal_validation_fraction"] < 0.5:
            raise ValueError("Invalid internal holdout fraction")
        if not config.get("init_checkpoint"):
            raise ValueError("CORA requires an SFT checkpoint")
        if Path(config["output"]) == Path(config["init_checkpoint"]):
            raise ValueError("CORA output must differ from its SFT input")
    else:
        if config.get("init_checkpoint"):
            raise ValueError("SFT starts from the base model; use --resume to continue an SFT run")
        budget = "tokens_per_step" if config["model_size"] == "2b" else "global_tokens_per_step"
        if type(config[budget]) is not int or config[budget] < 1:
            raise ValueError("Invalid token budget")
    return config


def configure_determinism(config):
    """Configure before Context initializes CUDA; defaults preserve the source recipe."""
    if not config.get("strict_determinism", False):
        return
    workspace = os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if workspace not in {":4096:8", ":16:8"}:
        raise ValueError("Strict determinism requires CUBLAS_WORKSPACE_CONFIG=:4096:8 or :16:8")
    import torch
    torch.use_deterministic_algorithms(True)
