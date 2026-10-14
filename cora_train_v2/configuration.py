"""Pinned trimodal training with an internal source-group validation split and independent testing.

v2 configuration: grouping is fixed to modality_confidence (each calibration
group holds questions of one modality; bins are fit within that group).
"""
import json
import math
from pathlib import Path
from campaign_train_v2.configuration import ROOT, sha256, load_config as load_base, check_paths
from campaign_train_v1.schema import read_jsonl
from confidence_group_v1.configuration import check_release
from .data import build_partitions
from . import VERSION


def load_config(path):
    path = Path(path)
    path = path if path.is_absolute() else ROOT / path
    c = json.loads(path.read_text())
    if c["version"] != VERSION or c["method"] != "cora_supervised_bins":
        raise ValueError("Not a supervised CORA configuration")
    if "validation_fraction" in c or "source_config" in c or "dev_data" in c:
        raise ValueError("Obsolete development-set/histogram role; validation is held out inside the training set")
    for k in ("base_config", "data", "final_eval_data", "release_manifest", "output"):
        p = Path(c[k])
        c[k] = str((p if p.is_absolute() else ROOT / p).resolve())
    for k in ("bins", "min_bin_count", "max_steps", "eval_every", "group_questions", "world_size", "memory_mib"):
        if isinstance(c[k], bool) or not isinstance(c[k], int) or c[k] < 1:
            raise ValueError("Invalid positive integer: " + k)
    if c["actor_mode"] not in {"head", "sft_adapters"} or c["grouping"] != "modality_confidence":
        raise ValueError("Invalid trainable scope or grouping; v2 bins by confidence within one modality")
    if c["target_statistic"] != "accuracy":
        raise ValueError("Unknown bin statistic; v2 targets within-modality bin accuracy")
    if not all(math.isfinite(c[k]) and c[k] > 0 for k in ("head_lr", "lora_lr", "merger_lr", "grad_clip")):
        raise ValueError("Invalid optimizer setting")
    fraction = c["internal_validation_fraction"]
    if isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or not 0 < fraction < 0.5:
        raise ValueError("Internal validation fraction must stay a small held-out share")
    if c["group_questions"] < c["world_size"]:
        raise ValueError("Group smaller than world size")
    if c["data"] == c["final_eval_data"]:
        raise ValueError("Training and test paths must differ")
    base = load_base(c["base_config"])
    for k in ("data", "final_eval_data"):
        if c[k] != base[k]:
            raise ValueError("CORA must use the declared SFT dataset roles: " + k)
    if c["output"] == base["output"] or c["output"] == base["init_checkpoint"]:
        raise ValueError("Cannot overwrite the SFT/legacy run")
    source = {k: c[k] for k in ("base_config", "data", "final_eval_data",
                               "release_manifest", "input_sha256", "checkpoint_sha256",
                               "expected_eval_counts")}
    return c, source


def prepare_inputs(c, source, thorough=True):
    check_release(source)
    required = {"base_config", "data", "final_eval_data"}
    if set(source["input_sha256"]) != required:
        raise ValueError("Missing pinned training/test input")
    for key, digest in source["input_sha256"].items():
        if sha256(source[key]) != digest:
            raise ValueError("Pinned input changed: " + key)
    base = load_base(source["base_config"])
    check_paths(base)
    if thorough and sha256(Path(base["init_checkpoint"]) / "checkpoint.pt") != source["checkpoint_sha256"]:
        raise ValueError("Initial SFT checkpoint changed")
    training = read_jsonl(source["data"])
    testing = read_jsonl(source["final_eval_data"])
    split = build_partitions(training, testing, c["internal_validation_fraction"])
    if split["audit"]["question_counts"]["test"] != source["expected_eval_counts"]:
        raise ValueError("Unexpected final test coverage")
    if set(split["audit"]["question_counts"]["validation"]) != set(source["expected_eval_counts"]):
        raise ValueError("Internal validation must cover every tested modality")
    if not split["audit"]["validation_questions"] < split["audit"]["train_questions"]:
        raise ValueError("Validation must stay the small side of the split")
    model_config = dict(base, actor_mode=c["actor_mode"],
                        gradient_checkpointing=c["actor_mode"] == "sft_adapters",
                        objective="cora_supervised_bins", output=c["output"],
                        head_lr=c["head_lr"], lora_lr=c["lora_lr"], merger_lr=c["merger_lr"],
                        world_size=c["world_size"], max_steps=c["max_steps"], memory_mib=c["memory_mib"])
    return {"train": training, "validation": training}, split, model_config
