"""Delta-only checkpoints, atomic writes, and reproducible continuation."""
import json
import os
from pathlib import Path

import torch

from .config import ROOT, sha256

FORMAT = "cora-training-v1"


def write_json(path, payload):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def source_hashes():
    return {str(p.relative_to(ROOT)): sha256(p) for p in sorted(ROOT.rglob("*.py"))}


def point_to(output, name, step):
    target = Path(output) / name
    if target.exists() and not target.is_symlink():
        raise FileExistsError(target)
    temporary = Path(output) / ("." + name + ".tmp")
    temporary.unlink(missing_ok=True)
    temporary.symlink_to(f"step-{step:05d}")
    os.replace(temporary, target)


def save_checkpoint(directory, model, config, progress, optimizer, rank_states=None):
    directory = Path(directory)
    if directory.exists():
        raise FileExistsError(directory)
    writing = directory.with_name(directory.name + ".writing")
    writing.mkdir(parents=True, exist_ok=False)
    parameters = dict(model.named_parameters())
    state = {name: parameters[name].detach().cpu().clone() for name in model.delta_parameter_names}
    torch.save({"format": FORMAT, "stage": config["stage"], "trainable": state,
                "progress": progress}, writing / "checkpoint.pt")
    write_json(writing / "config.json", config)
    torch.save({"optimizer": optimizer.state_dict(), "rank_states": rank_states,
                "progress": progress}, writing / "training_state.pt")
    write_json(writing / "manifest.json", {"format": FORMAT, "progress": progress,
               "files": {p.name: sha256(p) for p in sorted(writing.iterdir()) if p.is_file()}})
    writing.rename(directory)
    point_to(directory.parent, "latest", progress["step"])


def load_checkpoint(model, directory, initial_sft=False):
    directory = Path(directory)
    if (directory / "manifest.json").is_file():
        manifest = json.loads((directory / "manifest.json").read_text())
        for name, digest in manifest["files"].items():
            if Path(name).name != name or sha256(directory / name) != digest:
                raise ValueError("Checkpoint hash mismatch: " + name)
    else:
        manifest = None  # Original SFT-2B uses an unversioned delta checkpoint.
    payload = torch.load(directory / "checkpoint.pt", map_location="cpu", weights_only=True)
    config = json.loads((directory / "config.json").read_text())
    if initial_sft:
        if payload.get("format") == FORMAT:
            valid = payload.get("stage") == config.get("stage") == "sft"
        elif payload.get("format") == "campaign-train-v1":
            valid = config.get("objective") == "sft_soft" and payload.get("actor_mode") == "sft_adapters"
        else:
            valid = payload.get("format") is None and config.get("soft_label_prob") == 0.9
        if not valid:
            raise ValueError("CORA must initialize from a soft-label SFT checkpoint")
    elif payload.get("format") != FORMAT:
        raise ValueError("Resume requires this repository's optimizer and sampler state")
    if "log_temperature" in payload and float(payload["log_temperature"]) != 0:
        raise ValueError("The original SFT checkpoint must retain temperature 1")
    if manifest is not None and payload["progress"] != manifest["progress"]:
        raise ValueError("Checkpoint progress mismatch")
    parameters = dict(model.named_parameters())
    state = payload["trainable"]
    if set(state) != set(model.delta_parameter_names):
        raise ValueError("Checkpoint parameter names do not match the model")
    for name, tensor in state.items():
        if not isinstance(tensor, torch.Tensor) or tensor.shape != parameters[name].shape or not torch.isfinite(tensor).all():
            raise ValueError("Invalid checkpoint tensor: " + name)
    with torch.no_grad():
        for name, tensor in state.items():
            parameters[name].copy_(tensor.to(parameters[name]))
    return payload["progress"]


def load_training_state(directory, progress, optimizer):
    state = torch.load(Path(directory) / "training_state.pt", map_location="cpu", weights_only=False)
    if state["progress"] != progress:
        raise ValueError("Optimizer/checkpoint progress mismatch")
    optimizer.load_state_dict(state["optimizer"])
    return state["rank_states"]
