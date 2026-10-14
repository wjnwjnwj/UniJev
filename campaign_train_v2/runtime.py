"""Distributed reduction, strict SFT loading, atomic checkpoints and provenance."""
import json
import os
import platform
import subprocess
from pathlib import Path
import torch
from . import VERSION
from .configuration import ROOT, sha256

class Context:
    def __init__(self, device):
        self.rank = int(os.environ.get('RANK', 0))
        self.world_size = int(os.environ.get('WORLD_SIZE', 1))
        self.device = device
        if 'RANK' in os.environ:
            import torch.distributed as dist
            local_rank = int(os.environ.get('LOCAL_RANK', 0))
            if device.startswith('cuda'):
                torch.cuda.set_device(local_rank)
                self.device = f'cuda:{local_rank}'
            dist.init_process_group('nccl' if device.startswith('cuda') else 'gloo')
        self.primary = self.rank == 0

    @property
    def distributed(self):
        return self.world_size > 1

    def sum(self, value):
        if not self.distributed:
            return value
        import torch.distributed as dist
        tensor = torch.tensor(value, dtype=torch.float64, device=self.device)
        dist.all_reduce(tensor)
        return tensor.item()

    def gather(self, value):
        if not self.distributed:
            return [value]
        import torch.distributed as dist
        values = [None] * self.world_size
        dist.all_gather_object(values, value)
        return values

    def on_primary(self, function):
        error = None
        if self.primary:
            try:
                function()
            except Exception as exc:
                error = f'{type(exc).__name__}: {exc}'
        if self.distributed:
            import torch.distributed as dist
            payload = [error]
            dist.broadcast_object_list(payload, src=0)
            error = payload[0]
        if error is not None:
            raise RuntimeError(error)

    def broadcast(self, parameters):
        if self.distributed:
            import torch.distributed as dist
            for parameter in parameters:
                dist.broadcast(parameter.data, src=0)

    def mean_gradients(self, parameters, global_questions):
        if global_questions <= 0:
            raise ValueError('No labeled questions in update')
        for parameter in parameters:
            if parameter.grad is None:
                parameter.grad = torch.zeros_like(parameter)
            if self.distributed:
                import torch.distributed as dist
                dist.all_reduce(parameter.grad)
            parameter.grad.div_(global_questions)

    def close(self):
        import torch.distributed as dist
        if dist.is_initialized():
            dist.destroy_process_group()


def write_json(path, payload):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def _load_base(model, state):
    expected = set(model.sft_parameter_names)
    actual = set(state)
    if actual != expected:
        raise ValueError(f"Checkpoint parameters mismatch: missing={sorted(expected-actual)[:5]}, extra={sorted(actual-expected)[:5]}")
    parameters = dict(model.base.named_parameters())
    # Validate everything before mutating a single parameter.
    for name, tensor in state.items():
        if not isinstance(tensor, torch.Tensor) or tensor.shape != parameters[name].shape or not torch.isfinite(tensor).all():
            raise ValueError(f"Invalid SFT checkpoint tensor: {name}")
    with torch.no_grad():
        for name, tensor in state.items():
            parameters[name].copy_(tensor.to(parameters[name]))


def load_initial_checkpoint(model, directory):
    payload = torch.load(Path(directory) / "checkpoint.pt", map_location="cpu", weights_only=True)
    if payload.get("format") == VERSION:
        config = json.loads((Path(directory)/'config.json').read_text())
        if config['objective'] != 'sft_soft':
            raise ValueError('Campaign continuation requires a soft SFT checkpoint')
        return load_checkpoint(model,directory,initial_sft=True)
    _load_base(model, payload["trainable"])
    return payload["progress"]


def save_checkpoint(directory, model, config, progress, optimizer=None, rank_states=None):
    directory = Path(directory)
    writing = directory.with_name(directory.name + ".writing")
    latest = directory.parent / "latest"
    if latest.exists() and not latest.is_symlink():
        raise FileExistsError(f"Refusing to replace non-symlink {latest}")
    if directory.exists():
        raise FileExistsError(directory)
    writing.mkdir(parents=True, exist_ok=False)
    parameters = dict(model.base.named_parameters())
    # Preserve every original SFT delta, including frozen adapters and merger.
    state = {name: parameters[name].detach().cpu().clone() for name in model.sft_parameter_names}
    torch.save({"format": VERSION, "actor_mode": model.actor_mode, "trainable": state,
                "log_temperature": model.log_temperature.detach().cpu().clone(),
                "progress": progress}, writing / "checkpoint.pt")
    write_json(writing / "config.json", config)
    if optimizer is not None:
        torch.save({"optimizer": optimizer.state_dict(), "rank_states": rank_states,
                    "progress": progress, "resume_supported": True}, writing / "training_state.pt")
    manifest = {"format": VERSION, "progress": progress, "files": {
        p.name: sha256(p) for p in sorted(writing.iterdir()) if p.is_file()}}
    write_json(writing / "manifest.json", manifest)
    writing.rename(directory)
    temp_link = directory.parent / ".latest.tmp"
    temp_link.symlink_to(directory.name)
    os.replace(temp_link, latest)


def load_checkpoint(model, directory, initial_sft=False):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("format") != VERSION:
        raise ValueError("Unsupported checkpoint format")
    for name in manifest["files"]:
        if manifest["files"].get(name) != sha256(directory / name):
            raise ValueError(f"Checkpoint hash mismatch: {name}")
    payload = torch.load(directory / "checkpoint.pt", map_location="cpu", weights_only=True)
    if initial_sft and (json.loads((directory/'config.json').read_text())['objective']!='sft_soft'
                        or payload.get('actor_mode')!='sft_adapters'):
        raise ValueError('Only soft SFT may initialize a different actor mode')
    if payload.get("format") != VERSION or (not initial_sft and payload.get("actor_mode") != model.actor_mode):
        raise ValueError("Checkpoint actor/version mismatch")
    if payload["progress"] != manifest["progress"]:
        raise ValueError("Checkpoint progress mismatch")
    temperature = payload["log_temperature"]
    if not isinstance(temperature, torch.Tensor) or temperature.shape != model.log_temperature.shape or not torch.isfinite(temperature).all():
        raise ValueError("Invalid checkpoint temperature")
    value = float(temperature.exp())
    if not model.temperature_min * (1 - 1e-6) <= value <= model.temperature_max * (1 + 1e-6):
        raise ValueError("Checkpoint temperature outside configured bounds")
    if model.actor_mode != "temperature" and float(temperature) != 0:
        raise ValueError("Non-temperature actor must retain T=1")
    _load_base(model, payload["trainable"])
    with torch.no_grad():
        model.log_temperature.copy_(temperature.to(model.log_temperature))
    return payload["progress"]


def provenance():
    package = Path(__file__).resolve().parent
    packages = [package] + [ROOT/n for n in ('campaign_train_v1',)]
    result = {"source_sha256": {str(p.relative_to(ROOT)): sha256(p) for d in packages for p in sorted(d.glob("*.py"))},
              "python": platform.python_version(), "torch": str(torch.__version__), "implementation": "campaign-train-v2"}
    try:
        result["git_commit"] = subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
        result["git_status"] = subprocess.check_output(
            ["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"], text=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        result["git_commit"] = None
    return result


def snapshot_sources(output):
    """Physical source copies, independent of future working-tree changes."""
    import shutil
    for name in ('campaign_train_v2','campaign_train_v1'):
        destination = Path(output)/'source'/name
        destination.mkdir(parents=True,exist_ok=False)
        for path in (ROOT/name).glob('*.py'):
            shutil.copy2(path,destination/path.name)
