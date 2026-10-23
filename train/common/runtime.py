"""Launch-time machinery shared by the SFT and CORA entry points.

Media decoding and tokenization run in DataLoader worker processes, so the GPU
loop only moves ready tensors; gradients are all-reduced as one flat buffer per
dtype; the run directory holds the resolved config, step logs and checkpoints.
"""
import argparse
import json
import math
import os
import platform
import random
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import torch

from .config import REPO, sha256
from .data import modality as record_modality
from .schema import candidates, target_distribution

COST = {"video": 0, "mixed": 0, "image": 1, "text": 2}
MODALITIES = ("text", "image", "video", "mixed")

# Stage-specific defaults; everything else is shared. Nothing depends on the model's size.
STAGE_DEFAULTS = {
    "sft": {"max_steps": 0, "save_every": 100, "head_lr": 2e-4, "lora_lr": 5e-5, "merger_lr": 1e-5,
            "weight_decay": 0.01},
    "cora": {"max_steps": 100, "save_every": 10, "head_lr": 2e-5, "lora_lr": 5e-6, "merger_lr": 1e-6,
             "weight_decay": 0.0},
}


# ----------------------------------------------------------------------------
# configuration
# ----------------------------------------------------------------------------
def add_common_arguments(ap, stage):
    d = STAGE_DEFAULTS[stage]
    g = ap.add_argument_group("model / checkpoint")
    g.add_argument("--model", required=True,
                   help="base model directory (any Qwen3.5-style multimodal checkpoint with config.json)")
    g.add_argument("--init-ckpt", help="trainable (LoRA + merger + head) checkpoint to start from"
                   + ("; required: the SFT checkpoint" if stage == "cora" else "; optional"))
    g = ap.add_argument_group("data")
    g.add_argument("--data", required=True, help="training JSONL")
    g.add_argument("--media-root", help="base for relative media paths (default: data file's dir)")
    g.add_argument("--max-frames", type=int, default=128, help="frames per video")
    g.add_argument("--min-pixels", type=int, default=3136, help="per image / frame")
    g.add_argument("--max-pixels", type=int, default=65536, help="per image / frame")
    g.add_argument("--max-length", type=int, default=32768, help="tokens per branch; longer records are skipped")
    g.add_argument("--verify-media-hash", action="store_true",
                   help="sha256 every pinned asset once at startup (slow on large sets)")
    g = ap.add_argument_group("optimization")
    g.add_argument("--max-steps", type=int, default=d["max_steps"],
                   help="optimizer updates" + (" (0 = run until --epochs are done)" if stage == "sft" else ""))
    g.add_argument("--save-every", type=int, default=d["save_every"], help="checkpoint every N updates")
    g.add_argument("--head-lr", type=float, default=d["head_lr"])
    g.add_argument("--lora-lr", type=float, default=d["lora_lr"])
    g.add_argument("--merger-lr", type=float, default=d["merger_lr"])
    g.add_argument("--weight-decay", type=float, default=d["weight_decay"])
    g.add_argument("--grad-clip", type=float, default=1.0)
    g.add_argument("--lr-schedule", choices=["constant", "cosine", "linear"], default="constant")
    g.add_argument("--warmup-steps", type=int, default=0)
    g.add_argument("--min-lr-ratio", type=float, default=0.0, help="cosine/linear floor as a fraction of lr")
    g.add_argument("--lora-rank", type=int, default=32)
    g.add_argument("--lora-alpha", type=int, default=64)
    g.add_argument("--projection-dim", type=int, default=256, help="decision head width")
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--skip-nonfinite-update", type=int, choices=[0, 1], default=1,
                   help="1: skip an update whose gradient norm is inf/nan (logged); 0: stop the run")
    g = ap.add_argument_group("efficiency")
    g.add_argument("--attn", dest="attn_implementation", default="sdpa",
                   choices=["sdpa", "eager", "flash_attention_2"])
    g.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    g.add_argument("--grad-ckpt", dest="gradient_checkpointing", type=int, choices=[0, 1], default=1,
                   help="gradient checkpointing (0 is faster but needs more GPU memory)")
    g.add_argument("--cpu-offload", dest="activation_cpu_offload", type=int, choices=[0, 1], default=0,
                   help="offload saved activations to host memory (saves GPU memory, slower)")
    g.add_argument("--cpu-workers", type=int, default=6, help="DataLoader workers per rank (0 = in-process)")
    g.add_argument("--prefetch", type=int, default=4, help="compiled records queued per worker")
    g.add_argument("--threads", type=int, default=4, help="torch CPU threads per rank")
    g = ap.add_argument_group("run")
    g.add_argument("--output", required=True, help="run directory (created by the launch script)")
    g.add_argument("--resume", action="store_true", help="continue the run in --output from latest/")
    g.add_argument("--check-data", action="store_true", help="validate data and media, then exit (CPU only)")
    g.add_argument("--progress", type=int, choices=[0, 1], default=1,
                   help="progress bar on the console (rank 0); 0 = print every update as JSON")
    g.add_argument("--log-every", type=int, default=10,
                   help="with the progress bar, also print the full JSON line every N updates "
                        "(logs/train.jsonl always gets every update)")
    g.add_argument("--command-line", default="")


def resolve_model(model):
    path = Path(model).expanduser().resolve()
    if not (path / "config.json").is_file():
        raise FileNotFoundError(f"--model must be a model directory containing config.json: {path}")
    return path


def build_config(args, stage):
    """Command-line arguments (with their defaults) -> resolved, validated run config."""
    model_path = resolve_model(args.model)
    config = {k: v for k, v in vars(args).items()
              if k not in {"model", "max_frames", "min_pixels", "max_pixels", "check_data"}}
    data = Path(args.data).expanduser().resolve()
    config.update(
        stage=stage, model_path=str(model_path), model_name=model_path.name,
        media_kwargs={"max_frames": args.max_frames, "min_pixels": args.min_pixels,
                      "max_pixels": args.max_pixels},
        data=str(data),
        media_root=str(Path(args.media_root).expanduser().resolve() if args.media_root else data.parent),
        output=str(Path(args.output).expanduser().resolve()), deterministic=True)
    if config.get("init_ckpt"):
        config["init_ckpt"] = str(resolve_checkpoint(config["init_ckpt"]))
    for key in ("gradient_checkpointing", "activation_cpu_offload", "skip_nonfinite_update"):
        config[key] = bool(config[key])
    for key in ("save_every", "lora_rank", "lora_alpha", "projection_dim", "max_length",
                "prefetch", "threads", "max_frames", "min_pixels", "max_pixels"):
        value = config.get(key, config["media_kwargs"].get(key))
        if type(value) is not int or value < 1:
            raise ValueError(f"--{key.replace('_', '-')} must be a positive integer, got {value!r}")
    if config["log_every"] < 1:
        raise ValueError("--log-every must be >= 1")
    if config["cpu_workers"] < 0 or config["max_steps"] < 0 or config["warmup_steps"] < 0:
        raise ValueError("--cpu-workers, --max-steps and --warmup-steps must be >= 0")
    for key in ("head_lr", "lora_lr", "merger_lr", "grad_clip"):
        if not (math.isfinite(config[key]) and config[key] > 0):
            raise ValueError(f"Invalid --{key.replace('_', '-')}: {config[key]!r}")
    if not (math.isfinite(config["weight_decay"]) and config["weight_decay"] >= 0):
        raise ValueError("Invalid --weight-decay")
    if config["lr_schedule"] != "constant" and config["max_steps"] < 1:
        raise ValueError("--lr-schedule cosine/linear needs --max-steps > 0")
    return config


def resolve_checkpoint(path, root=REPO.parent / "cora-weights-hf"):
    """Absolute/relative path, or a name under cora-weights-hf/; a run dir resolves to latest/."""
    path = Path(path).expanduser()
    if not path.exists() and (Path(root) / path).exists():
        path = Path(root) / path
    path = path.resolve()
    if (path / "checkpoint.pt").is_file():
        return path
    for pointer in ("latest", "best"):
        if (path / pointer / "checkpoint.pt").is_file():
            return (path / pointer).resolve()
    raise FileNotFoundError(f"No checkpoint.pt (nor latest/ or best/) under {path}")


def environment():
    import transformers
    import peft
    return {"python": platform.python_version(), "torch": torch.__version__,
            "transformers": transformers.__version__, "peft": peft.__version__,
            "host": platform.node(), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "world_size": int(os.environ.get("WORLD_SIZE", 1))}


# ----------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------
def media_paths(record, root):
    state = record["request"]["state"]
    paths = []
    for message in state.get("messages", []) if isinstance(state, dict) else []:
        content = message["content"]
        for item in content if isinstance(content, list) else []:
            if item["type"] in {"image_url", "video_url"}:
                url = item[item["type"]]["url"]
                if "://" in url:
                    raise ValueError("Materialize remote media locally: " + url)
                path = Path(url)
                paths.append(path if path.is_absolute() else Path(root) / path)
    return paths


def audit_records(records, config, threads=32):
    """Labels, candidate counts and media existence (optionally hashes), using a thread pool."""
    from collections import Counter
    labeled = Counter()
    for record in records:
        for qid, question in record["request"]["questions"].items():
            keys = [k for k, _ in candidates(question)]
            if target_distribution(record.get("targets", {}).get(qid), keys) is not None:
                labeled[record_modality(record)] += 1
    paths = sorted({str(p) for r in records for p in media_paths(r, config["media_root"])})
    with ThreadPoolExecutor(threads) as pool:
        missing = [p for p, ok in zip(paths, pool.map(os.path.isfile, paths)) if not ok]
    if missing:
        raise FileNotFoundError(f"{len(missing)} media files missing, e.g. {missing[:3]}")
    mismatched = []
    if config.get("verify_media_hash"):
        assets = [(Path(a["path"]) if Path(a["path"]).is_absolute() else Path(config["media_root"]) / a["path"],
                   a["sha256"]) for r in records for a in r.get("assets", [])]
        with ThreadPoolExecutor(threads) as pool:
            digests = list(pool.map(lambda pair: sha256(pair[0]), assets))
        mismatched = [str(p) for (p, h), d in zip(assets, digests) if h != d]
        if mismatched:
            raise ValueError(f"{len(mismatched)} media hash mismatches, e.g. {mismatched[:3]}")
    return {"records": len(records), "labeled_questions": sum(labeled.values()),
            "records_by_modality": dict(Counter(record_modality(r) for r in records)),
            "labeled_questions_by_modality": dict(labeled), "media_files": len(paths),
            "media_hash_verified": bool(config.get("verify_media_hash")),
            "data_sha256": sha256(config["data"])}


def balanced_shard(items, records, rank, world, key=lambda item: item):
    """Most expensive records first, dealt round-robin: equal modality mix per rank."""
    ordered = sorted(items, key=lambda item: COST[record_modality(records[key(item)])])
    return ordered[rank::world]


def limit_threads(threads):
    torch.set_num_threads(threads)
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = str(threads)


class CompiledItems(torch.utils.data.Dataset):
    """item = (record_index, qid | None, candidate_seed | None) -> (item, CompiledState | None, error)."""

    def __init__(self, records, items, config):
        self.records, self.items, self.config = records, items, config
        self.compiler = None

    def __len__(self):
        return len(self.items)

    def __getitem__(self, position):
        item = self.items[position]
        if self.compiler is None:
            from .model import build_compiler
            self.compiler = build_compiler(self.config, self.config["media_root"])
            self.compiler.verify_hashes = False  # audited once at startup instead
        index, qid, seed = item
        try:
            rng = random.Random(seed) if seed is not None else None
            compiled = self.compiler.compile(self.records[index], rng=rng, labeled_only=True)
            if qid is not None:
                compiled.questions = [q for q in compiled.questions if q.qid == qid]
            return item, compiled, None
        except Exception as error:
            return item, None, f"{type(error).__name__}: {error}\n{traceback.format_exc(limit=2)}"


def make_loader(records, items, config):
    workers = config["cpu_workers"]
    return torch.utils.data.DataLoader(
        CompiledItems(records, items, config), batch_size=None, shuffle=False,
        num_workers=workers, prefetch_factor=config["prefetch"] if workers else None,
        collate_fn=_identity, worker_init_fn=_worker_init, persistent_workers=False)


def _identity(item):
    return item


def _worker_init(_):
    limit_threads(1)


# ----------------------------------------------------------------------------
# optimization
# ----------------------------------------------------------------------------
def reduce_gradients(parameters, divisor, context):
    """One all-reduce per dtype over a flat buffer, then divide."""
    by_dtype = {}
    for parameter in parameters:
        if parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter)
        by_dtype.setdefault(parameter.grad.dtype, []).append(parameter.grad)
    for grads in by_dtype.values():
        flat = torch.cat([g.reshape(-1) for g in grads])
        if context.distributed:
            import torch.distributed as dist
            dist.all_reduce(flat)
        flat.div_(divisor)
        offset = 0
        for g in grads:
            g.copy_(flat[offset:offset + g.numel()].view_as(g))
            offset += g.numel()


def all_reduce_vector(values, context):
    tensor = torch.tensor(values, dtype=torch.float64, device=context.device)
    if context.distributed:
        import torch.distributed as dist
        dist.all_reduce(tensor)
    return tensor.tolist()


def build_optimizer(model, config):
    from .model import optimizer_groups
    groups = optimizer_groups(model, config)
    optimizer = torch.optim.AdamW([{"params": g["params"], "lr": g["lr"], "name": g["name"]}
                                   for g in groups], weight_decay=config["weight_decay"])
    warmup, total = config.get("warmup_steps", 0), config["max_steps"]
    schedule, floor = config["lr_schedule"], config.get("min_lr_ratio", 0.0)

    def factor(step):  # step = number of optimizer updates already applied
        if warmup and step < warmup:
            return (step + 1) / warmup
        if schedule == "constant":
            return 1.0
        progress = min(1.0, (step - warmup) / max(1, total - warmup))
        shape = 0.5 * (1 + math.cos(math.pi * progress)) if schedule == "cosine" else 1 - progress
        return floor + (1 - floor) * shape
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
    return optimizer, scheduler, [p for g in groups for p in g["params"]]


def load_trainable(model, directory):
    """Load trainable deltas from any of this project's checkpoint formats (weights only)."""
    directory = Path(directory)
    manifest = directory / "manifest.json"
    if manifest.is_file():
        files = json.loads(manifest.read_text()).get("files", {})
        if "checkpoint.pt" in files and sha256(directory / "checkpoint.pt") != files["checkpoint.pt"]:
            raise ValueError("checkpoint.pt does not match its manifest")
    payload = torch.load(directory / "checkpoint.pt", map_location="cpu", weights_only=True)
    if float(payload.get("log_temperature", 0) or 0) != 0:
        raise ValueError("Checkpoint carries a non-unit temperature; not a trainable init")
    state, parameters = payload["trainable"], dict(model.named_parameters())
    expected = set(model.delta_parameter_names)
    if set(state) != expected:
        raise ValueError(f"Checkpoint/model parameter mismatch (wrong --model size or LoRA rank?): "
                         f"{len(expected - set(state))} missing, {len(set(state) - expected)} unexpected")
    with torch.no_grad():
        for name, tensor in state.items():
            if tensor.shape != parameters[name].shape or not torch.isfinite(tensor).all():
                raise ValueError(f"Invalid checkpoint tensor {name}: {tuple(tensor.shape)}")
            parameters[name].copy_(tensor.to(parameters[name]))
    return {"path": str(directory), "format": payload.get("format"), "progress": payload.get("progress"),
            "sha256": sha256(directory / "checkpoint.pt")}


# ----------------------------------------------------------------------------
# run directory
# ----------------------------------------------------------------------------
class Run:
    def __init__(self, config, context):
        self.output = Path(config["output"])
        self.context = context
        self.started = time.time()
        self.show_progress = bool(config.get("progress", 1))
        self.log_every = config.get("log_every", 10)
        self.bar = None

    def start_progress(self, total, initial=0, unit="step", desc=""):
        """Rank-0 tqdm bar; other ranks and --progress 0 get no bar."""
        if not (self.context.primary and self.show_progress):
            return
        import sys
        from tqdm import tqdm
        self.bar = tqdm(total=total, initial=initial, unit=unit, desc=desc, file=sys.stdout,
                        dynamic_ncols=True, mininterval=2.0, smoothing=0.05)

    def advance(self, n, postfix):
        if self.bar is not None:
            self.bar.set_postfix(postfix, refresh=False)
            self.bar.update(n)

    def close_progress(self):
        if self.bar is not None:
            self.bar.close()
            self.bar = None

    def echo(self, step):
        """Print this update's JSON line on the console?"""
        return self.bar is None or step % self.log_every == 0

    def write_json(self, name, payload):
        from .checkpoint import write_json
        if self.context.primary:
            write_json(self.output / name, payload)

    def log(self, name, entry, echo=True):
        if not self.context.primary:
            return
        line = json.dumps(entry, ensure_ascii=False, allow_nan=False)
        with (self.output / "logs" / name).open("a") as stream:
            stream.write(line + "\n")
        if echo:
            if self.bar is not None:
                self.bar.write(line)
            else:
                print(line, flush=True)

    def error(self, item, message):
        """Every rank appends its own skipped records."""
        with (self.output / "logs" / f"skipped.rank{self.context.rank:02d}.jsonl").open("a") as stream:
            stream.write(json.dumps({"item": list(item), "error": message}, ensure_ascii=False) + "\n")


def prepare_run(config, context, audit):
    output = Path(config["output"])

    def initialize():
        (output / "logs").mkdir(parents=True, exist_ok=True)
        if config["resume"]:
            saved = json.loads((output / "run_config.json").read_text())
            changed = {k for k in set(saved["config"]) | set(config)
                       if k not in {"resume", "command_line", "output"}
                       and saved["config"].get(k) != config.get(k)}
            if changed:
                raise ValueError(f"--resume with changed settings: {sorted(changed)}")
            if saved["environment"]["world_size"] != context.world_size:
                raise ValueError("--resume must use the same number of GPUs")
        else:
            if any(p.name.startswith("step-") for p in output.iterdir()):
                raise FileExistsError(f"{output} already holds checkpoints; use --resume")
            from .checkpoint import source_hashes, write_json
            write_json(output / "run_config.json", {
                "config": config, "data_audit": audit, "environment": environment(),
                "source_sha256": source_hashes(),
                "started": datetime.now().isoformat(timespec="seconds")})
    context.on_primary(initialize)
    return Run(config, context)


def save(run, step, model, config, progress, optimizer, scheduler, rank_states):
    from .checkpoint import save_checkpoint
    rank_states = [dict(s, scheduler=scheduler.state_dict()) for s in rank_states]
    run.context.on_primary(lambda: save_checkpoint(run.output / f"step-{step:05d}", model, config,
                                                   progress, optimizer, rank_states))
