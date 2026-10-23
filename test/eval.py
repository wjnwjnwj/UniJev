"""Evaluate a UniJev decision checkpoint on a trimodal JSONL validation file.

Three subcommands, driven by test/run_eval.sh:

  config   resolve model/checkpoint/data, write <run_dir>/eval_config.json
  predict  score one shard of the records on one GPU -> <run_dir>/shards/
  merge    join shards -> predictions.jsonl, errors.jsonl, metrics.json, summary.txt

Probabilities come from the same path as training (compiler -> DecisionModel ->
softmax over candidate logits), so every candidate's confidence is recorded.
"""
import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from train.common.schema import read_jsonl  # noqa: E402
from train.common.data import modality as record_modality  # noqa: E402
from train.cora.monitor import metrics as distribution_metrics, confidence_metrics  # noqa: E402

DEFAULT_MODELS = {"2b": "models/Qwen3.5-2B", "9b": "models/Qwen3.5-9B"}
ECE_BINS = 15


def sha256(path, chunk=1 << 20):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, payload):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


# ----------------------------------------------------------------------------
# resolution
# ----------------------------------------------------------------------------
def resolve_model(model, project_root):
    value = DEFAULT_MODELS.get(model.lower(), model)
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = project_root / path
    path = path.resolve()
    if not (path / "config.json").is_file():
        raise FileNotFoundError(f"Model directory without config.json: {path}")
    return path


def resolve_checkpoint(ckpt, ckpt_root):
    path = Path(ckpt).expanduser()
    if not path.is_absolute():
        path = Path(ckpt_root) / path if (Path(ckpt_root) / path).exists() else path
    path = path.resolve()
    if not (path / "checkpoint.pt").is_file():
        # A training run directory: prefer the selected checkpoint, then the newest.
        for pointer in ("best", "latest"):
            if (path / pointer / "checkpoint.pt").is_file():
                return (path / pointer).resolve()
        raise FileNotFoundError(f"No checkpoint.pt (nor best/ or latest/) under {path}")
    return path


def checkpoint_name(path, ckpt_root):
    path, root = Path(path), Path(ckpt_root).resolve()
    try:
        parts = path.relative_to(root).parts
    except ValueError:
        parts = path.parts[-2:] if path.name.startswith("step-") else path.parts[-1:]
    return "-".join(parts)


# ----------------------------------------------------------------------------
# config
# ----------------------------------------------------------------------------
def command_config(args):
    project_root = Path(args.project_root).resolve()
    model_path = resolve_model(args.model, project_root)
    ckpt_path = resolve_checkpoint(args.ckpt, args.ckpt_root)
    ckpt_config = json.loads((ckpt_path / "config.json").read_text())
    val_path = Path(args.val).expanduser().resolve()
    records = read_jsonl(val_path)
    media_kwargs = dict(ckpt_config.get("media_kwargs") or
                        {"max_frames": 128, "min_pixels": 3136, "max_pixels": 65536})
    if args.max_frames:
        media_kwargs["max_frames"] = args.max_frames
    if args.max_pixels:
        media_kwargs["max_pixels"] = args.max_pixels
    run_name = f"{model_path.name}_{checkpoint_name(ckpt_path, args.ckpt_root)}_{args.timestamp}"
    run_dir = Path(args.output_dir).expanduser().resolve() / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "shards").mkdir()
    (run_dir / "logs").mkdir()
    import torch
    import transformers
    import peft
    config = {
        "run_name": run_name,
        "run_dir": str(run_dir),
        "timestamp": args.timestamp,
        "model": {"name": model_path.name, "path": str(model_path),
                  "argument": args.model},
        "checkpoint": {"name": checkpoint_name(ckpt_path, args.ckpt_root), "path": str(ckpt_path),
                       "argument": args.ckpt, "checkpoint_sha256": sha256(ckpt_path / "checkpoint.pt"),
                       "training_config": ckpt_config},
        "data": {"val": str(val_path), "sha256": sha256(val_path), "records": len(records),
                 "questions": sum(len(r["request"]["questions"]) for r in records),
                 "records_by_modality": dict(Counter(record_modality(r) for r in records)),
                 "media_root": str(Path(args.media_root).resolve()), "limit": args.limit},
        "inference": {
            "dtype": args.dtype, "attn_implementation": args.attn,
            "projection_dim": ckpt_config.get("projection_dim", 256),
            "lora_rank": ckpt_config.get("lora_rank", 32),
            "lora_alpha": ckpt_config.get("lora_alpha", 64),
            "max_length": ckpt_config.get("max_length", 32768),
            "media_kwargs": media_kwargs,
            "candidate_order": "as written in the file (no shuffling)",
            "probabilities": "softmax over decision-head logits per question"
                             " (divided by exp(log_temperature) if the checkpoint stores one)",
            "seed": args.seed,
        },
        "metrics": {"ece_bins": ECE_BINS, "confidence": "max candidate probability",
                    "accuracy": "argmax candidate equals argmax target"},
        "launch": {"gpus": args.gpus.split(","), "procs_per_gpu": args.procs_per_gpu,
                   "num_shards": len(args.gpus.split(",")) * args.procs_per_gpu,
                   "cpu_workers_per_proc": args.cpu_workers, "threads_per_proc": args.threads,
                   "command": args.command_line},
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "transformers": transformers.__version__, "peft": peft.__version__,
                        "host": platform.node()},
    }
    write_json(run_dir / "eval_config.json", config)
    print(run_dir)


# ----------------------------------------------------------------------------
# predict
# ----------------------------------------------------------------------------
def load_weights(model, ckpt_dir):
    import torch
    ckpt_dir = Path(ckpt_dir)
    manifest = ckpt_dir / "manifest.json"
    if manifest.is_file():
        files = json.loads(manifest.read_text()).get("files", {})
        if "checkpoint.pt" in files and sha256(ckpt_dir / "checkpoint.pt") != files["checkpoint.pt"]:
            raise ValueError("checkpoint.pt does not match its manifest hash")
    payload = torch.load(ckpt_dir / "checkpoint.pt", map_location="cpu", weights_only=True)
    state = payload["trainable"]
    parameters = dict(model.named_parameters())
    expected = set(model.delta_parameter_names)
    if set(state) != expected:
        missing, extra = sorted(expected - set(state)), sorted(set(state) - expected)
        raise ValueError(f"Checkpoint/model parameter mismatch (wrong MODEL size?): "
                         f"missing={missing[:3]}... ({len(missing)}) extra={extra[:3]}... ({len(extra)})")
    with torch.no_grad():
        for name, tensor in state.items():
            if tensor.shape != parameters[name].shape:
                raise ValueError(f"Shape mismatch for {name}: {tuple(tensor.shape)} vs "
                                 f"{tuple(parameters[name].shape)} (wrong MODEL size?)")
            if not torch.isfinite(tensor).all():
                raise ValueError(f"Non-finite checkpoint tensor: {name}")
            parameters[name].copy_(tensor.to(parameters[name]))
    log_temperature = float(payload.get("log_temperature", 0.0) or 0.0)
    return {"format": payload.get("format"), "progress": payload.get("progress"),
            "log_temperature": log_temperature}


RECORD_COST = {"video": 0, "mixed": 0, "image": 1, "text": 2}


def shard_indices(records, shard, num_shards):
    """Expensive records first, dealt round-robin: every shard gets the same modality mix."""
    order = sorted(range(len(records)), key=lambda i: (RECORD_COST[record_modality(records[i])], i))
    return sorted(order[shard::num_shards])


class CompiledRecords:
    """Map-style dataset: DataLoader workers decode media and tokenize off the GPU loop."""

    def __init__(self, records, indices, model_config, media_root):
        self.records, self.indices = records, indices
        self.model_config, self.media_root = model_config, media_root
        self.compiler = None

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        if self.compiler is None:
            from train.common.model import build_compiler
            self.compiler = build_compiler(self.model_config, self.media_root)
        index = self.indices[position]
        try:
            return index, self.compiler.compile(self.records[index]), None
        except Exception as error:
            return index, None, (f"{type(error).__name__}: {error}", traceback.format_exc(limit=3))


def limit_threads(threads):
    import torch
    torch.set_num_threads(threads)
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = str(threads)


def command_predict(args):
    import torch
    from train.common.model import build_model
    limit_threads(args.threads)
    config = json.loads((Path(args.run_dir) / "eval_config.json").read_text())
    inference = config["inference"]
    torch.manual_seed(inference["seed"])
    model_config = {
        "model_path": config["model"]["path"], "dtype": inference["dtype"],
        "attn_implementation": inference["attn_implementation"],
        "lora_rank": inference["lora_rank"], "lora_alpha": inference["lora_alpha"],
        "projection_dim": inference["projection_dim"], "gradient_checkpointing": False,
        "deterministic": True, "device": "cuda", "max_length": inference["max_length"],
        "media_kwargs": inference["media_kwargs"],
    }
    started = time.time()
    model = build_model(model_config)
    info = load_weights(model, config["checkpoint"]["path"])
    model.eval()
    temperature = math.exp(info["log_temperature"])

    records = read_jsonl(config["data"]["val"])
    if config["data"]["limit"]:
        records = records[: config["data"]["limit"]]
    indices = shard_indices(records, args.shard, args.num_shards)
    from torch.utils.data import DataLoader
    loader = DataLoader(
        CompiledRecords(records, indices, model_config, config["data"]["media_root"]),
        batch_size=None, shuffle=False, num_workers=args.cpu_workers,
        prefetch_factor=args.prefetch if args.cpu_workers else None,
        persistent_workers=False, collate_fn=lambda item: item,
        worker_init_fn=lambda _: limit_threads(1))
    out_dir = Path(args.run_dir) / "shards"
    pred_path = out_dir / f"predictions.{args.shard:02d}.jsonl"
    err_path = out_dir / f"errors.{args.shard:02d}.jsonl"
    print(f"[shard {args.shard}/{args.num_shards}] {len(indices)} records, model load "
          f"{time.time() - started:.0f}s, checkpoint format={info['format']} "
          f"log_temperature={info['log_temperature']}", flush=True)

    done = 0
    with open(pred_path, "w") as pred_file, open(err_path, "w") as err_file, torch.inference_mode():
        for position, (index, compiled, failure) in enumerate(loader, 1):
            record = records[index]
            meta = record.get("meta", {})
            base = {"record_index": index, "record_id": meta.get("record_id"),
                    "group_id": record["group_id"], "modality": record_modality(record),
                    "domain": meta.get("domain") or meta.get("source")}
            if failure is not None:
                err_file.write(json.dumps(dict(base, error=failure[0], traceback=failure[1]),
                                          ensure_ascii=False) + "\n")
                err_file.flush()
                continue
            try:
                for question in compiled.questions:
                    logits = model(question).float().cpu()
                    probs = (logits / temperature).softmax(-1).tolist()
                    probabilities = dict(zip(question.keys, probs))
                    pred = question.keys[max(range(len(probs)), key=probs.__getitem__)]
                    row = dict(base, qid=question.qid, task=question.kind,
                               candidates=dict(zip(question.keys, question.descriptions)),
                               logits=dict(zip(question.keys, logits.tolist())),
                               probabilities=probabilities, prediction=pred,
                               confidence=probabilities[pred])
                    if question.target is not None:
                        target = dict(zip(question.keys, question.target))
                        gold = max(target, key=target.get)
                        row.update(target=target, gold=gold, correct=int(pred == gold),
                                   nll=-math.log(max(probabilities[gold], 1e-12)),
                                   brier=sum((probabilities[k] - target[k]) ** 2 for k in target))
                    pred_file.write(json.dumps(row, ensure_ascii=False) + "\n")
                    done += 1
                pred_file.flush()
            except Exception as error:  # keep going; failures are listed in errors.jsonl
                err_file.write(json.dumps(dict(base, error=f"{type(error).__name__}: {error}",
                                               traceback=traceback.format_exc(limit=3)),
                                          ensure_ascii=False) + "\n")
                err_file.flush()
                if isinstance(error, torch.cuda.OutOfMemoryError):
                    torch.cuda.empty_cache()
            if position % args.log_every == 0 or position == len(indices):
                elapsed = time.time() - started
                print(f"[shard {args.shard}] {position}/{len(indices)} records, {done} questions, "
                      f"{elapsed:.0f}s", flush=True)
    write_json(out_dir / f"done.{args.shard:02d}.json",
               {"shard": args.shard, "records": len(indices), "questions": done,
                "seconds": time.time() - started, "checkpoint": info})


# ----------------------------------------------------------------------------
# merge
# ----------------------------------------------------------------------------
def summarize(rows):
    clean = [dict(r, metrics={"accuracy": r["correct"]}) for r in rows]
    full = distribution_metrics(clean, bins=ECE_BINS)
    binary = confidence_metrics(clean, "confidence", bins=ECE_BINS)
    return {"questions": full["questions"], "accuracy": full["acc"],
            "mean_confidence": full["conf"], "ece_15": binary["ece_15"],
            "nll": full["nll"], "brier": full["brier"],
            "correctness_brier": binary["correctness_brier"],
            "aurc": binary["aurc_tie_averaged"], "reliability": binary["reliability"]}


def grouped(rows, key):
    groups = defaultdict(list)
    for row in rows:
        groups[key(row)].append(row)
    return {str(k): summarize(v) for k, v in sorted(groups.items(), key=lambda kv: str(kv[0]))}


def command_merge(args):
    run_dir = Path(args.run_dir)
    config = json.loads((run_dir / "eval_config.json").read_text())
    shards = config["launch"]["num_shards"]
    missing = [i for i in range(shards) if not (run_dir / "shards" / f"done.{i:02d}.json").is_file()]
    rows, errors = [], []
    for i in range(shards):
        for name, sink in (("predictions", rows), ("errors", errors)):
            path = run_dir / "shards" / f"{name}.{i:02d}.jsonl"
            if path.is_file():
                sink.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    rows.sort(key=lambda r: (r["record_index"], str(r["qid"])))
    errors.sort(key=lambda r: r["record_index"])
    with open(run_dir / "predictions.jsonl", "w") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(run_dir / "errors.jsonl", "w") as stream:
        for row in errors:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    labeled = [r for r in rows if "correct" in r]
    if not labeled:
        raise SystemExit("No labeled predictions to score")
    expected = config["data"]["records"] if not config["data"]["limit"] else min(
        config["data"]["records"], config["data"]["limit"])
    seconds = [json.loads((run_dir / "shards" / f"done.{i:02d}.json").read_text())["seconds"]
               for i in range(shards) if i not in missing]
    result = {
        "run_name": config["run_name"],
        "model": config["model"]["name"], "checkpoint": config["checkpoint"]["name"],
        "val": config["data"]["val"],
        "coverage": {"records_expected": expected,
                     "records_scored": len({r["record_index"] for r in rows}),
                     "records_failed": len(errors), "questions_scored": len(labeled),
                     "unfinished_shards": missing},
        "wall_seconds_max_shard": max(seconds) if seconds else None,
        "overall": summarize(labeled),
        "by_modality": grouped(labeled, lambda r: r["modality"]),
        "by_task": grouped(labeled, lambda r: r["task"]),
        "by_modality_task": grouped(labeled, lambda r: f'{r["modality"]}/{r["task"]}'),
        "by_domain": grouped(labeled, lambda r: f'{r["modality"]}/{r["domain"]}'),
    }
    write_json(run_dir / "metrics.json", result)

    def line(name, m):
        return (f"{name:<44} {m['questions']:>6} {m['accuracy'] * 100:>7.2f} "
                f"{m['mean_confidence'] * 100:>7.2f} {m['ece_15']:>7.4f} {m['nll']:>7.4f} "
                f"{m['brier']:>7.4f} {m['aurc']:>7.4f}")
    header = f"{'':<44} {'n':>6} {'acc%':>7} {'conf%':>7} {'ECE15':>7} {'NLL':>7} {'Brier':>7} {'AURC':>7}"
    text = [f"run: {config['run_name']}", f"model: {config['model']['path']}",
            f"ckpt:  {config['checkpoint']['path']}", f"val:   {config['data']['val']}",
            f"coverage: {result['coverage']}", "", header, line("overall", result["overall"])]
    for title in ("by_modality", "by_task", "by_modality_task", "by_domain"):
        text += ["", f"[{title}]"] + [line(k, v) for k, v in result[title].items()]
    (run_dir / "summary.txt").write_text("\n".join(text) + "\n")
    print("\n".join(text[:8] + [""] + ["[by_modality]"] +
                    [line(k, v) for k, v in result["by_modality"].items()]))
    if missing or errors:
        print(f"WARNING: unfinished shards={missing}, failed records={len(errors)} "
              f"(see errors.jsonl, logs/)", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    c = sub.add_parser("config")
    c.add_argument("--model", required=True)
    c.add_argument("--ckpt", required=True)
    c.add_argument("--ckpt-root", required=True)
    c.add_argument("--project-root", required=True)
    c.add_argument("--val", required=True)
    c.add_argument("--media-root", required=True)
    c.add_argument("--output-dir", required=True)
    c.add_argument("--timestamp", required=True)
    c.add_argument("--gpus", required=True)
    c.add_argument("--procs-per-gpu", type=int, default=1)
    c.add_argument("--cpu-workers", type=int, default=6)
    c.add_argument("--threads", type=int, default=4)
    c.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    c.add_argument("--attn", default="sdpa")
    c.add_argument("--max-frames", type=int, default=0)
    c.add_argument("--max-pixels", type=int, default=0)
    c.add_argument("--limit", type=int, default=0)
    c.add_argument("--seed", type=int, default=42)
    c.add_argument("--command-line", default="")
    p = sub.add_parser("predict")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--shard", type=int, required=True)
    p.add_argument("--num-shards", type=int, required=True)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--cpu-workers", type=int, default=6, help="DataLoader workers for media decoding")
    p.add_argument("--prefetch", type=int, default=4, help="compiled records queued per worker")
    p.add_argument("--threads", type=int, default=4, help="torch CPU threads in the GPU process")
    m = sub.add_parser("merge")
    m.add_argument("--run-dir", required=True)
    args = ap.parse_args()
    {"config": command_config, "predict": command_predict, "merge": command_merge}[args.command](args)


if __name__ == "__main__":
    main()
