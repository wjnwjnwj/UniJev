"""SFT: soft cross-entropy on decision logits, token-budget steps, optional DDP.

Usage:
  python -m videojev.train --config configs/sft_jev_easy_2b.json
  torchrun --standalone --nproc_per_node=8 -m videojev.train --config ...

The loss for one labeled question is CE(target distribution, softmax(logits)),
optionally plus a Brier term. One-hot targets are softened at training time
(soft_label_prob, default 0.9, to the correct option; the rest split evenly).
Choice candidates are reshuffled every epoch so
the head cannot memorize positions or letters.

DDP model: each rank owns an interleaved shard of the epoch order (padded by
wrapping so every rank sees the same record count). Gradients accumulate
locally until any rank fills its per-rank token budget (agreed on with a MAX
all-reduce), then everyone sums gradients, averages, and steps in lockstep.
Only rank 0 logs and saves checkpoints.
"""
import argparse
import hashlib
import json
import math
import os
import random
import shutil
import time
from pathlib import Path

import torch

from .schema import read_jsonl
from .modeling import build_model, build_compiler, optimizer_groups, trainable_state
from .evaluate import question_metrics, summarize


def soften_target(target, correct_prob=0.9):
    """One-hot -> soft label: the correct option keeps correct_prob and the
    remaining mass is split evenly over the other candidates. Targets that are
    already soft (no entry at 1.0) and single-candidate questions pass through
    unchanged. Training-time only; eval still scores against the raw labels."""
    size = len(target)
    if size < 2 or not math.isclose(max(target), 1.0, abs_tol=1e-6):
        return target
    correct = max(range(size), key=target.__getitem__)
    rest = (1.0 - correct_prob) / (size - 1)
    return [correct_prob if i == correct else rest for i in range(size)]


def question_loss(logits, target, brier_weight=0.0):
    target = torch.tensor(target, dtype=torch.float32, device=logits.device)
    log_probs = torch.log_softmax(logits.float(), -1)
    loss = -(target * log_probs).sum()
    if brier_weight:
        loss = loss + brier_weight * ((log_probs.exp() - target) ** 2).sum()
    return loss


class Context:
    """torchrun environment; degrades to single-process when RANK is absent."""

    def __init__(self, device):
        self.rank, self.world_size, self.device = 0, 1, device
        if "RANK" in os.environ:
            import torch.distributed as dist
            self.rank = int(os.environ["RANK"])
            self.world_size = int(os.environ["WORLD_SIZE"])
            local_rank = int(os.environ["LOCAL_RANK"])
            torch.cuda.set_device(local_rank)
            dist.init_process_group("nccl")
            self.device = f"cuda:{local_rank}"
        self.primary = self.rank == 0

    @property
    def distributed(self):
        return self.world_size > 1

    def close(self):
        if self.distributed:
            import torch.distributed as dist
            dist.barrier()
            dist.destroy_process_group()

    def barrier(self):
        if self.distributed:
            import torch.distributed as dist
            dist.barrier()

    def agree_step(self, want_step):
        """True iff any rank filled its budget; collective, so all ranks agree."""
        if not self.distributed:
            return want_step
        import torch.distributed as dist
        flag = torch.tensor([1 if want_step else 0], device=self.device)
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        return bool(flag.item())

    def sum_scalar(self, value):
        if not self.distributed:
            return value
        import torch.distributed as dist
        tensor = torch.tensor([float(value)], dtype=torch.float64, device=self.device)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return tensor.item()

    def broadcast_trainable(self, model):
        if not self.distributed:
            return
        import torch.distributed as dist
        for parameter in model.parameters():
            if parameter.requires_grad:
                dist.broadcast(parameter.data, src=0)

    def synchronize_gradients(self, parameters):
        """Sum gradients across ranks and average; zeros stand in for no-grad."""
        if not self.distributed:
            return
        import torch.distributed as dist
        for parameter in parameters:
            if parameter.grad is None:
                parameter.grad = torch.zeros_like(parameter)
            dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
            parameter.grad.div_(self.world_size)


def save_checkpoint(output, model, config, progress):
    output.mkdir(parents=True, exist_ok=True)
    torch.save({"trainable": trainable_state(model), "progress": progress}, output / "checkpoint.pt")
    (output / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run(config):
    context = Context(config.get("device", "cuda"))
    config = dict(config, device=context.device)
    seed = config.get("seed", 42)
    random.seed(seed)
    torch.manual_seed(seed)
    shuffle_rng = random.Random(seed)            # epoch order: identical on every rank
    branch_rng = random.Random(seed + context.rank)  # candidate shuffles: per rank
    data_path = Path(config["data"])
    records = read_jsonl(data_path)
    config = dict(config, data_sha256=hashlib.sha256(data_path.read_bytes()).hexdigest())
    model = build_model(config)
    context.broadcast_trainable(model)
    compiler = build_compiler(config, data_path.parent)
    groups = optimizer_groups(model, config)
    trainable = [p for g in groups for p in g["params"]]
    optimizer = torch.optim.AdamW(
        [{"params": g["params"], "lr": g["lr"]} for g in groups],
        weight_decay=config.get("weight_decay", 0.01))
    output = Path(config["output"])
    if context.primary:
        output.mkdir(parents=True, exist_ok=True)
        manifest = {"config": config, "world_size": context.world_size,
                    "trainable_parameters": {g["name"]: sum(p.numel() for p in g["params"]) for g in groups},
                    "optimizer_groups": [{"name": g["name"], "lr": g["lr"]} for g in groups]}
        (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps(manifest["trainable_parameters"]), flush=True)

    max_steps = config.get("max_steps", 2 ** 31)
    token_budget = config.get("tokens_per_step", 32768)
    if token_budget <= 0:
        raise ValueError("tokens_per_step must be positive")
    brier_weight = config.get("brier_weight", 0.0)
    soft_label_prob = config.get("soft_label_prob", 0.9)
    if not 0.0 < soft_label_prob <= 1.0:
        raise ValueError("soft_label_prob must be in (0, 1]")
    grad_clip = config.get("grad_clip", 1.0)
    save_every = config.get("save_every", 10)
    log_path = output / "training_log.jsonl"
    eval_data = config.get("eval_data")
    final_eval_data = config.get("final_eval_data")
    eval_records = read_jsonl(Path(eval_data)) if eval_data else None
    final_eval_records = None
    if final_eval_data:
        final_eval_records = eval_records if final_eval_data == eval_data else read_jsonl(Path(final_eval_data))

    def evaluate_all_ranks(records_to_eval, source, eval_output, at_step):
        """Paused-training val: every rank scores an interleaved shard of the val
        set with the in-memory model (no extra process, no extra VRAM), predictions
        are gathered to rank 0 and written inside the checkpoint directory."""
        model.eval()
        if context.device.startswith("cuda"):
            torch.cuda.empty_cache()
        eval_started = time.monotonic()
        predictions = []
        with torch.no_grad():
            for index in range(context.rank, len(records_to_eval), context.world_size):
                record = records_to_eval[index]
                compiled = compiler.compile(record, labeled_only=True)
                for question in compiled.questions:
                    logits = model(question)
                    meta = record.get("meta", {})
                    probabilities = logits.float().softmax(-1).cpu().tolist()
                    predictions.append({
                        "record_index": index, "record_id": meta.get("record_id"),
                        "group_id": record["group_id"], "qid": question.qid, "task": question.kind,
                        "modality": compiled.modality, "domain": meta.get("domain", "unknown"),
                        "metrics": question_metrics(question, logits),
                        "target": dict(zip(question.keys, question.target)),
                        "probabilities": dict(zip(question.keys, probabilities)),
                        "answer": question.keys[int(logits.argmax())]})
        if context.distributed:
            import torch.distributed as dist
            shards = [None] * context.world_size
            dist.all_gather_object(shards, predictions)
            predictions = [row for shard in shards for row in shard]
        if context.primary:
            predictions.sort(key=lambda row: (row["record_index"], str(row["qid"])))
            eval_output.mkdir(parents=True, exist_ok=True)
            with (eval_output / "predictions.jsonl").open("w", encoding="utf-8") as stream:
                for row in predictions:
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            report = {"training_step": at_step, "data": source,
                      "records": len(records_to_eval), "questions": len(predictions),
                      "world_size": context.world_size,
                      "elapsed_seconds": time.monotonic() - eval_started,
                      "metrics": summarize(predictions)}
            (eval_output / "metrics.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(json.dumps({"event": "val", "step": at_step, "data": source,
                              "accuracy": report["metrics"]["accuracy"],
                              "nll": report["metrics"]["nll"],
                              "accuracy_by_modality": report["metrics"]["accuracy_by_modality"],
                              "elapsed_seconds": report["elapsed_seconds"]}, ensure_ascii=False), flush=True)
        if context.device.startswith("cuda"):
            torch.cuda.empty_cache()
        model.train()

    last_saved_step = -1

    def save_and_eval(at_step, at_epoch):
        """All ranks together: rank 0 writes an immutable step-numbered checkpoint
        and repoints the latest symlink, then everyone runs the paused-training val."""
        nonlocal last_saved_step
        ckpt = output / f"step-{at_step:05d}"
        if at_step == last_saved_step:
            return ckpt
        last_saved_step = at_step
        if context.primary:
            save_checkpoint(ckpt, model, config, {"step": at_step, "epoch": at_epoch})
            latest = output / "latest"
            if latest.is_symlink():
                latest.unlink()
            elif latest.exists():
                shutil.rmtree(latest)
            latest.symlink_to(ckpt.name)
        context.barrier()
        if eval_records is not None:
            evaluate_all_ranks(eval_records, eval_data, ckpt / "eval", at_step)
        context.barrier()
        return ckpt

    step, pending_tokens, pending_losses, pending_questions = 0, 0, [], 0
    model.train()
    started = time.monotonic()
    done = False
    for epoch in range(config.get("epochs", 1)):
        if done:
            break
        order = list(range(len(records)))
        shuffle_rng.shuffle(order)
        # Same record count per rank: wrap the tail like DistributedSampler.
        per_rank = math.ceil(len(order) / context.world_size)
        padded = order + order[: per_rank * context.world_size - len(order)]
        shard = padded[context.rank::context.world_size]
        for index in shard:
            record = records[index]
            compiled = compiler.compile(record, rng=branch_rng, labeled_only=True)
            if config.get("stage") == "text" and compiled.modality != "text":
                raise ValueError(f"stage=text requires a text-only dataset, got a {compiled.modality} record "
                                 f"(group_id={record.get('group_id')})")
            for question in compiled.questions:
                logits = model(question)
                loss = question_loss(logits, soften_target(question.target, soft_label_prob), brier_weight)
                loss.backward()
                pending_losses.append(float(loss.detach()))
            pending_tokens += compiled.compute_tokens
            pending_questions += len(compiled.questions)
            if context.agree_step(pending_tokens >= token_budget):
                context.synchronize_gradients(trainable)
                if grad_clip:
                    torch.nn.utils.clip_grad_norm_(trainable, grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if context.device.startswith("cuda"):
                    torch.cuda.empty_cache()
                step += 1
                total_tokens = context.sum_scalar(pending_tokens)
                total_questions = context.sum_scalar(pending_questions)
                loss_sum = context.sum_scalar(sum(pending_losses))
                if context.primary:
                    entry = {"step": step, "epoch": epoch, "tokens": int(total_tokens),
                             "questions": int(total_questions),
                             "loss_mean": loss_sum / max(total_questions, 1),
                             "elapsed_seconds": time.monotonic() - started}
                    with log_path.open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(entry) + "\n")
                    print(json.dumps(entry), flush=True)
                pending_tokens, pending_losses, pending_questions = 0, [], 0
                if step % save_every == 0:
                    save_and_eval(step, epoch)
                if step >= max_steps:
                    done = True
                    break
        if done:
            break
    # Flush a trailing partial accumulation so short datasets still update.
    if context.agree_step(bool(pending_losses)):
        context.synchronize_gradients(trainable)
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(trainable, grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1
    final_ckpt = save_and_eval(step, config.get("epochs", 1))
    if final_eval_records is not None:
        evaluate_all_ranks(final_eval_records, final_eval_data, final_ckpt / "eval_full", step)
        context.barrier()
    if context.primary:
        print(json.dumps({"finished": True, "steps": step,
                          "elapsed_seconds": time.monotonic() - started}), flush=True)
    context.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", help="Override output directory")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if args.output:
        config["output"] = args.output
    if not math.isfinite(config.get("brier_weight", 0.0)) or config.get("brier_weight", 0.0) < 0:
        raise ValueError("brier_weight must be finite and nonnegative")
    run(config)


if __name__ == "__main__":
    main()
