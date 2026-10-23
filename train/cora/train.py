"""CORA: train each answer's confidence toward its confidence bin's current accuracy.

Launch with train/run_cora.sh. Every update draws `group_questions` labeled
questions, scores them once without gradients, bins the whole group by the
confidence of the selected answer (15 equal-width bins by default, as in
test/eval.py), and regresses each supported question's confidence onto its bin's
empirical accuracy. Questions are compiled once in DataLoader workers and the
same compiled inputs serve both passes; each rank receives an equal mix of
video, image and text. No validation is run here.
"""
import argparse
import json
import math
import random
import time

from ..common.data import modality as record_modality
from ..common.runtime import (add_common_arguments, all_reduce_vector, audit_records, balanced_shard,
                             build_config, build_optimizer, limit_threads, load_trainable,
                             make_loader, prepare_run, reduce_gradients, save)
from ..common.schema import candidates, target_distribution
from .objective import loss as cora_loss
from .targets import group_targets


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_arguments(ap, "cora")
    g = ap.add_argument_group("cora")
    g.add_argument("--group-questions", type=int, default=256, help="questions binned together per update")
    g.add_argument("--bins", type=int, default=15, help="equal-width confidence bins (15 = test/eval.py ECE)")
    g.add_argument("--min-bin-count", type=int, default=20, help="bins with fewer questions get no target")
    return ap.parse_args()


def labeled_items(records):
    """(record_index, qid, gold_key) for every one-hot labeled question."""
    items, soft = [], 0
    for index, record in enumerate(records):
        for qid, question in record["request"]["questions"].items():
            keys = [k for k, _ in candidates(question)]
            target = target_distribution(record.get("targets", {}).get(qid), keys)
            if target is None or len(keys) < 2:
                continue
            if target.count(1.0) != 1 or any(v not in (0, 1) for v in target):
                soft += 1  # CORA needs a single correct answer
                continue
            items.append((index, qid, keys[target.index(1.0)]))
    return items, soft


def main():
    import torch
    from ..common.distributed import Context
    from ..common.model import build_model
    from ..common.schema import read_jsonl
    from ..common.checkpoint import load_checkpoint, load_training_state
    args = parse()
    config = build_config(args, "cora")
    for key in ("max_steps", "group_questions", "bins", "min_bin_count"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if not config.get("init_ckpt") and not config["resume"]:
        raise ValueError("CORA starts from an SFT checkpoint: pass --init-ckpt")
    limit_threads(config["threads"])
    records = read_jsonl(config["data"])
    items, soft = labeled_items(records)
    if args.check_data:
        print(json.dumps(dict(audit_records(records, config), cora_questions=len(items),
                              skipped_soft_targets=soft), indent=2, ensure_ascii=False))
        return

    context = Context("cuda")
    try:
        if config["group_questions"] < context.world_size:
            raise ValueError("--group-questions must be at least the number of GPUs")
        holder = {}
        context.on_primary(lambda: holder.update(audit=dict(
            audit_records(records, config), cora_questions=len(items), skipped_soft_targets=soft)))
        run = prepare_run(config, context, holder.get("audit"))
        torch.manual_seed(config["seed"])
        model = build_model(dict(config, device=context.device))
        optimizer, scheduler, parameters = build_optimizer(model, config)
        progress = {"step": 0}
        if config["resume"]:
            latest = run.output / "latest"
            progress = load_checkpoint(model, latest)
            state = load_training_state(latest, progress, optimizer)[context.rank]
            scheduler.load_state_dict(state["scheduler"])
        else:
            run.write_json("init_checkpoint.json", load_trainable(model, config["init_ckpt"]))
        context.broadcast(parameters)
        model.train()
        train(config, records, items, model, optimizer, scheduler, parameters, context, run, progress)
    finally:
        context.close()


def train(config, records, items, model, optimizer, scheduler, parameters, context, run, progress):
    import torch
    world, rank, bins = context.world_size, context.rank, config["bins"]
    first = progress["step"] + 1
    steps = list(range(first, config["max_steps"] + 1))
    # Randomness selects distinct questions per update, never confidence actions.
    plan = []
    for step in steps:
        group = random.Random(config["seed"] + step).sample(items, min(config["group_questions"], len(items)))
        plan.append(balanced_shard(group, records, rank, world, key=lambda item: item[0]))
    stream = iter(make_loader(records, [(i, q, None) for local in plan for i, q, _ in local], config))
    step, saved_step = progress["step"], progress["step"]
    run.start_progress(config["max_steps"], initial=step, unit="step", desc="cora")

    for step, local in zip(steps, plan):
        started = time.time()
        wait, kept, observations, failed = 0.0, [], [], 0
        with torch.no_grad():
            for record_index, qid, gold in local:
                fetched = time.time()
                item, compiled, error = next(stream)
                wait += time.time() - fetched
                if error is not None or not compiled.questions:
                    run.error(item, error or "question not compiled")
                    failed += 1
                    continue
                question = compiled.questions[0]
                p = model(question).float().softmax(-1)
                index = int(p.argmax())
                key = f"{record_index}/{qid}"
                row = {"key": key, "confidence": float(p[index]),
                       "correct": int(question.keys[index] == gold),
                       "modality": record_modality(records[record_index])}
                observations.append(row)
                kept.append((key, question, index, row["confidence"]))
        scored = time.time()
        group = [row for shard in context.gather(observations) for row in shard]
        targets, fitted = group_targets(group, bins, config["min_bin_count"])
        local_loss, used, mismatched = 0.0, 0, 0
        for key, question, index, confidence in kept:
            if targets[key] is None:
                continue
            logits = model(question)
            if int(logits.detach().float().argmax()) != index:
                mismatched += 1  # numerically unstable tie; its bin target no longer applies
                continue
            value = cora_loss(logits, targets[key])
            value.backward()
            local_loss += float(value.detach())
            used += 1
        g_loss, g_used, g_mismatched, g_failed, g_wait = all_reduce_vector(
            [local_loss, used, mismatched, failed, wait], context)
        norm, update_skipped = None, True
        if g_used > 0:
            reduce_gradients(parameters, g_used, context)
            norm = float(torch.nn.utils.clip_grad_norm_(parameters, config["grad_clip"]))
            update_skipped = not math.isfinite(norm)
            if update_skipped and not config["skip_nonfinite_update"]:
                raise FloatingPointError(f"Non-finite gradient norm at step {step}")
            if not update_skipped:
                optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        lrs = {g["name"]: g["lr"] for g in optimizer.param_groups}
        scheduler.step()
        seconds = time.time() - started
        count = len(group)
        ece = sum(abs(g["successes"] - g["mean_confidence"] * g["count"])
                  for g in fitted["groups"] if g["count"]) / count
        by_modality = {}
        for m in sorted({r["modality"] for r in group}):
            rows = [r for r in group if r["modality"] == m]
            by_modality[m] = {"questions": len(rows), "acc": sum(r["correct"] for r in rows) / len(rows),
                              "mean_confidence": sum(r["confidence"] for r in rows) / len(rows)}
        run.advance(1, {"loss": f"{g_loss / g_used:.4f}" if g_used else "-",
                        "acc": f"{sum(r['correct'] for r in group) / count:.3f}",
                        f"ece{bins}": f"{ece:.4f}", "support": f"{g_used / count:.2f}",
                        "lr": f"{lrs.get('lora', next(iter(lrs.values()))):.2e}"})
        run.log("train.jsonl", {
            "step": step, "loss": g_loss / g_used if g_used else None, "questions": count,
            "supported_questions": int(g_used), "support_fraction": g_used / count,
            "recompute_mismatch": int(g_mismatched), "failed_questions": int(g_failed),
            "group_acc_before_update": sum(r["correct"] for r in group) / count,
            "group_mean_confidence": sum(r["confidence"] for r in group) / count,
            f"group_ece_{bins}_before_update": ece, "by_modality": by_modality,
            "grad_norm": norm if norm is None or math.isfinite(norm) else None,
            "update_skipped": update_skipped, "lr": lrs,
            "step_seconds": round(seconds, 3), "score_pass_seconds": round(scored - started, 3),
            "data_wait_seconds_mean": round(g_wait / world, 3),
            "elapsed_seconds": round(time.time() - run.started, 1)}, echo=run.echo(step))
        run.log("bins.jsonl", {"step": step, "bins": fitted["groups"]}, echo=False)
        if step % config["save_every"] == 0 or step == config["max_steps"]:
            save(run, step, model, config, {"step": step}, optimizer, scheduler, context.gather({}))
            saved_step = step
    if saved_step != step:
        save(run, step, model, config, {"step": step}, optimizer, scheduler, context.gather({}))
    run.close_progress()
    run.write_json("finished.json", {"steps": step, "final_checkpoint": f"step-{step:05d}",
                                     "bins": bins, "seconds": round(time.time() - run.started, 1)})


if __name__ == "__main__":
    main()
