"""Soft-label SFT of the shared decision head, LoRA adapters and vision merger.

Launch with train/run_sft.sh. Each rank walks its shard of a per-epoch shuffle,
accumulating gradients until it has its share of the update: batch_size /
world_size questions with --batch-size, otherwise tokens_per_step / world_size
compute tokens. Ranks then synchronize once per optimizer update (one small all-reduce
for statistics, one flat all-reduce per dtype for gradients). Media decoding
and tokenization run ahead in DataLoader workers. No validation is run here.
"""
import argparse
import json
import math
import random
import time
from contextlib import nullcontext

from .common.runtime import (MODALITIES, add_common_arguments, all_reduce_vector, audit_records,
                             build_config, build_optimizer, limit_threads, load_trainable,
                             make_loader, prepare_run, reduce_gradients, save)


def soften_target(target, correct_prob=0.9):
    """One-hot -> correct_prob on the gold answer, the rest spread evenly; soft targets pass through."""
    size = len(target)
    if size < 2 or not math.isclose(max(target), 1.0, abs_tol=1e-6):
        return target
    correct = max(range(size), key=target.__getitem__)
    rest = (1.0 - correct_prob) / (size - 1)
    return [correct_prob if i == correct else rest for i in range(size)]


def question_loss(logits, target, correct_prob):
    import torch
    target = torch.tensor(soften_target(target, correct_prob), dtype=torch.float32, device=logits.device)
    return -(target * logits.float().log_softmax(-1)).sum()


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_arguments(ap, "sft")
    g = ap.add_argument_group("sft")
    g.add_argument("--epochs", type=int, default=1, help="passes over the data")
    g.add_argument("--batch-size", type=int, default=0,
                   help="questions per optimizer update, summed over GPUs (0 = use --tokens-per-step)")
    g.add_argument("--tokens-per-step", type=int, default=262144,
                   help="global compute tokens per optimizer update; used when --batch-size is 0")
    g.add_argument("--grad-reduce", choices=["question_mean", "rank_mean"], default="question_mean",
                   help="question_mean: mean over every question in the update (independent of #GPUs); "
                        "rank_mean: per-GPU summed losses averaged over GPUs (the original 2B recipe)")
    g.add_argument("--soft-label-prob", type=float, default=0.9, help="probability kept on the gold answer")
    return ap.parse_args()


def main():
    import torch
    from .common.distributed import Context
    from .common.model import build_model
    from .common.schema import read_jsonl
    from .common.checkpoint import load_checkpoint, load_training_state
    args = parse()
    config = build_config(args, "sft")
    for key in ("epochs", "tokens_per_step"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if type(config["batch_size"]) is not int or config["batch_size"] < 0:
        raise ValueError("batch_size must be a non-negative integer")
    if not 0 < config["soft_label_prob"] <= 1:
        raise ValueError("soft_label_prob must be in (0, 1]")
    limit_threads(config["threads"])
    records = read_jsonl(config["data"])
    if args.check_data:
        print(json.dumps(audit_records(records, config), indent=2, ensure_ascii=False))
        return

    context = Context("cuda")
    try:
        holder = {}
        context.on_primary(lambda: holder.update(audit=audit_records(records, config)))
        run = prepare_run(config, context, holder.get("audit"))
        torch.manual_seed(config["seed"])
        model = build_model(dict(config, device=context.device))
        optimizer, scheduler, parameters = build_optimizer(model, config)
        progress, offset = {"step": 0, "epoch": 0}, 0
        if config["resume"]:
            latest = run.output / "latest"
            progress = load_checkpoint(model, latest)
            state = load_training_state(latest, progress, optimizer)[context.rank]
            scheduler.load_state_dict(state["scheduler"])
            offset = state["offset"]
        elif config.get("init_ckpt"):
            info = load_trainable(model, config["init_ckpt"])
            run.write_json("init_checkpoint.json", info)
        context.broadcast(parameters)
        model.train()
        train(config, records, model, optimizer, scheduler, parameters, context, run, progress, offset)
    finally:
        context.close()


def train(config, records, model, optimizer, scheduler, parameters, context, run, progress, offset):
    import torch
    world, rank = context.world_size, context.rank
    max_steps = config["max_steps"] or math.inf
    if config["batch_size"]:
        if config["batch_size"] < world:
            raise ValueError(f"--batch-size {config['batch_size']} is smaller than the {world} GPUs")
        # Remainder questions go to the first ranks; records with several questions may overshoot slightly.
        local_quota = config["batch_size"] // world + (rank < config["batch_size"] % world)
        def needs_more(tokens, questions):
            return questions < local_quota
    else:
        local_budget = config["tokens_per_step"] / world
        def needs_more(tokens, questions):
            return tokens < local_budget
    offload = (lambda: torch.autograd.graph.save_on_cpu(pin_memory=True)) \
        if config["activation_cpu_offload"] else nullcontext
    step, saved_step = progress["step"], progress["step"]
    epoch, consumed = progress["epoch"], offset
    slots = {m: i for i, m in enumerate(MODALITIES)}

    def checkpoint():
        nonlocal saved_step
        states = context.gather({"offset": consumed, "epoch": epoch})
        save(run, step, model, config, {"step": step, "epoch": epoch}, optimizer, scheduler, states)
        saved_step = step

    if config["max_steps"]:
        run.start_progress(config["max_steps"], initial=step, unit="step", desc="sft")
    else:  # epoch-driven: count records over all epochs
        done = all_reduce_vector([consumed], context)[0]
        run.start_progress(len(records) * config["epochs"], initial=int(epoch * len(records) + done),
                           unit="rec", desc="sft")

    while epoch < config["epochs"] and step < max_steps:
        order = list(range(len(records)))
        random.Random(config["seed"] * 1_000_003 + epoch).shuffle(order)
        shard = order[rank::world]
        base = config["seed"] * 1_000_003 + epoch * len(records)
        items = [(i, None, base + i) for i in shard[consumed:]]
        stream = iter(make_loader(records, items, config))
        exhausted = False
        while step < max_steps:
            started = time.time()
            tokens = questions = loss_sum = correct = nrecords = skipped = nonfinite = 0
            wait = 0.0
            per_modality = [0.0] * (3 * len(MODALITIES))  # questions, loss, correct
            while not exhausted and needs_more(tokens, questions):
                fetched = time.time()
                try:
                    item, compiled, error = next(stream)
                except StopIteration:
                    exhausted = True
                    break
                wait += time.time() - fetched
                consumed += 1
                nrecords += 1
                if error is not None:
                    run.error(item, error)
                    skipped += 1
                    continue
                slot = slots[compiled.modality]
                for question in compiled.questions:
                    with offload():
                        logits = model(question)
                        loss = question_loss(logits, question.target, config["soft_label_prob"])
                    if not torch.isfinite(loss):
                        nonfinite += 1
                        continue
                    loss.backward()
                    value = float(loss.detach())
                    hit = int(int(logits.detach().argmax()) == max(range(len(question.target)),
                                                                     key=question.target.__getitem__))
                    questions += 1
                    loss_sum += value
                    correct += hit
                    per_modality[3 * slot] += 1
                    per_modality[3 * slot + 1] += value
                    per_modality[3 * slot + 2] += hit
                tokens += compiled.compute_tokens
            totals = all_reduce_vector([tokens, questions, loss_sum, correct, nrecords, skipped,
                                        nonfinite, int(exhausted), wait] + per_modality, context)
            (g_tokens, g_questions, g_loss, g_correct, g_records, g_skipped,
             g_nonfinite, g_exhausted, g_wait), g_modality = totals[:9], totals[9:]
            if g_questions > 0:
                divisor = world if config["grad_reduce"] == "rank_mean" else g_questions
                reduce_gradients(parameters, divisor, context)
                norm = float(torch.nn.utils.clip_grad_norm_(parameters, config["grad_clip"]))
                update_skipped = not math.isfinite(norm)
                if update_skipped and not config["skip_nonfinite_update"]:
                    raise FloatingPointError(f"Non-finite gradient norm at step {step + 1}")
                if not update_skipped:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                lrs = {group["name"]: group["lr"] for group in optimizer.param_groups}
                scheduler.step()
                step += 1
                seconds = time.time() - started
                run.advance(1 if config["max_steps"] else int(g_records), {
                    "step": step, "ep": epoch, "loss": f"{g_loss / g_questions:.4f}",
                    "acc": f"{g_correct / g_questions:.3f}",
                    "tok/s": f"{g_tokens / max(seconds, 1e-9):.0f}",
                    "lr": f"{lrs.get('lora', next(iter(lrs.values()))):.2e}"})
                run.log("train.jsonl", {
                    "step": step, "epoch": epoch, "loss": g_loss / g_questions,
                    "train_acc": g_correct / g_questions, "questions": int(g_questions),
                    "records": int(g_records), "tokens": int(g_tokens),
                    "grad_norm": norm if math.isfinite(norm) else None,
                    "update_skipped": update_skipped, "lr": lrs,
                    "skipped_records": int(g_skipped), "nonfinite_losses": int(g_nonfinite),
                    "step_seconds": round(seconds, 3), "data_wait_seconds_mean": round(g_wait / world, 3),
                    "tokens_per_second": round(g_tokens / max(seconds, 1e-9), 1),
                    "by_modality": {m: {"questions": int(g_modality[3 * i]),
                                        "loss": g_modality[3 * i + 1] / g_modality[3 * i],
                                        "acc": g_modality[3 * i + 2] / g_modality[3 * i]}
                                    for i, m in enumerate(MODALITIES) if g_modality[3 * i]},
                    "elapsed_seconds": round(time.time() - run.started, 1)}, echo=run.echo(step))
                if step % config["save_every"] == 0 or step >= max_steps:
                    checkpoint()
            if g_exhausted == world:
                break
        if step >= max_steps:
            break
        epoch, consumed = epoch + 1, 0
    if saved_step != step:
        checkpoint()
    run.close_progress()
    run.write_json("finished.json", {"steps": step, "epochs_completed": epoch,
                                     "final_checkpoint": f"step-{step:05d}",
                                     "seconds": round(time.time() - run.started, 1)})


if __name__ == "__main__":
    main()
