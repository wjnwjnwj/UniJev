"""Soft-label SFT with the distinct original 2B and 9B update recipes."""
import json
import math
import random
from pathlib import Path

from .common.config import load_config, parser, configure_determinism
from .common.data import check_data


def soften_target(target, correct_prob=0.9):
    size = len(target)
    if size < 2 or not math.isclose(max(target), 1.0, abs_tol=1e-6):
        return target
    correct = max(range(size), key=target.__getitem__)
    rest = (1.0 - correct_prob) / (size - 1)
    return [correct_prob if i == correct else rest for i in range(size)]


def question_loss(logits, target, model_size):
    import torch
    if model_size == "2b":
        target = torch.tensor(soften_target(target), dtype=torch.float32, device=logits.device)
    else:
        # Preserve the 9B recipe's argmax conversion, including soft input labels.
        correct = int(torch.tensor(target).argmax())
        target = torch.full_like(logits.float(), 0.1 / (len(target) - 1))
        target[correct] = 0.9
    return -(target * logits.float().log_softmax(-1)).sum()


def _rng_state(context, **extra):
    import torch
    return dict(torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state() if context.device.startswith("cuda") else None,
                **extra)


def _restore_rng(state, context):
    import torch
    torch.set_rng_state(state["torch_rng"])
    if context.device.startswith("cuda"):
        torch.cuda.set_rng_state(state["cuda_rng"])


def train_2b(config, records, model, compiler, optimizer, parameters, context, output, resume):
    """Per-rank token budget, padded shards, rank-mean accumulated gradients."""
    import torch
    from .common.checkpoint import save_checkpoint, load_checkpoint, load_training_state
    shuffle_rng = random.Random(config["seed"])
    branch_rng = random.Random(config["seed"] + context.rank)
    step, start_epoch, next_offset, saved_order = 0, 0, 0, None
    last_saved = -1
    if resume:
        progress = load_checkpoint(model, output / "latest")
        states = load_training_state(output / "latest", progress, optimizer)
        state = states[context.rank]
        step, start_epoch, next_offset = progress["step"], state["epoch"], state["next_offset"]
        shuffle_rng.setstate(state["shuffle_rng"])
        branch_rng.setstate(state["branch_rng"])
        saved_order = state["order"]
        _restore_rng(state, context)
        last_saved = step

    def save(epoch, offset, order):
        nonlocal last_saved
        if last_saved == step:
            return
        states = context.gather(_rng_state(context, epoch=epoch, next_offset=offset, order=order,
                                shuffle_rng=shuffle_rng.getstate(), branch_rng=branch_rng.getstate()))
        context.on_primary(lambda: save_checkpoint(output / f"step-{step:05d}", model, config,
                           {"step": step, "epoch": epoch}, optimizer, states))
        last_saved = step

    pending_tokens, pending_questions, pending_losses = 0, 0, []
    token_budget = config["tokens_per_step"]
    model.train()
    epoch, offset, order = start_epoch, next_offset, []
    for epoch in range(start_epoch, config["epochs"]):
        if step >= config["max_steps"]:
            break
        if saved_order is not None:
            order, saved_order = saved_order, None
        else:
            order = list(range(len(records)))
            shuffle_rng.shuffle(order)
        per_rank = math.ceil(len(order) / context.world_size)
        padded = order + order[:per_rank * context.world_size - len(order)]
        shard = padded[context.rank::context.world_size]
        for offset in range(next_offset, len(shard)):
            compiled = compiler.compile(records[shard[offset]], rng=branch_rng, labeled_only=True)
            for question in compiled.questions:
                loss = question_loss(model(question), question.target, "2b")
                loss.backward()
                pending_losses.append(float(loss.detach()))
            pending_tokens += compiled.compute_tokens
            pending_questions += len(compiled.questions)
            if not context.agree_step(pending_tokens >= token_budget):
                continue
            context.average_rank_gradients(parameters)
            torch.nn.utils.clip_grad_norm_(parameters, config["grad_clip"])
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if context.device.startswith("cuda"):
                torch.cuda.empty_cache()
            step += 1
            tokens = context.sum(pending_tokens)
            questions = context.sum(pending_questions)
            loss_sum = context.sum(sum(pending_losses))
            _log(context, output, {"step": step, "epoch": epoch, "tokens": int(tokens),
                                  "questions": int(questions), "loss_mean": loss_sum / max(questions, 1)})
            pending_tokens, pending_questions, pending_losses = 0, 0, []
            if step % config["save_every"] == 0 or step == config["max_steps"]:
                save(epoch, offset + 1, order)
            if step >= config["max_steps"]:
                break
        next_offset = 0
    if context.agree_step(bool(pending_losses)):
        context.average_rank_gradients(parameters)
        torch.nn.utils.clip_grad_norm_(parameters, config["grad_clip"])
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1
    save(epoch, offset + 1, order)
    return step


def train_9b(config, records, model, compiler, optimizer, parameters, context, output, resume):
    """Global token budget, unpadded rounds, global question-mean gradients."""
    import torch
    from contextlib import nullcontext
    from .common.checkpoint import save_checkpoint, load_checkpoint, load_training_state
    shuffle = random.Random(config["seed"])
    step, epoch, resume_offset = 0, 0, 0
    total_tokens, total_questions = 0, 0
    if resume:
        progress = load_checkpoint(model, output / "latest")
        states = load_training_state(output / "latest", progress, optimizer)
        state = states[context.rank]
        step, epoch = progress["step"], progress["epoch"]
        total_tokens, total_questions = progress["total_tokens"], progress["total_questions"]
        resume_offset = state["next_record_offset"]
        for _ in range(epoch):
            dummy = list(range(len(records)))
            shuffle.shuffle(dummy)
        _restore_rng(state, context)
    else:
        torch.manual_seed(config["seed"] + context.rank)
    pending_tokens, local_questions, loss_sum = 0, 0, 0.0
    model.train()
    while step < config["max_steps"]:
        order = list(range(len(records)))
        shuffle.shuffle(order)
        for offset in range(0, len(order), context.world_size):
            if offset < resume_offset:
                continue
            position = offset + context.rank
            round_tokens = 0
            if position < len(order):
                index = order[position]
                candidate_rng = random.Random(config["seed"] + epoch * len(records) + index)
                compiled = compiler.compile(records[index], rng=candidate_rng, labeled_only=True)
                round_tokens = compiled.compute_tokens
                for question in compiled.questions:
                    offload = torch.autograd.graph.save_on_cpu(pin_memory=True) if config["activation_cpu_offload"] else nullcontext()
                    with offload:
                        loss = question_loss(model(question), question.target, "9b")
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"Nonfinite loss at record {index}")
                    loss.backward()
                    loss_sum += float(loss.detach())
                    local_questions += 1
            pending_tokens += int(context.sum(round_tokens))
            if pending_tokens < config["global_tokens_per_step"]:
                continue
            questions = int(context.sum(local_questions))
            context.mean_gradients(parameters, questions)
            norm = torch.nn.utils.clip_grad_norm_(parameters, config["grad_clip"], error_if_nonfinite=True)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            total_tokens += pending_tokens
            total_questions += questions
            losses = context.gather(loss_sum)
            _log(context, output, {"step": step, "epoch": epoch, "global_tokens": pending_tokens,
                                  "global_questions": questions, "gradient_norm": float(norm),
                                  "loss_mean": sum(losses) / questions})
            pending_tokens, local_questions, loss_sum = 0, 0, 0.0
            if step % config["save_every"] == 0 or step == config["max_steps"]:
                progress = {"step": step, "epoch": epoch, "total_tokens": total_tokens,
                            "total_questions": total_questions}
                states = context.gather(_rng_state(context, next_record_offset=offset + context.world_size))
                context.on_primary(lambda: save_checkpoint(output / f"step-{step:05d}", model,
                                                           config, progress, optimizer, states))
            if step == config["max_steps"]:
                break
        epoch += 1
        resume_offset = 0
    return step


def _log(context, output, entry):
    def write():
        with (output / "logs/training.jsonl").open("a") as stream:
            stream.write(json.dumps(entry, allow_nan=False) + "\n")
        print(json.dumps(entry), flush=True)
    context.on_primary(write)


def run(config, records, audit, resume=False):
    import torch
    from .common.model import build_model, build_compiler, optimizer_groups
    from .common.distributed import Context
    from .common.checkpoint import initialize_run, write_json
    configure_determinism(config)
    context = Context(config["device"])
    try:
        if context.world_size != config["world_size"]:
            raise ValueError("Launch with the configured world size (or use --smoke)")
        output = initialize_run(config, context, resume, audit)
        if config["model_size"] == "2b":
            random.seed(config["seed"])
        torch.manual_seed(config["seed"])
        model = build_model(dict(config, device=context.device))
        compiler = build_compiler(config, config["media_root"])
        groups = optimizer_groups(model, config)
        parameters = [p for group in groups for p in group["params"]]
        context.broadcast(parameters)
        optimizer = torch.optim.AdamW([{"params": g["params"], "lr": g["lr"]} for g in groups],
                                      weight_decay=config["weight_decay"])
        trainer = train_2b if config["model_size"] == "2b" else train_9b
        steps = trainer(config, records, model, compiler, optimizer, parameters, context, output, resume)
        context.on_primary(lambda: write_json(output / "finished.json", {"steps": steps,
                           "final_checkpoint": f"step-{steps:05d}"}))
    finally:
        context.close()


def main():
    args = parser(__doc__).parse_args()
    config = load_config(args, "sft")
    records, audit = check_data(config)
    if args.check_data:
        print(json.dumps(audit, indent=2))
        return
    run(config, records, audit, args.resume)


if __name__ == "__main__":
    main()
