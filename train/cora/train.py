"""Train CORA with fresh confidence-bin accuracy targets on every output group."""
import json
import random
from pathlib import Path
from ..common.config import load_config, parser
from ..common.data import check_data, build_partitions


def forward_question(model, compiler, records, item):
    dataset = records[item["split"]] if isinstance(records, dict) else records
    compiled = compiler.compile(dataset[item["record_index"]], labeled_only=True)
    question = next(q for q in compiled.questions if q.qid == item["qid"])
    return model(question), question.keys


def predict(model, compiler, records, item):
    logits, keys = forward_question(model, compiler, records, item)
    from ..common.data import expected_questions, modality
    dataset = records[item["split"]] if isinstance(records, dict) else records
    record = dataset[item["record_index"]]
    probabilities = dict(zip(keys, logits.detach().float().softmax(-1).cpu().tolist()))
    truth = expected_questions([record])[(0, item["qid"])]
    return {"record_index": item["record_index"], "qid": item["qid"], "group_id": record["group_id"],
            "modality": modality(record), "task": truth["task"], "target": truth["target"],
            "probabilities": probabilities}


def score(rows):
    from .monitor import metrics, confidence_metrics
    clean = []
    for row in rows:
        # Canonical ordering matches final-suite argmax, including ties.
        p = {key: row["probabilities"][key] for key in row["target"]}
        hit = max(p, key=p.get) == max(row["target"], key=row["target"].get)
        clean.append(dict(row, probabilities=p, confidence=max(p.values()), metrics={"accuracy": int(hit)}))
    return dict(metrics(clean), binary=confidence_metrics(clean, "confidence"))


def validate(model, compiler, records, items, context):
    import torch
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            rows = [predict(model, compiler, records, item)
                    for item in items[context.rank::context.world_size]]
    finally:
        model.train(was_training)
    rows = [row for shard in context.gather(rows) for row in shard]
    rows.sort(key=lambda r: (r["record_index"], str(r["qid"])))
    return score(rows)


def train(c, records, split, audit, resume=False):
    import torch
    from ..common.model import build_model, build_compiler, optimizer_groups
    from ..common.distributed import Context
    from ..common.checkpoint import (initialize_run, load_checkpoint, load_training_state,
                                     save_checkpoint, write_json, point_to)
    from .objective import update
    def point_best(output, step):
        point_to(output, "best", step)
    context = Context(c["device"])
    try:
        if context.world_size != c["world_size"]:
            raise ValueError("Launch with configured world size")
        output = initialize_run(c, context, resume, dict(audit, split=split["audit"]))
        if not resume:
            context.on_primary(lambda: write_json(output / "split.json", split))
        torch.manual_seed(c["seed"])
        model = build_model(dict(c, device=context.device))
        load_checkpoint(model, c["init_checkpoint"], initial_sft=True)
        model.train()
        parameters = [p for p in model.parameters() if p.requires_grad]
        context.broadcast(parameters)
        compiler = build_compiler(c, c["media_root"])
        groups = optimizer_groups(model, c)
        optimizer = torch.optim.AdamW([{"params": g["params"], "lr": g["lr"]} for g in groups],
                                      weight_decay=0.0)
        training = [r for r in split["items"] if r["split"] == "train"]
        validation = [r for r in split["items"] if r["split"] == "validation"]
        checkpoint_config = c
        if resume:
            progress = load_checkpoint(model, output / "latest")
            load_training_state(output / "latest", progress, optimizer)
        else:
            baseline = validate(model, compiler, records, validation, context)
            progress = {"step": 0, "best_step": 0, "best_brier": baseline["binary"]["correctness_brier"]}
            context.on_primary(lambda: write_json(output / "logs/validation-00000.json", baseline))
            context.on_primary(lambda: save_checkpoint(output / "step-00000", model,
                                                       checkpoint_config, progress, optimizer, []))
            context.on_primary(lambda: point_best(output, 0))
        for step in range(progress["step"] + 1, c["max_steps"] + 1):
            # Randomness selects distinct questions, never confidence actions.
            batch = random.Random(c["seed"] + step).sample(training, min(c["group_questions"], len(training)))
            stats = update(model, optimizer, batch,
                           lambda item: forward_question(model, compiler, records, item),
                           context, c)
            context.on_primary(lambda: write_json(output / f"logs/train-{step:05d}.json", dict(step=step, **stats)))
            if context.primary:
                print(json.dumps({"step": step, **{k: v for k, v in stats.items() if k != "bins"}}), flush=True)
            progress["step"] = step
            if step % c["eval_every"] == 0 or step == c["max_steps"]:
                measured = validate(model, compiler, records, validation, context)
                brier = measured["binary"]["correctness_brier"]
                if brier < progress["best_brier"]:
                    progress.update(best_step=step, best_brier=brier)
                context.on_primary(lambda: write_json(output / f"logs/validation-{step:05d}.json", measured))
                context.on_primary(lambda: save_checkpoint(output / f"step-{step:05d}", model,
                                                           checkpoint_config, dict(progress), optimizer, []))
                context.on_primary(lambda: point_best(output, progress["best_step"]))
        context.on_primary(lambda: point_best(output, progress["best_step"]))
        context.on_primary(lambda: write_json(output / "finished.json", {
            "method": "cora_supervised_bins", "steps": c["max_steps"],
            "selected_checkpoint": f'step-{progress["best_step"]:05d}',
            "selection": "internal_validation_binary_correctness_brier; earliest on ties, includes SFT",
            "validation_brier": progress["best_brier"], "gradient_training": True,
            "trainable_parameters": sum(p.numel() for p in parameters),
            "test_used_for_selection": False}))
    finally:
        context.close()


def main():
    args = parser(__doc__).parse_args()
    c = load_config(args, "cora")
    records, audit = check_data(c)
    split = build_partitions(records, c["internal_validation_fraction"])
    if not c.get("smoke") and split["audit"]["train_questions"] < c["group_questions"]:
        raise ValueError("The full recipe requires at least group_questions training questions; use --smoke for examples")
    if args.check_data:
        print(json.dumps(dict(audit, internal_split=split["audit"]), indent=2))
        return
    train(c, records, split, audit, args.resume)


if __name__ == "__main__":
    main()
