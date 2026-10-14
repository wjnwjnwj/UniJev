"""Train UniJev on its trimodal training set using current output-group accuracy within one modality per group."""
import argparse
import json
import os
import random
from pathlib import Path
from .configuration import load_config, prepare_inputs, ROOT
from . import VERSION


def source_hashes():
    from campaign_train_v2.configuration import sha256
    return {str(p.relative_to(ROOT)): sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))}


def forward_question(model, compiler, records, item):
    dataset = records[item["split"]] if isinstance(records, dict) else records
    compiled = compiler.compile(dataset[item["record_index"]], labeled_only=True)
    question = next(q for q in compiled.questions if q.qid == item["qid"])
    return model(question), question.keys


def predict(model, compiler, records, item):
    logits, keys = forward_question(model, compiler, records, item)
    from campaign_train_v1.report import expected_questions, modality
    dataset = records[item["split"]] if isinstance(records, dict) else records
    record = dataset[item["record_index"]]
    probabilities = dict(zip(keys, logits.detach().float().softmax(-1).cpu().tolist()))
    truth = expected_questions([record])[(0, item["qid"])]
    return {"record_index": item["record_index"], "qid": item["qid"], "group_id": record["group_id"],
            "modality": modality(record), "task": truth["task"], "target": truth["target"],
            "probabilities": probabilities}


def score(rows):
    from campaign_train_v1.report import metrics
    from confidence_group_v1.report import confidence_metrics
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


def point_best(output, step):
    target = output / "best"
    if target.exists() and not target.is_symlink():
        raise FileExistsError(target)
    temporary = output / ".best.tmp"
    temporary.unlink(missing_ok=True)
    temporary.symlink_to(f"step-{step:05d}")
    os.replace(temporary, target)


def train(c, source, records, split, base, resume=False):
    import torch
    from campaign_train_v2.modeling import build_model, build_compiler, optimizer_groups
    from campaign_train_v2.runtime import (Context, load_initial_checkpoint, load_checkpoint,
                                          save_checkpoint, write_json)
    from campaign_train_v2.configuration import sha256
    from .objective import configure_training, update
    context = Context("cuda")
    try:
        if context.world_size != c["world_size"]:
            raise ValueError("Launch with configured world size")
        if context.device.startswith("cuda"):
            torch.cuda.set_per_process_memory_fraction(
                c["memory_mib"] * 1024**2 / torch.cuda.get_device_properties(context.device).total_memory,
                context.device)
        output = Path(c["output"])
        manifest = {"version": VERSION, "config": c, "source": source, "split": split,
                    "source_hashes": source_hashes(), "method": "modality_output_group_supervised_binning"}
        def initialize():
            if resume:
                if json.loads((output / "run_manifest.json").read_text()) != manifest:
                    raise ValueError("Resume configuration, split, input or source changed")
                if (output / "finished.json").exists():
                    raise ValueError("Training already completed; run evaluation instead")
            else:
                output.mkdir(parents=True, exist_ok=False)
                (output / "logs").mkdir()
                write_json(output / "run_manifest.json", manifest)
                write_json(output / "split.json", split)
        context.on_primary(initialize)
        torch.manual_seed(c["seed"])
        model = build_model(dict(base, device=context.device))
        load_initial_checkpoint(model, base["init_checkpoint"])
        parameters = configure_training(model, c.get("actor_mode", "head"))
        context.broadcast(parameters)
        compiler = build_compiler(base, Path(base["media_root"]))
        groups = optimizer_groups(model, dict(base, head_lr=c["head_lr"],
                                             lora_lr=c.get("lora_lr", 5e-6),
                                             merger_lr=c.get("merger_lr", 1e-6)))
        optimizer = torch.optim.AdamW([{"params": g["params"], "lr": g["lr"]} for g in groups],
                                      weight_decay=0.0)
        training = [r for r in split["items"] if r["split"] == "train"]
        validation = [r for r in split["items"] if r["split"] == "validation"]
        # Modality-pure sampling pools: every group's bin statistics stay within one modality.
        pools = {m: [r for r in training if r["modality"] == m]
                 for m in sorted({r["modality"] for r in training})}
        if min(len(pool) for pool in pools.values()) < c["group_questions"]:
            raise ValueError("Every modality pool must fill a whole calibration group")
        checkpoint_config = dict(base, cora=c, device=context.device)
        if resume:
            progress = load_checkpoint(model, output / "latest")
            state = torch.load(output / "latest/training_state.pt", map_location="cpu", weights_only=False)
            if state["progress"] != progress:
                raise ValueError("Resume optimizer/checkpoint mismatch")
            optimizer.load_state_dict(state["optimizer"])
        else:
            baseline = validate(model, compiler, records, validation, context)
            progress = {"step": 0, "best_step": 0, "best_brier": baseline["binary"]["correctness_brier"]}
            context.on_primary(lambda: write_json(output / "logs/validation-00000.json", baseline))
            context.on_primary(lambda: save_checkpoint(output / "step-00000", model,
                                                       checkpoint_config, progress, optimizer, []))
            context.on_primary(lambda: point_best(output, 0))
        for step in range(progress["step"] + 1, c["max_steps"] + 1):
            # Randomness selects a modality (by its training-question share) and
            # then distinct questions of that modality, never confidence actions.
            rng = random.Random(c["seed"] + step)
            names = sorted(pools)
            chosen = rng.choices(names, weights=[len(pools[m]) for m in names])[0]
            batch = rng.sample(pools[chosen], c["group_questions"])
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
            "selected_checkpoint_sha256": sha256(output / "best/checkpoint.pt"),
            "validation_brier": progress["best_brier"], "gradient_training": True,
            "trainable_parameters": sum(p.numel() for p in parameters),
            "test_used_for_selection": False}))
    finally:
        context.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--check-data", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    c, source = load_config(args.config)
    records, split, base = prepare_inputs(c, source)
    if args.check_data:
        print(json.dumps({"status": "validated_only", **split["audit"],
                          "method": "modality_output_group_supervised_binning",
                          "trainable_scope": c["actor_mode"], "targets": "recomputed each modality-pure training output group",
                          "source_hashes": source_hashes()}, indent=2))
        return
    train(c, source, records, split, base, args.resume)


if __name__ == "__main__":
    main()
