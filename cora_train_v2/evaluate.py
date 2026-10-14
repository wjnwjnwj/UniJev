"""Evaluate CORA's own answers/probabilities on the independent final test suite."""
import argparse
import json
from pathlib import Path
from .configuration import load_config, prepare_inputs
from .train import predict, score, source_hashes


def run(c, source, base, checkpoint, output):
    import torch
    from campaign_train_v2.modeling import build_model, build_compiler
    from campaign_train_v2.runtime import Context, load_checkpoint, write_json
    from campaign_train_v2.configuration import sha256
    from campaign_train_v1.schema import read_jsonl
    from campaign_train_v1.report import expected_questions, validate_predictions, summarize_predictions
    from confidence_group_v1.train import read_rows, resume_rows
    context = Context("cuda")
    try:
        if context.world_size != c["world_size"]:
            raise ValueError("Launch evaluation with configured world size")
        checkpoint, output = Path(checkpoint).resolve(), Path(output).resolve()
        training_root = Path(c["output"])
        if output == training_root.resolve():
            raise ValueError("Evaluation output cannot overwrite training root")
        proof = json.loads((training_root / "finished.json").read_text())
        final = (training_root / f'step-{c["max_steps"]:05d}').resolve()
        if proof["steps"] != c["max_steps"] or checkpoint != final:
            raise ValueError("Reporting requires the fixed-budget final checkpoint")
        checkpoint_sha256 = sha256(checkpoint / "checkpoint.pt")
        if json.loads((training_root / "run_manifest.json").read_text())["config"] != c:
            raise ValueError("Evaluation/training configuration mismatch")
        records = read_jsonl(source["final_eval_data"])
        expected = expected_questions(records)
        manifest = {"method": "cora_supervised_bins", "checkpoint": str(checkpoint),
                    "checkpoint_sha256": checkpoint_sha256,
                    "data_sha256": sha256(source["final_eval_data"]), "world_size": context.world_size,
                    "source_hashes": source_hashes(), "inference": "native_trained_candidate_probabilities"}
        def initialize():
            output.mkdir(parents=True, exist_ok=True)
            path = output / "run.json"
            if path.exists() and json.loads(path.read_text()) != manifest:
                raise ValueError("Evaluation resume inputs changed")
            write_json(path, manifest)
        context.on_primary(initialize)
        assigned = [key for key in expected if key[0] % context.world_size == context.rank]
        shard = output / f"predictions.rank{context.rank}.jsonl"
        old = resume_rows(shard)
        seen = {(r["record_index"], r["qid"]) for r in old}
        if not seen <= set(assigned):
            raise ValueError("Unexpected resumed prediction")
        if seen != set(assigned):
            if context.device.startswith("cuda"):
                torch.cuda.set_per_process_memory_fraction(
                    c["memory_mib"] * 1024**2 / torch.cuda.get_device_properties(context.device).total_memory,
                    context.device)
            model = build_model(dict(base, device=context.device))
            load_checkpoint(model, checkpoint)
            model.requires_grad_(False)
            model.eval()
            compiler = build_compiler(base, Path(base["media_root"]))
            with torch.inference_mode(), shard.open("a", buffering=1) as stream:
                for index, qid in assigned:
                    if (index, qid) in seen:
                        continue
                    row = predict(model, compiler, records, {"record_index": index, "qid": qid})
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
                    seen.add((index, qid))
        context.gather(len(seen))
        def finish():
            rows = [r for rank in range(context.world_size)
                    for r in read_rows(output / f"predictions.rank{rank}.jsonl")]
            clean = validate_predictions(rows, records, c_expected := source["expected_eval_counts"])
            with (output / "predictions.jsonl.tmp").open("w") as stream:
                for row in clean:
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
            (output / "predictions.jsonl.tmp").replace(output / "predictions.jsonl")
            primary = summarize_predictions(clean, records, c_expected)
            binary = {modality: score([r for r in clean if r["modality"] == modality])["binary"]
                      for modality in sorted(c_expected)}
            write_json(output / "comparison_metrics.json", {
                "method": "cora_supervised_bins", "checkpoint_sha256": checkpoint_sha256,
                "metrics": primary, "binary_confidence_metrics": binary,
                "answers_may_change": True, "posthoc_mapping": False})
            names = ["run.json", "predictions.jsonl", "comparison_metrics.json"]
            write_json(output / "finished.json", {"method": "cora_supervised_bins",
                                                 "files": {name: sha256(output / name) for name in names}})
        context.on_primary(finish)
    finally:
        context.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--output")
    args = parser.parse_args()
    c, source = load_config(args.config)
    _, _, base = prepare_inputs(c, source)
    run(c, source, base, args.checkpoint or str(Path(c["output"]) / f'step-{c["max_steps"]:05d}'),
        args.output or str(Path(c["output"]) / "evaluation"))


if __name__ == "__main__":
    main()
