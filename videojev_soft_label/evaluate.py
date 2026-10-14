"""Evaluate a checkpoint: accuracy / NLL / Brier plus per-question latency.

Usage:
  python -m videojev.evaluate --checkpoint output/sft_video_smoke_2b/latest \
      --data data/video_mc_6cases.jsonl --output output/eval_smoke
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from .schema import read_jsonl
from .modeling import build_model, build_compiler, load_trainable_state


def question_metrics(question, logits):
    probabilities = logits.float().softmax(-1)
    target = torch.tensor(question.target, dtype=torch.float32, device=logits.device)
    log_probs = torch.log_softmax(logits.float(), -1)
    return {
        "accuracy": float(int(probabilities.argmax()) == int(target.argmax())),
        "nll": float(-(target * log_probs).sum()),
        "brier": float(((probabilities - target) ** 2).sum()),
        "confidence": float(probabilities.max()),
    }


def summarize(rows):
    summary = {"questions": len(rows)}
    for name in ("accuracy", "nll", "brier", "confidence"):
        summary[name] = sum(row["metrics"][name] for row in rows) / len(rows)
    for field in ("modality", "domain"):
        buckets = {}
        for row in rows:
            buckets.setdefault(row[field], []).append(row["metrics"]["accuracy"])
        summary[f"accuracy_by_{field}"] = {
            key: {"accuracy": sum(values) / len(values), "questions": len(values)}
            for key, values in sorted(buckets.items())}
    return summary


@torch.no_grad()
def run(checkpoint, data, output, device="cuda"):
    output, data, checkpoint = Path(output), Path(data), Path(checkpoint)
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    config.update(device=device, gradient_checkpointing=False)
    torch.manual_seed(config.get("seed", 42))
    model = build_model(config)
    payload = torch.load(checkpoint / "checkpoint.pt", map_location="cpu", weights_only=True)
    load_trainable_state(model, payload["trainable"])
    model.eval()
    compiler = build_compiler(config, data.parent)
    records = read_jsonl(data)

    def synchronize():
        if device.startswith("cuda"):
            torch.cuda.synchronize()

    rows, started = [], time.monotonic()
    with (output / "predictions.jsonl").open("w", encoding="utf-8") as stream:
        for index, record in enumerate(records):
            compile_started = time.monotonic()
            compiled = compiler.compile(record, labeled_only=True)
            compile_seconds = time.monotonic() - compile_started
            for question in compiled.questions:
                synchronize()
                forward_started = time.monotonic()
                logits = model(question)
                synchronize()
                forward_seconds = time.monotonic() - forward_started
                meta = record.get("meta", {})
                probabilities = logits.float().softmax(-1).cpu().tolist()
                row = {"record_index": index, "record_id": meta.get("record_id"),
                       "group_id": record["group_id"], "qid": question.qid, "task": question.kind,
                       "modality": compiled.modality,
                       "domain": meta.get("domain", "unknown"),
                       "reasoning_effort": meta.get("reasoning_effort"),
                       "timing": {"compile_seconds": compile_seconds, "forward_seconds": forward_seconds,
                                  "compile_plus_forward_seconds": compile_seconds + forward_seconds},
                       "metrics": question_metrics(question, logits)}
                rows.append(row)
                prediction = dict(row, target=dict(zip(question.keys, question.target)),
                                  probabilities=dict(zip(question.keys, probabilities)),
                                  answer=question.keys[int(logits.argmax())])
                stream.write(json.dumps(prediction, ensure_ascii=False) + "\n")
            if (index + 1) % 20 == 0 or index + 1 == len(records):
                print(json.dumps({"event": "evaluation_progress", "records": index + 1,
                                  "total": len(records),
                                  "elapsed_seconds": time.monotonic() - started}), flush=True)
    if not rows:
        raise ValueError("No labeled questions evaluated")
    report = {"checkpoint": str(checkpoint), "training_step": payload["progress"]["step"],
              "model_path": config["model_path"], "seed": config.get("seed", 42),
              "data": str(data), "data_sha256": hashlib.sha256(data.read_bytes()).hexdigest(),
              "records": len(records), "questions": len(rows),
              "elapsed_seconds": time.monotonic() - started,
              "device": torch.cuda.get_device_name() if device.startswith("cuda") else "cpu",
              "metrics": summarize(rows), "latency_seconds": {}}
    for name in ("compile_seconds", "forward_seconds", "compile_plus_forward_seconds"):
        values = torch.tensor([row["timing"][name] for row in rows], dtype=torch.float64)
        report["latency_seconds"][name] = {"mean": float(values.mean()), "p50": float(values.quantile(.5)),
                                           "p95": float(values.quantile(.95))}
    (output / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"metrics": report["metrics"],
                      "latency_seconds": report["latency_seconds"]}, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    run(args.checkpoint, args.data, args.output, args.device)


if __name__ == "__main__":
    main()
