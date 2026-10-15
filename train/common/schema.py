"""Strict JSONL contract. IDs and labels never enter the model prompt.

Record shape (one line per record):
  {
    "group_id": "<video id, required for leakage-free splits>",
    "request": {
      "state": "<text>" | {"messages": [...]},   # messages may carry video_url/image_url
      "questions": {"<qid>": {"type": "choice|noul|score", "instructions": str, "criteria": ...}}
    },
    "targets": {"<qid>": {"probabilities": {key: float}}},   # optional per question
    "assets":  [{"path": str, "sha256": str}],               # optional media pinning
    "meta":    {...}                                          # free-form provenance
  }
"""
import json
import math
from pathlib import Path


def candidates(question):
    """Return the ordered [(key, description)] list for a question."""
    kind = question.get("type")
    if not isinstance(question.get("instructions"), str) or not question["instructions"].strip():
        raise ValueError("instructions must be nonempty text")
    criteria = question.get("criteria")
    if kind == "noul":
        return [("true", "True / 是：满足问题中的条件。"), ("false", "False / 否：不满足问题中的条件。")]
    if kind == "choice":
        if not isinstance(criteria, dict) or not 1 <= len(criteria) <= 255:
            raise ValueError("Choice needs 1–255 named criteria")
        result = list(criteria.items())
    elif kind == "score":
        if not isinstance(criteria, list) or not 2 <= len(criteria) <= 10:
            raise ValueError("Score criteria must be an ordered list of 2–10 descriptions")
        result = [(str(i), text) for i, text in enumerate(criteria)]
    else:
        raise ValueError(f"Unsupported question type: {kind}")
    if any(not isinstance(k, str) or not k or not isinstance(v, str) or not v.strip() for k, v in result):
        raise ValueError("Candidate names and descriptions must be nonempty strings")
    return result


def target_distribution(target, keys):
    """Validate and return the probability vector aligned with keys, or None."""
    if target is None or target.get("probabilities") is None:
        return None
    probs = target["probabilities"]
    if not isinstance(probs, dict) or set(probs) != set(keys):
        raise ValueError(f"Probability keys must exactly match {keys}")
    values = [probs[k] for k in keys]
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in values):
        raise ValueError("Probabilities must be finite nonnegative numbers")
    if not math.isclose(sum(values), 1.0, abs_tol=1e-6):
        raise ValueError("Target probabilities must sum to 1")
    return values


def validate_record(record):
    request = record["request"]
    state = request["state"]
    if not isinstance(state, (str, dict)):
        raise ValueError("state must be text or a messages object")
    if not isinstance(request["questions"], dict) or not request["questions"]:
        raise ValueError("questions must be a nonempty mapping")
    targets = record.get("targets", {})
    if set(targets) - set(request["questions"]):
        raise ValueError("Unknown target question IDs")
    for qid, question in request["questions"].items():
        target_distribution(targets.get(qid), [k for k, _ in candidates(question)])
    if not record.get("group_id"):
        raise ValueError("group_id is required for leakage-free splits")
    return record


def read_jsonl(path):
    records = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            records.append(validate_record(json.loads(line)))
        except Exception as error:
            raise ValueError(f"{path}:{number}: {error}") from error
    if not records:
        raise ValueError(f"{path}: no records")
    return records
