"""Prediction-row persistence helpers shared with the CORA evaluation loop."""
import json
import time
from pathlib import Path


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def resume_rows(path):
    path = Path(path)
    if not path.exists():
        return []
    raw = path.read_bytes()
    end = raw.rfind(b"\n") + 1
    if end != len(raw):
        path.with_name(path.name + ".torn-" + str(time.time_ns())).write_bytes(raw[end:])
        path.write_bytes(raw[:end])
    rows = read_rows(path)
    keys = [(r["record_index"], r["qid"]) for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate resumed question")
    return rows
