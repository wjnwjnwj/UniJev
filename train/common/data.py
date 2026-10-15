"""Training-question identities and the original source-group holdout split."""
import hashlib
import json
from collections import Counter
from .schema import candidates, target_distribution

def modality(record):
    state = record['request']['state']
    kinds = set()
    if isinstance(state, dict):
        for message in state['messages']:
            if isinstance(message['content'], list):
                kinds.update(item['type'] for item in message['content'] if item['type'] in {'image_url', 'video_url'})
    return 'mixed' if len(kinds) > 1 else 'image' if 'image_url' in kinds else 'video' if 'video_url' in kinds else 'text'


def expected_questions(records):
    expected = {}
    for index, record in enumerate(records):
        for qid, question in record['request']['questions'].items():
            keys = [k for k, _ in candidates(question)]
            target = target_distribution(record.get('targets', {}).get(qid), keys)
            if target is not None:
                if target.count(1.0) != 1 or any(v not in (0, 1) for v in target):
                    raise ValueError('CORA training requires one-hot targets')
                expected[(index, qid)] = {'group_id': record['group_id'], 'modality': modality(record),
                                          'target': dict(zip(keys, target)), 'task': question['type']}
    if not expected:
        raise ValueError('No labeled questions')
    return expected


def held_out(group_id, validation_fraction=0.01):
    digest = hashlib.sha256(("cora-internal-validation:" + str(group_id)).encode()).hexdigest()
    return int(digest, 16) % 10**6 < round(validation_fraction * 10**6)


def build_partitions(training, validation_fraction=0.01):
    assignment = {r["group_id"]: held_out(r["group_id"], validation_fraction) for r in training}
    seen = {False: set(), True: set()}
    for record in training:
        request = json.dumps(record["request"], sort_keys=True, ensure_ascii=False)
        seen[assignment[record["group_id"]]].add(hashlib.sha256(request.encode()).hexdigest())
    if seen[False] & seen[True]:
        raise ValueError("Train/validation request leakage")
    items, counts = [], {"train": Counter(), "validation": Counter()}
    for index, record in enumerate(training):
        role = "validation" if assignment[record["group_id"]] else "train"
        for (_, qid), truth in expected_questions([record]).items():
            counts[role][truth["modality"]] += 1
            items.append({"record_index": index, "qid": qid, "group_id": truth["group_id"],
                          "split": role, "modality": truth["modality"],
                          "gold_answer": max(truth["target"], key=truth["target"].get)})
    if set(counts["train"]) != {"text", "image", "video"} or set(counts["validation"]) != set(counts["train"]):
        raise ValueError("Train and internal holdout must each cover text, image and video")
    if sum(counts["validation"].values()) >= sum(counts["train"].values()):
        raise ValueError("Internal holdout must be the smaller partition")
    return {"items": items, "audit": {"question_counts": {k: dict(v) for k, v in counts.items()},
            "train_questions": sum(counts["train"].values()),
            "validation_questions": sum(counts["validation"].values()),
            "validation_fraction": validation_fraction}}


def check_data(config):
    from pathlib import Path
    from .schema import read_jsonl
    from .config import sha256
    records = read_jsonl(config["data"])
    questions = expected_questions(records)
    if any(len(q["target"]) < 2 for q in questions.values()):
        raise ValueError("Training requires at least two candidates per question")
    paths = set()
    root = Path(config["media_root"])
    for record in records:
        state = record["request"]["state"]
        for message in state.get("messages", []) if isinstance(state, dict) else []:
            content = message["content"]
            for item in content if isinstance(content, list) else []:
                kind = item["type"]
                if kind in {"image_url", "video_url"}:
                    value = item[kind]["url"]
                    if "://" in value:
                        raise ValueError("Materialize media locally: " + value)
                    path = Path(value)
                    path = path if path.is_absolute() else root / path
                    if not path.is_file():
                        raise FileNotFoundError(path)
                    paths.add(str(path.resolve()))
        for asset in record.get("assets", []):
            path = Path(asset["path"])
            path = path if path.is_absolute() else root / path
            if sha256(path) != asset["sha256"]:
                raise ValueError("Media hash mismatch: " + str(path))
    audit = {"records": len(records), "questions": len(questions), "media_files": len(paths),
             "records_by_modality": dict(Counter(modality(r) for r in records)),
             "questions_by_modality": dict(Counter(q["modality"] for q in questions.values())),
             "data_sha256": sha256(config["data"])}
    return records, audit

