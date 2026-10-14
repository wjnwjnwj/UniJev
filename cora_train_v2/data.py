"""Hash source groups of the trimodal training set into train/validation; the test suite stays separate."""
import hashlib
import json
from collections import Counter
from campaign_train_v1.report import expected_questions, modality

SPLIT_DENOMINATOR = 10**6


def request_hash(record):
    return hashlib.sha256(json.dumps(record["request"], sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def held_out(group_id, validation_fraction):
    digest = hashlib.sha256(("cora-internal-validation:" + str(group_id)).encode()).hexdigest()
    return int(digest, 16) % SPLIT_DENOMINATOR < round(validation_fraction * SPLIT_DENOMINATOR)


def assert_disjoint(datasets):
    seen_groups, seen_requests = {}, {}
    for role, records in datasets.items():
        groups = {r["group_id"] for r in records}
        requests = {request_hash(r) for r in records}
        for earlier in seen_groups:
            if groups & seen_groups[earlier]:
                raise ValueError(f"{earlier}/{role} source-group leakage")
            if requests & seen_requests[earlier]:
                raise ValueError(f"{earlier}/{role} request leakage")
        seen_groups[role], seen_requests[role] = groups, requests


def build_partitions(training, testing, validation_fraction):
    if not 0 < validation_fraction < 0.5:
        raise ValueError("Internal validation fraction must stay a small held-out share")
    assignment = {r["group_id"]: held_out(r["group_id"], validation_fraction) for r in training}
    datasets = {"train": [r for r in training if not assignment[r["group_id"]]],
                "validation": [r for r in training if assignment[r["group_id"]]],
                "test": testing}
    assert_disjoint(datasets)
    counts = {role: Counter() for role in datasets}
    record_counts = {role: Counter(modality(r) for r in records) for role, records in datasets.items()}
    groups = {role: len({r["group_id"] for r in records}) for role, records in datasets.items()}
    for record in testing:
        for (_, _), truth in expected_questions([record]).items():
            counts["test"][truth["modality"]] += 1
    items = []
    for index, record in enumerate(training):
        role = "validation" if assignment[record["group_id"]] else "train"
        for (_, qid), truth in expected_questions([record]).items():
            counts[role][truth["modality"]] += 1
            items.append({"record_index": index, "qid": qid, "group_id": truth["group_id"],
                          "split": role, "modality": truth["modality"],
                          "gold_answer": max(truth["target"], key=truth["target"].get)})
    return {"items": items, "audit": {
        "train_questions": sum(counts["train"].values()),
        "validation_questions": sum(counts["validation"].values()),
        "test_questions": sum(counts["test"].values()),
        "question_counts": {k: dict(v) for k, v in counts.items()},
        "record_counts": {k: dict(v) for k, v in record_counts.items()},
        "source_group_counts": groups, "train_dev_test_overlap": 0,
        "validation_fraction": validation_fraction,
        "roles": "train/validation are a source-group hash split of the trimodal training set; "
                 "validation only selects checkpoints; test only scores",
    }}
