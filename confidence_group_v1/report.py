"""Correctness-confidence metrics, separate from multiclass probability metrics."""
import itertools
import math
from .grouping import bin_index


def confidence_metrics(rows, field, bins=15):
    pairs = [(row[field], int(row["metrics"]["accuracy"])) for row in rows]
    if not pairs:
        raise ValueError("No predictions")
    groups = [[] for _ in range(bins)]
    for c, hit in pairs:
        groups[bin_index(c, bins)].append((c, hit))
    reliability = []
    for index, group in enumerate(groups):
        if group:
            reliability.append({"bin": index, "count": len(group),
                                "confidence": sum(c for c, _ in group) / len(group),
                                "accuracy": sum(h for _, h in group) / len(group)})
    ece = sum(g["count"] * abs(g["confidence"] - g["accuracy"]) for g in reliability) / len(pairs)
    # Average risk over uniformly random order within tied confidence blocks.
    count, errors, previous_risk, area = 0, 0, 0., 0.
    for _, block in itertools.groupby(sorted(pairs, key=lambda x: -x[0]), key=lambda x: x[0]):
        block = list(block)
        error_fraction = sum(1 - hit for _, hit in block) / len(block)
        for j in range(1, len(block) + 1):
            risk = (errors + j * error_fraction) / (count + j)
            area += (previous_risk + risk) / (2 * len(pairs))
            previous_risk = risk
        count += len(block)
        errors += sum(1 - hit for _, hit in block)
    return {"questions": len(pairs), "acc": sum(h for _, h in pairs) / len(pairs),
            "mean_confidence": sum(c for c, _ in pairs) / len(pairs), "ece_15": ece,
            "correctness_brier": sum((c-h)**2 for c,h in pairs) / len(pairs),
            "correctness_nll": -sum(math.log(max(c if h else 1-c, 1e-12)) for c,h in pairs) / len(pairs),
            "aurc_tie_averaged": area, "reliability": reliability}
