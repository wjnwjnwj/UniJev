"""Detached, cross-question bin statistics and label-free confidence lookup."""
import bisect
import math


def bin_index(confidence, bins=10):
    if isinstance(bins, bool) or not isinstance(bins, int) or bins < 1:
        raise ValueError("bins must be a positive integer")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise ValueError("confidence must be numeric")
    if not math.isfinite(confidence) or not 0 <= confidence <= 1:
        raise ValueError("confidence must be in [0, 1]")
    # [left, right); exactly 1 belongs to the final bin.
    return min(bins - 1, bisect.bisect_right([i / bins for i in range(1, bins)], confidence))


def fit_bins(observations, bins=10, min_count=20):
    """Fit with calibration data only; each distinct question contributes once.

    observations: iterable of {key, confidence, correct}. This key is the
    question identity, never an individual repeated sample of the same question.
    Sparse bins retain the original confidence at application time.
    """
    bin_index(0., bins)
    if isinstance(min_count, bool) or not isinstance(min_count, int) or min_count < 1:
        raise ValueError("min_count must be a positive integer")
    groups = [[] for _ in range(bins)]
    seen = set()
    for item in observations:
        key = item["key"]
        if key in seen:
            raise ValueError("Repeated question in calibration group: " + str(key))
        seen.add(key)
        correct = item["correct"]
        if not isinstance(correct, (int, float, bool)) or correct not in (0, 1):
            raise ValueError("correct must be binary")
        groups[bin_index(item["confidence"], bins)].append(item)
    if not seen:
        raise ValueError("No calibration questions")
    fitted = []
    for index, rows in enumerate(groups):
        count = len(rows)
        successes = sum(int(row["correct"]) for row in rows)
        fitted.append({
            "index": index, "lower": index / bins, "upper": (index + 1) / bins,
            "count": count, "successes": successes,
            "mean_confidence": sum(row["confidence"] for row in rows) / count if count else None,
            "empirical_accuracy": successes / count if count else None,
            "target": successes / count if count >= min_count else None,
            "status": "fitted" if count >= min_count else "identity_fallback",
        })
    return {"bins": bins, "min_count": min_count, "questions": len(seen),
            "confidence_semantics": "probability_the_original_selected_answer_is_correct",
            "intervals": "left_closed_right_open_last_includes_one", "groups": fitted}


def calibrate(confidence, fitted):
    """Inference reads only confidence and fitted statistics, never test labels."""
    index = bin_index(confidence, fitted["bins"])
    target = fitted["groups"][index]["target"]
    if target is None:
        return confidence
    bin_index(target, fitted["bins"])  # Validate persisted scalar.
    return target
