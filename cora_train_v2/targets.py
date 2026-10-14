"""Detached bin targets from CURRENT training-group outputs.

v2 calls this on modality-pure groups only, so every bin statistic is
estimated within a single modality; the binning itself is unchanged.
"""
from confidence_group_v1.grouping import fit_bins, bin_index


def group_targets(observations, bins=10, min_count=20):
    """Fresh statistics of this training forward group, never a persistent lookup."""
    observations = list(observations)
    fitted = fit_bins(observations, bins, min_count)
    targets = {}
    for row in observations:
        group = fitted["groups"][bin_index(row["confidence"], bins)]
        targets[row["key"]] = group["empirical_accuracy"] if group["count"] >= min_count else None
    return targets, fitted
