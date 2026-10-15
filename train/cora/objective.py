"""Current-group confidence gradients, with recomputation to bound activation memory."""
import math
import torch
from .targets import group_targets


def loss(logits, target):
    probabilities = logits.float().softmax(-1)
    confidence = probabilities[probabilities.detach().argmax()]
    target = confidence.new_tensor(target).detach()
    if not 0 <= float(target) <= 1:
        raise ValueError("Target outside [0,1]")
    result = (confidence - target).square()
    if not torch.isfinite(result):
        raise FloatingPointError("Nonfinite CORA loss")
    return result


def update(model, optimizer, batch, forward, context, config):
    """At fixed parameters: collect the group, bin globally, recompute/backprop, step.

    No whole-group backbone graphs are retained. Parameters, dropout mode and
    candidate order are held fixed across both passes; this preserves the loss.
    """
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer.zero_grad(set_to_none=True)
    local = batch[context.rank::context.world_size]
    observations, first = [], {}
    with torch.no_grad():
        for item in local:
            logits, keys = forward(item)
            if len(keys) != len(set(keys)) or len(keys) != logits.numel():
                raise ValueError("Invalid candidate identities")
            p = logits.float().softmax(-1)
            index = int(p.argmax())
            key = (item["record_index"], item["qid"])
            row = {"key": key, "confidence": float(p[index]),
                   "correct": int(keys[index] == item["gold_answer"])}
            observations.append(row)
            first[key] = (list(keys), index, row["confidence"])
    whole_group = [row for shard in context.gather(observations) for row in shard]
    targets, fitted = group_targets(whole_group, config["bins"], config["min_bin_count"])
    supported = sum(value is not None for value in targets.values())
    if supported == 0:
        raise ValueError("No supported bins; increase output-group size or lower minimum support")
    local_loss = 0.0
    for item in local:
        key = (item["record_index"], item["qid"])
        if targets[key] is None:
            continue
        logits, keys = forward(item)
        p = logits.detach().float().softmax(-1)
        index = int(p.argmax())
        expected_keys, expected_index, expected_confidence = first[key]
        if list(keys) != expected_keys or index != expected_index or not math.isclose(
                float(p[index]), expected_confidence, rel_tol=1e-4, abs_tol=1e-5):
            raise ValueError("Group/recomputed prediction differs; inputs and model must stay deterministic")
        value = loss(logits, targets[key])
        value.backward()
        local_loss += float(value.detach())
    context.mean_gradients(parameters, supported)
    norm = torch.nn.utils.clip_grad_norm_(parameters, config["grad_clip"], error_if_nonfinite=not config["skip_nonfinite_update"])
    stats = {"loss": context.sum(local_loss) / supported, "questions": len(whole_group),
             "supported_questions": supported, "support_fraction": supported / len(whole_group),
             "group_accuracy_before_update": sum(r["correct"] for r in whole_group) / len(whole_group),
             "gradient_norm": float(norm), "bins": fitted["groups"]}
    if torch.isfinite(norm):
        optimizer.step()
    else:
        # Rare bf16 overflow in one question's backward. Gradients are already
        # all-reduced, so every rank sees the same non-finite norm and skips the
        # same update; parameters stay unchanged and training continues.
        optimizer.zero_grad(set_to_none=True)
        stats.update(gradient_norm=-1.0, skipped_nonfinite_update=1)
    return stats
