"""CORA's per-question loss: squared gap between selected-answer confidence and its bin target."""


def loss(logits, target):
    import torch
    probabilities = logits.float().softmax(-1)
    confidence = probabilities[probabilities.detach().argmax()]
    target = confidence.new_tensor(target).detach()
    if not 0 <= float(target) <= 1:
        raise ValueError("Target outside [0,1]")
    result = (confidence - target).square()
    if not torch.isfinite(result):
        raise FloatingPointError("Nonfinite CORA loss")
    return result
