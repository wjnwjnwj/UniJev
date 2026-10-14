"""Soft-label SFT objective. Other campaign recipes are not part of this release."""
import math
import torch


def compute(model, anchor, compiler, record, question, config, step):
    if config['objective'] != 'sft_soft':
        raise ValueError('This release ships only the soft-label SFT objective (sft_soft)')
    z = model(question)
    k = len(question.target)
    q = torch.full_like(z.float(), .1/(k-1))
    q[int(torch.tensor(question.target).argmax())] = .9
    loss = -(q*z.float().log_softmax(-1)).sum()
    stats = {'reward': -float(loss.detach())}
    stats = {key: value for key, value in stats.items() if math.isfinite(value)}
    stats['extra_forward_tokens'] = 0
    return loss, stats
