from __future__ import annotations

import torch
import torch.nn.functional as F


def pairwise_ranking_loss(input1, input2, label1, label2, fn: str = "hinge", margin: float = 1.0):
    target = torch.where(label1 > label2, 1.0, -1.0)
    if fn == "hinge":
        return F.margin_ranking_loss(input1, input2, target, margin=margin)
    if fn == "exp":
        return torch.exp(-target * (input1 - input2)).mean()
    if fn == "log":
        return torch.log1p(torch.exp(-target * (input1 - input2))).mean()
    raise ValueError(f"Unknown pairwise ranking function: {fn}")


def listwise_ranking_loss(predicts, targets):
    indices = targets.sort(descending=True, dim=-1).indices
    predicts = torch.gather(predicts, dim=1, index=indices)
    cumsums = predicts.exp().flip(dims=[1]).cumsum(dim=1).flip(dims=[1])
    return (torch.log(cumsums + 1e-10) - predicts).sum(dim=1).mean()
