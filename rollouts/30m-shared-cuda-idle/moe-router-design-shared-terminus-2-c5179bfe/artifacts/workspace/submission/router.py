from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from moe_api import RouteSpec, register


@register("dev_limited")
class DevLimitedRouter(nn.Module):
    """Top-k token-choice limited to n_dev expert-parallel devices per token.

    Device score = sum of the top t = max(1, k//n_dev) expert probs in it.
    The k choices are then taken from the selected devices' experts only,
    which bounds expert-parallel traffic at n_dev-1 extra devices per token.
    Switch per-sequence load-balancing loss + ST-MoE z-loss as in baseline.
    """

    def __init__(self, cfg, aux_coef: float = 1e-2, z_coef: float = 1e-3,
                 n_dev: int = 2):
        super().__init__()
        self.w_gate = nn.Linear(cfg.n_embd, cfg.n_expert, bias=False)
        nn.init.normal_(self.w_gate.weight, mean=0.0, std=0.02)
        self.aux_coef = aux_coef
        self.z_coef = z_coef
        self.n_dev = n_dev

    def forward(self, x: torch.Tensor, spec: RouteSpec):
        n_exp = spec.n_expert
        k = spec.top_k
        D = spec.ep_devices
        logits = self.w_gate(x.float())                         # (G, S, E)
        probs = F.softmax(logits, dim=-1)
        if D is not None and D > 1 and n_exp % D == 0 and self.n_dev < D:
            Ed = n_exp // D
            n_dev = self.n_dev if self.n_dev > 0 else 1
            t = k // n_dev
            if t < 1:
                t = 1
            p4 = probs.reshape(probs.shape[0], probs.shape[1], D, Ed)
            dev_score = p4.topk(t, dim=-1).values.sum(dim=-1)   # (G,S,D)
            sel = dev_score.topk(n_dev, dim=-1).indices         # (G,S,n_dev)
            dev_mask = torch.zeros_like(dev_score).scatter_(-1, sel, 1.0)
            mask = dev_mask.unsqueeze(-1).expand(
                probs.shape[0], probs.shape[1], D, Ed).reshape(
                probs.shape[0], probs.shape[1], n_exp)
            masked = probs * mask
        else:
            masked = probs
        topk_p, topk_i = masked.topk(k, dim=-1)                 # (G, S, k)

        frac_tokens = F.one_hot(topk_i, n_exp).float().sum(dim=-2).mean(dim=1) / k
        frac_prob = probs.mean(dim=1)
        aux = self.aux_coef * n_exp * (frac_tokens * frac_prob).sum(dim=-1).mean()
        z = self.z_coef * (torch.logsumexp(logits, dim=-1) ** 2).mean()

        gate = topk_p / topk_p.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        return topk_i, gate, aux + z, None
