# Baseline Switch routing + overflow-aware keep mask for tight deployment capacity.
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from moe_api import RouteSpec, register, assign_slots


@register("keep_drop")
class KeepDropRouter(nn.Module):
    """Top-k token-choice with Switch load-balancing loss and ST-MoE z-loss
    (the published baseline recipe), plus an overflow-aware keep mask.

    At the deployment capacity factor (0.75), roughly a quarter of choices
    overflow and are dropped by the harness. A choice dropped via the keep mask
    never enters its expert's queue, so the slot it frees lets a later choice
    of the same expert (which would itself have overflowed) be served. We drop,
    per token, the lowest-gate-share overflowing choice, and any overflowing
    choice whose share is below uniform (1/k), while guaranteeing each token
    keeps at least `min_keep` choices. All decisions use only the token's own
    choices and the causal per-group prefix ranks, so causality is preserved.
    Loss terms are batch statistics and never affect routing decisions.
    """

    def __init__(self, cfg, aux_coef: float = 1e-2, z_coef: float = 1e-3,
                 noise_std: float = 0.0, min_keep: int = 2):
        super().__init__()
        self.w_gate = nn.Linear(cfg.n_embd, cfg.n_expert, bias=False)
        nn.init.normal_(self.w_gate.weight, mean=0.0, std=0.02)
        self.aux_coef = aux_coef
        self.z_coef = z_coef
        self.noise_std = noise_std
        self.min_keep = min_keep

    def forward(self, x: torch.Tensor, spec: RouteSpec):
        n_exp = spec.n_expert
        k = spec.top_k
        logits = self.w_gate(x.float())
        if self.training and self.noise_std > 0:
            logits = logits + torch.randn_like(logits) * self.noise_std

        probs = F.softmax(logits, dim=-1)
        topk_p, topk_i = probs.topk(k, dim=-1)                # (G,S,k)

        keep = None
        if spec.capacity * n_exp < spec.group_size * k and k > self.min_keep:
            slot = assign_slots(topk_i, None, n_exp, spec.capacity)
            overflow = slot < 0                               # (G,S,k)
            share = topk_p / topk_p.sum(dim=-1, keepdim=True).clamp_min(1e-9)
            share_ov = torch.where(overflow, share, torch.full_like(share, 2.0))
            min_share = share_ov.min(dim=-1, keepdim=True).values
            n_ov = overflow.sum(dim=-1, keepdim=True)
            can_keep = n_ov <= (k - self.min_keep)
            drop = ((share_ov <= min_share) | (share < (1.0 / k))) & overflow & can_keep
            keep = ~drop

        # Switch load-balancing loss, per routing group (sequence).
        dims = 1
        frac_tokens = F.one_hot(topk_i, n_exp).float().sum(dim=-2).mean(dim=dims) / k
        frac_prob = probs.mean(dim=dims)
        aux = self.aux_coef * n_exp * (frac_tokens * frac_prob).sum(dim=-1).mean()
        z = self.z_coef * (torch.logsumexp(logits, dim=-1) ** 2).mean()

        gate = topk_p / topk_p.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        return topk_i, gate, aux + z, keep
