"""Aux-loss-free load-balanced router (DeepSeek-V3 style).

Token-choice top-k routing with sigmoid scores. A per-expert bias buffer is
added to scores for selection only (never for gating) and updated each
training step by the sign of the per-expert load error, pushing load toward
uniform without any balancing gradient through the LM loss. A small ST-MoE
z-loss keeps router logits bounded. Balance is maintained by the bias, so at
the tight deployment capacity factor overflow drops stay rare.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from moe_api import RouteSpec, register


@register("bias_balance")
class BiasBalanceRouter(nn.Module):
    def __init__(self, cfg, z_coef: float = 1e-3, bias_lr: float = 1e-3):
        super().__init__()
        self.w_gate = nn.Linear(cfg.n_embd, cfg.n_expert, bias=False)
        nn.init.normal_(self.w_gate.weight, mean=0.0, std=0.02)
        self.register_buffer("expert_bias", torch.zeros(cfg.n_expert), persistent=True)
        self.z_coef = z_coef
        self.bias_lr = bias_lr

    def forward(self, x: torch.Tensor, spec: RouteSpec):
        n_exp = spec.n_expert
        logits = self.w_gate(x.float())                            # (G, S, E)
        scores = torch.sigmoid(logits)
        sel = scores + self.expert_bias                            # selection only
        topk_s, topk_i = sel.topk(spec.top_k, dim=-1)              # (G, S, k)

        # Gates from unbiased sigmoid scores of the chosen experts, normalized
        # over the k choices so a fully-routed token's weights sum to 1.
        topk_p = scores.gather(-1, topk_i)
        gate = topk_p / topk_p.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        aux = self.z_coef * (torch.logsumexp(logits, dim=-1) ** 2).mean()

        if self.training:
            idx = topk_i.detach()
            counts = F.one_hot(idx, n_exp).float().sum(dim=-2).mean(dim=(0, 1)) / spec.top_k
            err = counts - (1.0 / n_exp)
            self.expert_bias.add_(-torch.sign(err) * self.bias_lr)

        return topk_i, gate, aux, None
