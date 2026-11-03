from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from moe_api import register, RouteSpec


@register("sig_bfree")
class SigmoidBiasFreeRouter(nn.Module):
    """Token-choice top-k routing with sigmoid gates and an aux-loss-free
    load-balancing bias (DeepSeek-V3 style). A per-expert bias buffer shifts
    scores only for expert selection; gates use raw sigmoid scores. The bias
    is updated in-place during training with sign(mean_load - load) * bias_lr.
    A small optional sequence-scope Switch aux loss and ST-MoE z-loss remain.
    """

    def __init__(self, cfg, aux_coef: float = 3e-3, z_coef: float = 1e-3,
                 bias_lr: float = 1e-3):
        super().__init__()
        self.w_gate = nn.Linear(cfg.n_embd, cfg.n_expert, bias=False)
        nn.init.normal_(self.w_gate.weight, mean=0.0, std=0.02)
        self.register_buffer("e_bias", torch.zeros(cfg.n_expert))
        self.aux_coef = aux_coef
        self.z_coef = z_coef
        self.bias_lr = bias_lr

    def forward(self, x: torch.Tensor, spec: RouteSpec):
        E = spec.n_expert
        k = spec.top_k
        logits = self.w_gate(x.float())                     # (G, S, E)
        scores = torch.sigmoid(logits)
        routed = scores + self.e_bias                       # selection only
        topk_s, topk_i = routed.topk(k, dim=-1)             # (G, S, k)

        gate_vals = scores.gather(2, topk_i)
        gate = gate_vals / gate_vals.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        aux = scores.new_zeros(())
        if self.aux_coef > 0.0:
            frac_tokens = F.one_hot(topk_i, E).float().sum(dim=-2).mean(dim=1) / k
            frac_prob = scores.mean(dim=1) / scores.mean(dim=1).sum(dim=-1, keepdim=True).clamp_min(1e-9)
            aux = self.aux_coef * E * (frac_tokens * frac_prob).sum(dim=-1).mean()
        z = self.z_coef * (torch.logsumexp(logits, dim=-1) ** 2).mean()

        if self.training:
            with torch.no_grad():
                oh = F.one_hot(topk_i, E).float().sum(dim=(0, 1, 2))
                err = oh.mean() - oh.detach()
                self.e_bias.add_(torch.sign(err) * self.bias_lr)
        return topk_i, gate, aux + z, None
