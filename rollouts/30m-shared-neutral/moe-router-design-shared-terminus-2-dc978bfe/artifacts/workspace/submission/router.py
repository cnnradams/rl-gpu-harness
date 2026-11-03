"""Sigmoid-gated top-k token-choice router with aux-loss-free load balancing.

Gating: per-expert sigmoid scores; top-k selection on score + learnable-rate
bias buffer; gates are the selected sigmoid scores normalized to sum to 1.
Balancing: a persistent per-expert bias buffer is nudged by +/- bias_lr *
sign(load_error) on every training call (DeepSeek-V3 style). The bias shifts
selection pressure toward under-used experts without an auxiliary gradient
fighting the LM loss. Updates happen only when self.training is true; in eval
the router is a pure function of its input. Decisions use the bias value at
call start, and the update uses batch statistics only after decisions are made,
so prefix routing cannot depend on later positions. A small sequence-scope
Switch-style balancing loss and a small z-loss remain for stability.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from moe_api import register, RouteSpec


@register("sigmoid_bias")
class SigmoidBiasRouter(nn.Module):
    def __init__(self, cfg, bias_lr: float = 1e-3, aux_coef: float = 1e-3,
                 z_coef: float = 1e-4):
        super().__init__()
        self.w_gate = nn.Linear(cfg.n_embd, cfg.n_expert, bias=False)
        nn.init.normal_(self.w_gate.weight, mean=0.0, std=0.02)
        self.register_buffer("expert_bias", torch.zeros(cfg.n_expert))
        self.bias_lr = float(bias_lr)
        self.aux_coef = float(aux_coef)
        self.z_coef = float(z_coef)

    def forward(self, x: torch.Tensor, spec: RouteSpec):
        n_exp = spec.n_expert
        k = spec.top_k
        logits = self.w_gate(x.float())                 # (G, S, E)
        scores = torch.sigmoid(logits)
        biased = scores + self.expert_bias
        topk_v, topk_i = biased.topk(k, dim=-1)         # (G, S, k)

        g = scores.gather(-1, topk_i)
        gate = g / g.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        z = self.z_coef * (torch.logsumexp(logits, dim=-1) ** 2).mean()

        probs = scores / scores.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        frac_tokens = F.one_hot(topk_i, n_exp).float().sum(dim=-2).mean(dim=1) / k
        frac_prob = probs.mean(dim=1)
        aux = self.aux_coef * n_exp * (frac_tokens * frac_prob).sum(dim=-1).mean()

        if self.training:
            with torch.no_grad():
                counts = F.one_hot(topk_i, n_exp).float().sum(dim=(0, 1, 2))
                total = float(topk_i.shape[0] * topk_i.shape[1] * k)
                err = counts / total - 1.0 / n_exp
                self.expert_bias.add_(torch.sign(err) * (-self.bias_lr))

        return topk_i, gate, aux + z, None
