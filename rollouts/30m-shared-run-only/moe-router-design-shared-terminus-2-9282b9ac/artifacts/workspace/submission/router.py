"""Capacity-aware token-choice router.

Same family as the baseline (top-k token choice, Switch load-balancing loss,
ST-MoE z-loss), but the gates are normalized over the choices that actually
survive capacity at the capacity factor of the current call, instead of over
all k choices. The router replicates the harness's causal slot assignment
(moe_api.assign_slots) on its own choices: kept gates are renormalized to sum
to 1, dropped choices get gate 0, so a token routed into full experts loses no
output mass. Scoring happens at deployment capacity factor 0.75 while training
requires >= 1.0, so this mainly changes behavior where it is scored.

The load-balancing loss is measured per routing group over all choices (as
the baseline), with the group-size-dependent normalization folded in so the
coefficient means the same thing at every capacity factor. An optional bonus
on experts of the first top-1 device's devices concentrates each token's
choices on fewer expert-parallel devices, cutting the device-traffic charge.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from moe_api import RouteSpec, register


@register("aware")
class AwareRouter(nn.Module):
    def __init__(self, cfg, aux_coef: float = 1e-2, z_coef: float = 1e-3,
                 dev_bonus: float = 0.0, renorm_train: bool = True):
        super().__init__()
        self.w_gate = nn.Linear(cfg.n_embd, cfg.n_expert, bias=False)
        nn.init.normal_(self.w_gate.weight, mean=0.0, std=0.02)
        self.aux_coef = aux_coef
        self.z_coef = z_coef
        self.dev_bonus = dev_bonus
        self.renorm_train = renorm_train

    def forward(self, x: torch.Tensor, spec: RouteSpec):
        with torch.autocast(device_type=x.device.type, enabled=False):
            return self._route(x, spec)

    def _route(self, x: torch.Tensor, spec: RouteSpec):
        n_exp, k, cap = spec.n_expert, spec.top_k, spec.capacity
        G, S, _ = x.shape
        logits = self.w_gate(x.float()).float()                             # (G, S, E)
        probs = F.softmax(logits, dim=-1)
        topk_p, topk_i = probs.topk(k, dim=-1)                      # (G, S, k)

        # Switch load-balancing loss per routing group over all choices, with
        # the uniform-baseline value folded out so aux_coef is comparable
        # across capacity factors and group sizes.
        frac_tokens = F.one_hot(topk_i, n_exp).float().sum(dim=-2).mean(dim=1) / k
        frac_prob = probs.mean(dim=1)
        base = k / (n_exp * S)
        aux = self.aux_coef * n_exp * ((frac_tokens * frac_prob).sum(dim=-1) / base - 1.0).clamp_min(0.0).mean()
        z = self.z_coef * (torch.logsumexp(logits, dim=-1) ** 2).mean()

        sel = logits
        if self.dev_bonus > 0.0:
            epd = spec.ep_devices
            if epd > 0 and k > 1:
                exp_per_dev = n_exp // epd
                dev1 = topk_i[..., 0] // exp_per_dev                # (G, S)
                devs = torch.arange(n_exp, device=x.device) // exp_per_dev
                bonus = (devs.unsqueeze(0).unsqueeze(0) == dev1.unsqueeze(-1)).float() * self.dev_bonus
                sel = logits + bonus

        if self.renorm_train or not self.training:
            # Replicate the harness's causal capacity assignment on the selected
            # experts, then normalize gates over the surviving choices only.
            flat = sel.topk(k, dim=-1).indices                      # (G, S, k)
            oh = F.one_hot(flat.reshape(G, S * k), n_exp).float()
            rank = (oh.cumsum(dim=1) - oh).gather(2, flat.reshape(G, S * k).unsqueeze(-1)).squeeze(-1)
            kept = (rank < cap).reshape(G, S, k)
            kept_score = torch.where(kept, sel.gather(-1, flat),
                                     torch.full_like(sel.gather(-1, flat), -1e9)).float()
            gate = F.softmax(kept_score, dim=-1)
            return flat, gate, aux + z, None

        topk_p2, topk_i2 = sel.topk(k, dim=-1)
        gate = topk_p2 / topk_p2.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        return topk_i2, gate, aux + z, None
