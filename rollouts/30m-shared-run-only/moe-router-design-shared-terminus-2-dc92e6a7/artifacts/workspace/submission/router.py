# Capacity-aware token-choice router.
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from moe_api import RouteSpec, register, assign_slots


@register("cap_aware")
class CapAwareRouter(nn.Module):
    """Top-4 token-choice with the Switch load-balancing loss and ST-MoE
    z-loss (the baseline recipe), plus a capacity-aware gate correction.

    Scoring happens at a deployment capacity factor (0.75) far below the
    training floor (1.0), where a sizeable share of each token's choices
    overflow their expert's slots and are dropped; the baseline lets that gate
    mass vanish, so an overflowing token's output shrinks. This router
    predicts, causally and at the capacity factor of the current call, which
    of each token's choices actually fit (moe_api.assign_slots queues choices
    in position order, which is exactly what the harness enforces), and
    renormalizes the gates of tokens that overflow so their computed choices
    carry the token's full top-k mass.

    Training dynamics are unchanged: at the training capacity factor the aux
    loss keeps overflow rare, the correction almost never fires, and the gate
    values and gradients equal the baseline's (the correction factor is fully
    detached; it is a per-token scale, not a routing gradient). Routing stays a
    pure causal function of positions <= t in every mode: assign_slots uses
    only earlier positions, and batch statistics feed only the aux loss.
    """

    def __init__(self, cfg, aux_coef: float = 1e-2, z_coef: float = 1e-3,
                 noise_std: float = 0.0):
        super().__init__()
        self.w_gate = nn.Linear(cfg.n_embd, cfg.n_expert, bias=False)
        nn.init.normal_(self.w_gate.weight, mean=0.0, std=0.02)
        self.aux_coef = aux_coef
        self.z_coef = z_coef
        self.noise_std = noise_std

    def forward(self, x: torch.Tensor, spec: RouteSpec):
        n_exp = spec.n_expert
        logits = self.w_gate(x.float())                             # (G, S, E)
        if self.training and self.noise_std > 0:
            logits = logits + torch.randn_like(logits) * self.noise_std

        probs = F.softmax(logits, dim=-1)
        topk_p, topk_i = probs.topk(spec.top_k, dim=-1)             # (G, S, k)

        # Switch load-balancing loss per routing group (loss only: batch
        # statistics never feed a routing decision).
        frac_tokens = F.one_hot(topk_i, n_exp).float().sum(dim=-2).mean(dim=1) / spec.top_k
        frac_prob = probs.mean(dim=1)
        aux = self.aux_coef * n_exp * (frac_tokens * frac_prob).sum(dim=-1).mean()
        z = self.z_coef * (torch.logsumexp(logits, dim=-1) ** 2).mean()

        # Baseline gates: softmax mass normalized over the k choices.
        base = topk_p.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        gate = topk_p / base

        # Capacity-aware correction (detached): where choices overflow at this
        # call's capacity, rescale the surviving choices to the full mass.
        slot = assign_slots(topk_i, None, n_exp, spec.capacity)
        fits = (slot >= 0).to(topk_p.dtype).detach()
        fit_mass = (gate * fits).sum(dim=-1, keepdim=True).detach()
        over = (fit_mass < 1.0 - 1e-4).to(gate.dtype)
        gate = gate * torch.where(over > 0, (1.0 / fit_mass.clamp_min(1e-6)),
                                  torch.ones_like(fit_mass))
        return topk_i, gate, aux + z, None
