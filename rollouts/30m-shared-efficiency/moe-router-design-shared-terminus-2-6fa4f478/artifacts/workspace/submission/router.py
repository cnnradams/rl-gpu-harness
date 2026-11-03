# Deploy-simulated token-choice router: routing decisions are always made
# against the deployment capacity factor (0.75), in training and eval alike,
# so the model trains with exactly the routing it is scored with. Choices
# that would overflow the simulated capacity are causally rerouted to the
# token's next-best experts (one replacement per token per round), and gates
# are renormalized over choices that survive, so tokens keep their full
# output mass at deployment capacity.
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from moe_api import RouteSpec, register, assign_slots, capacity_of


@register("token_choice_deploysim")
class TokenChoiceDeploySim(nn.Module):
    def __init__(self, cfg, aux_coef: float = 1e-2, z_coef: float = 1e-3,
                 n_reroute: int = 8, deploy_cf: float = 0.75):
        super().__init__()
        self.w_gate = nn.Linear(cfg.n_embd, cfg.n_expert, bias=False)
        nn.init.normal_(self.w_gate.weight, mean=0.0, std=0.02)
        self.aux_coef = float(aux_coef)
        self.z_coef = float(z_coef)
        self.n_reroute = int(n_reroute)
        self.deploy_cf = float(deploy_cf)

    def forward(self, x: torch.Tensor, spec: RouteSpec):
        E = spec.n_expert
        k = spec.top_k
        r = self.n_reroute
        cap = capacity_of(self.deploy_cf, spec.group_size, k, E)
        logits = self.w_gate(x.float())
        probs = F.softmax(logits, dim=-1)
        cand_p, cand_i = probs.topk(k + r, dim=-1)

        idx = cand_i[..., :k]
        pr = cand_p[..., :k]
        # Causal overflow rerouting against the simulated deployment capacity.
        for j in range(r):
            slot = assign_slots(idx, None, E, cap)
            dropped = slot < 0
            csum = dropped.to(torch.long).cumsum(dim=-1)
            first = dropped & (csum == 1)
            idx = torch.where(first, cand_i[..., k + j : k + j + 1], idx)
            pr = torch.where(first, cand_p[..., k + j : k + j + 1], pr)

        slot = assign_slots(idx, None, E, cap)
        keep = slot >= 0
        w = pr * keep.to(pr.dtype)
        gate = w / w.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        # Switch-style balancing loss over choices that survive capacity.
        oh = F.one_hot(idx, E).float() * keep.to(probs.dtype).unsqueeze(-1)
        frac_tokens = (oh.sum(dim=-2) / keep.to(probs.dtype).sum(dim=-1, keepdim=True).clamp_min(1.0)).mean(dim=1)
        frac_prob = probs.mean(dim=1)
        aux = self.aux_coef * E * (frac_tokens * frac_prob).sum(dim=-1).mean()
        z = self.z_coef * (torch.logsumexp(logits, dim=-1) ** 2).mean()
        return idx, gate, aux + z, keep
