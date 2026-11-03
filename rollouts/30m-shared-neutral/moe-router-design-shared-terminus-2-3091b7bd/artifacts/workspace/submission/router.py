"""Aux-loss-free balanced router with causal overflow rerouting.

Scores: sigmoid(W x). Selection: score + per-expert bias (persistent buffer,
updated only in training by sign(target_share - observed_share) -- DeepSeek-V3
style aux-loss-free balancing). Gates: unbiased sigmoid scores of the chosen
experts, normalized over the top-k choices. Aux loss: z-loss only.

Overflow rerouting: choices that would overflow their expert's capacity
(predicted causally with moe_api.assign_slots) are replaced by the token's
best backup expert whose prefix count (earlier primary choices in the same
group, causal) still leaves a free slot. This recovers dropped-token signal at
the tight deployment capacity factor.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from moe_api import RouteSpec, assign_slots, register


@register("bias_balance")
class BiasBalanceRouter(nn.Module):
    def __init__(self, cfg, z_coef: float = 1e-3, bias_lr: float = 1e-3,
                 reroute: bool = True):
        super().__init__()
        self.w_gate = nn.Linear(cfg.n_embd, cfg.n_expert, bias=False)
        nn.init.normal_(self.w_gate.weight, mean=0.0, std=0.02)
        self.register_buffer("bias", torch.zeros(cfg.n_expert))
        self.z_coef = z_coef
        self.bias_lr = bias_lr
        self.reroute = reroute

    def forward(self, x: torch.Tensor, spec: RouteSpec):
        n_exp = spec.n_expert
        k = spec.top_k
        G, S, _ = x.shape
        logits = self.w_gate(x.float())                       # (G, S, E)
        scores = torch.sigmoid(logits)
        biased = scores + self.bias.unsqueeze(0).unsqueeze(0)
        cand_s, cand_i = biased.topk(2 * k, dim=-1)           # (G, S, 2k)
        topk_i = cand_i[..., :k].contiguous()                 # (G, S, k)

        if self.reroute:
            slot = assign_slots(topk_i, None, n_exp, spec.capacity)
            dropped = slot < 0                                # (G, S, k)
            oh = F.one_hot(topk_i.reshape(G, S * k), n_exp).to(scores.dtype)
            prefix = (oh.cumsum(dim=1) - oh).reshape(G, S, k, n_exp)
            cnt_before = prefix[:, :, 0]                      # (G, S, E)
            backup_i = cand_i[..., k:]                        # (G, S, k)
            final_i = topk_i.clone()
            used = torch.zeros(G, S, n_exp, dtype=torch.bool, device=x.device)
            for j in range(k):
                idx_j = topk_i[..., j]                        # (G, S)
                drop_j = dropped[..., j]                      # (G, S)
                used_j = used.gather(-1, idx_j.unsqueeze(-1)).squeeze(-1)
                drop_j = drop_j | used_j
                free = cnt_before < spec.capacity             # (G, S, E)
                cand_ok = free.gather(-1, backup_i) & ~used.gather(-1, backup_i)
                first_ok = cand_ok.to(torch.int64).argmax(dim=-1, keepdim=True)
                any_ok = cand_ok.any(dim=-1)
                chosen_b = backup_i.gather(-1, first_ok).squeeze(-1)
                new_j = torch.where(drop_j & any_ok, chosen_b, idx_j)
                final_i[..., j] = new_j
                used = used | F.one_hot(new_j, n_exp).to(torch.bool)
        else:
            final_i = topk_i

        gates = scores.gather(-1, final_i)
        gate = gates / gates.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        z = self.z_coef * (torch.logsumexp(logits, dim=-1) ** 2).mean()

        if self.training:
            fi = F.one_hot(final_i, n_exp).to(scores.dtype).sum(dim=-2).mean(dim=(0, 1)) / k
            target = torch.full_like(fi, 1.0 / n_exp)
            self.bias.add_(torch.sign(target - fi) * self.bias_lr)
        return final_i, gate, z, None
