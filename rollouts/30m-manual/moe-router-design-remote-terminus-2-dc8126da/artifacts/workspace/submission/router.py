from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from moe_api import RouteSpec, assign_slots, register


@register("elastic_bias")
class ElasticBiasRouter(nn.Module):
    def __init__(self, cfg, aux_coef: float = 5e-3, z_coef: float = 1e-3,
                 bias_lr: float = 0.05, reroute_rounds: int = 3):
        super().__init__()
        self.w_gate = nn.Linear(cfg.n_embd, cfg.n_expert, bias=False)
        nn.init.normal_(self.w_gate.weight, mean=0.0, std=0.02)
        self.register_buffer("bias", torch.zeros(cfg.n_expert))
        self.aux_coef = float(aux_coef)
        self.z_coef = float(z_coef)
        self.bias_lr = float(bias_lr)
        self.reroute_rounds = int(reroute_rounds)

    def forward(self, x: torch.Tensor, spec: RouteSpec):
        G, S, _ = x.shape
        E = spec.n_expert
        k = spec.top_k
        logits = self.w_gate(x.float())
        probs = F.softmax(logits, dim=-1)
        sel = probs + self.bias

        ranked = torch.argsort(sel, dim=-1, descending=True)
        idx = ranked[:, :, :k].contiguous()

        keep_count = min(k, (spec.capacity * E) // S)
        keep = (torch.arange(k, device=x.device) < keep_count).view(1, 1, k).expand(G, S, k).contiguous()

        repl = torch.zeros(G, S, dtype=torch.long, device=x.device)
        for r in range(self.reroute_rounds - 1):
            slot = assign_slots(idx, keep, E, spec.capacity)
            dropped = (slot < 0) & keep
            cum = dropped.long().cumsum(dim=-1) - dropped.long()
            rank_pos = (k + repl.unsqueeze(-1) + cum).clamp(max=E - 1)
            cand = torch.gather(ranked, 2, rank_pos)
            idx = torch.where(dropped, cand, idx)
            repl = repl + dropped.long().sum(dim=-1)

        gate = torch.gather(probs, 2, idx) * keep.to(probs.dtype)
        gate = gate / gate.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        oh = F.one_hot(idx, E).to(probs.dtype) * keep.to(probs.dtype).unsqueeze(-1)
        f = oh.sum(dim=(1, 2)) / float(S * keep_count)
        p = probs.mean(dim=1)
        aux = self.aux_coef * E * (f * p).sum(dim=-1).mean()
        z = self.z_coef * (torch.logsumexp(logits, dim=-1) ** 2).mean()

        if self.training:
            load = oh.detach().sum(dim=(1, 2)).mean(dim=0) / float(S * k)
            self.bias.add_(self.bias_lr * (1.0 / float(E) - load))

        return idx, gate, aux + z, keep
