"""Capacity-aware causal rerouting router.

The harness queues each group's kept choices in position order and drops a
choice whose expert already holds `capacity` kept choices. For the plain
top-k queue (keep=None) delivery is exactly predictable causally: a choice
of expert e is delivered iff fewer than `capacity` earlier positions also
have e among their top-k. This router computes that prediction with a
cumsum over the position axis, drops overflowing choices router-side
(keep=False, gate 0) and reroutes each token's first overflowing choice to
the token's next-best candidate (rank top_k) when that expert provably has
a free slot: earlier kept top-k choices plus earlier reroute listings
(accepted reroutes are a subset of listings, and only needy tokens list, so
the test is conservative and kept choices are never dropped by the harness
at any capacity factor, including the deployment factor 0.75). Gates are
renormalized over kept choices so tokens keep full output magnitude. All
counts use strictly-earlier positions: routing is a pure causal function,
with no buffers and no RNG, so eval is identical across processes. Load
balancing: Switch aux loss over kept final choices (per sequence) plus the
ST-MoE z-loss.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from moe_api import register, RouteSpec


@register("reroute")
class RerouteRouter(nn.Module):
    def __init__(self, cfg, aux_coef: float = 1e-2, z_coef: float = 1e-3):
        super().__init__()
        self.w_gate = nn.Linear(cfg.n_embd, cfg.n_expert, bias=False)
        nn.init.normal_(self.w_gate.weight, mean=0.0, std=0.02)
        self.aux_coef = float(aux_coef)
        self.z_coef = float(z_coef)

    def forward(self, x: torch.Tensor, spec: RouteSpec):
        G, S, C = x.shape
        E = spec.n_expert
        K = spec.top_k
        cap = float(spec.capacity)

        logits = self.w_gate(x.float())                        # (G,S,E)
        z = self.z_coef * (torch.logsumexp(logits, dim=-1) ** 2).mean()
        probs = F.softmax(logits, dim=-1)

        M = min(K + 1, E)
        cand_p, cand_i = probs.topk(M, dim=-1)                 # (G,S,M)
        p4, i4 = cand_p[..., :K], cand_i[..., :K]              # (G,S,K)

        oh4 = F.one_hot(i4, E).float()                         # (G,S,K,E)
        counts4 = oh4.sum(dim=2)                               # (G,S,E)
        before4 = counts4.cumsum(dim=1) - counts4
        rank4 = before4.gather(2, i4)                          # (G,S,K)
        keep4 = rank4 < cap                                    # exact queue

        kept_counts = (oh4 * keep4.unsqueeze(-1).float()).sum(dim=2)
        earlier_kept = kept_counts.cumsum(dim=1) - kept_counts # (G,S,E)

        need = ~keep4
        nf = need.float()
        first = need & ((nf.cumsum(dim=-1) - nf) == 0.0)       # first overflow
        needy = first.any(dim=-1, keepdim=True)                # (G,S,1)

        rep_i = cand_i[..., K:K + 1]                           # (G,S,1)
        rep_p = cand_p[..., K:K + 1]
        list_e = F.one_hot(rep_i, E).float().squeeze(2) * needy.float()
        earlier_list = list_e.cumsum(dim=1) - list_e           # (G,S,E)
        load = (earlier_kept + earlier_list).gather(2, rep_i)  # (G,S,1)
        acc = first & (load < cap)                             # (G,S,K)

        # Accepted reroutes also consume capacity: finalize the top-k keep
        # mask against earlier kept top-k plus earlier accepted reroutes.
        acc_e = F.one_hot((acc.long() * rep_i.long()).sum(dim=-1), E).float()
        acc_e = acc_e * acc.any(dim=-1, keepdim=True).float()  # (G,S,E)
        earlier_acc = acc_e.cumsum(dim=1) - acc_e
        keep4 = (rank4 + earlier_acc.gather(2, i4)) < cap

        expert_idx = torch.where(acc, rep_i.expand(G, S, K), i4)
        keep = keep4 | acc
        pk = p4 * keep4.float() + acc.float() * rep_p.expand(G, S, K)
        gate = pk / pk.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        keep_f = keep.float()
        ft = (F.one_hot(expert_idx, E).float() * keep_f.unsqueeze(-1)).sum(dim=-2)
        ft = ft.mean(dim=1) / float(K)                         # (G,E)
        fp = probs.mean(dim=1)                                 # (G,E)
        aux = self.aux_coef * float(E) * (ft * fp).sum(dim=-1).mean()

        return expert_idx, gate, aux + z, keep
