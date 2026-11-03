from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from moe_api import register, RouteSpec


@register("devroute")
class DevRoute(nn.Module):
    """Token-choice top-k, but each token's choices are confined to its best
    `dev_top` expert-parallel devices (by router prob mass on that device).
    This cuts charged device traffic from ~2.6 remote visits/token to
    dev_top-1, freeing budget for more training steps. Switch aux loss on the
    masked distribution keeps experts (and hence devices) balanced; z-loss as
    in ST-MoE."""

    def __init__(self, cfg, aux_coef: float = 1e-2, z_coef: float = 1e-3,
                 n_dev: int = 16, dev_top: int = 2):
        super().__init__()
        self.w_gate = nn.Linear(cfg.n_embd, cfg.n_expert, bias=False)
        nn.init.normal_(self.w_gate.weight, mean=0.0, std=0.02)
        self.aux_coef = aux_coef
        self.z_coef = z_coef
        self.n_dev = n_dev
        self.dev_top = dev_top

    def forward(self, x: torch.Tensor, spec: RouteSpec):
        n_exp = spec.n_expert
        G, S, _ = x.shape
        nd = self.n_dev
        epd = n_exp // nd
        logits = self.w_gate(x.float())                       # (G,S,E)
        probs = F.softmax(logits, dim=-1)
        p4 = probs.reshape(G, S, nd, epd)
        pdev = p4.sum(dim=-1)                                 # (G,S,nd)
        top_dev = pdev.topk(self.dev_top, dim=-1).indices     # (G,S,dt)
        dmask = F.one_hot(top_dev, nd).to(probs.dtype).sum(dim=-2)
        flat = (p4 * dmask.unsqueeze(-1)).reshape(G, S, n_exp)
        topk_p, topk_i = flat.topk(spec.top_k, dim=-1)        # (G,S,k)
        gate = topk_p / topk_p.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        frac_tokens = F.one_hot(topk_i, n_exp).to(probs.dtype).sum(dim=-2).mean(dim=1) / spec.top_k
        frac_prob = flat.mean(dim=1)
        aux = self.aux_coef * n_exp * (frac_tokens * frac_prob).sum(dim=-1).mean()
        z = self.z_coef * (torch.logsumexp(logits, dim=-1) ** 2).mean()
        return topk_i, gate, aux + z, None
