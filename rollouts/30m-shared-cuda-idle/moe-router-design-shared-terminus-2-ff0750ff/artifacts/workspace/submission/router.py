from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from moe_api import RouteSpec, register


@register("slotfix")
class SlotFixRouter(nn.Module):
    """Top-k token choice with device-limited routing and slot-aware gates.

    Gates are normalized over the choices that actually receive a capacity
    slot (predicted causally with the same prefix-count the harness uses),
    so at the tight deployment capacity a token keeps a full-magnitude output
    instead of losing the share that overflows.
    """

    def __init__(self, cfg, aux_coef: float = 1e-2, z_coef: float = 1e-3,
                 noise_std: float = 0.0, n_device_choices: int = 0,
                 slot_norm: bool = True):
        super().__init__()
        self.w_gate = nn.Linear(cfg.n_embd, cfg.n_expert, bias=False)
        nn.init.normal_(self.w_gate.weight, mean=0.0, std=0.02)
        self.aux_coef = aux_coef
        self.z_coef = z_coef
        self.noise_std = noise_std
        self.n_device_choices = n_device_choices
        self.slot_norm = slot_norm

    def forward(self, x: torch.Tensor, spec: RouteSpec):
        G, S, _ = x.shape
        E = spec.n_expert
        k = spec.top_k
        logits = self.w_gate(x.float())                              # (G, S, E)
        if self.training and self.noise_std > 0:
            logits = logits + torch.randn_like(logits) * self.noise_std

        probs = F.softmax(logits, dim=-1)
        if self.n_device_choices > 0 and spec.ep_devices > 1:
            D = spec.ep_devices
            per = E // D
            pd = probs.view(G, S, D, per)
            dscore = pd.amax(dim=-1)                                 # (G, S, D)
            nd = min(self.n_device_choices, D)
            thr = dscore.topk(nd, dim=-1).values[..., -1:]
            dmask = dscore >= thr
            probs_sel = (pd * dmask.unsqueeze(-1).to(pd.dtype)).reshape(G, S, E)
        else:
            probs_sel = probs

        topk_p, topk_i = probs_sel.topk(k, dim=-1)                   # (G, S, k)

        # Causal prediction of which choices get a slot (mirrors assign_slots).
        flat = topk_i.reshape(G, S * k)
        oh = F.one_hot(flat, E)
        rank = (oh.cumsum(dim=1) - oh).gather(2, flat.unsqueeze(-1)).squeeze(-1)
        slotok = (rank < spec.capacity).reshape(G, S, k)

        if self.slot_norm:
            denom = (topk_p * slotok.to(topk_p.dtype)).sum(dim=-1, keepdim=True)
            fb = topk_p.sum(dim=-1, keepdim=True)
            denom = torch.where(denom > 1e-9, denom, fb).clamp_min(1e-9)
            gate = topk_p / denom
        else:
            gate = topk_p / topk_p.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        # Switch load-balancing loss per sequence over all choices + z-loss.
        frac_tokens = F.one_hot(topk_i, E).float().sum(dim=-2).mean(dim=1) / k
        frac_prob = probs.mean(dim=1)
        aux = self.aux_coef * E * (frac_tokens * frac_prob).sum(dim=-1).mean()
        z = self.z_coef * (torch.logsumexp(logits, dim=-1) ** 2).mean()
        return topk_i, gate, aux + z, None
