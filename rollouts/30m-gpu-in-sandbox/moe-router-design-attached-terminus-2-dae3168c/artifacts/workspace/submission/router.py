import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from moe_api import register, RouteSpec, capacity_of


@register("balanced_reroute")
class BalancedRerouteRouter(nn.Module):
    """Token-choice top-k routing with three changes from the Switch baseline:

    1) A per-expert selection bias (persistent buffer) is added to the routing
       scores before top-k. In training it follows the aux-loss-free DeepSeek-V3
       sign rule: experts chosen more than the uniform share get their bias
       decreased, others increased. It is frozen in eval and saved in the
       state_dict, so the exported eval programs reuse the trained balance.
    2) When capacity is tight (capacity * n_expert < tokens * k, which holds at
       the deployment capacity factor), choices whose expert is already full are
       greedily rerouted, in position order, to the not-yet-full expert with the
       most remaining room. This uses only prefix counts (like the harness's own
       assign_slots), so it is causal, and it recovers choices the baseline
       would drop at deployment capacity.
    3) Gates are the softmax probabilities of the final (possibly rerouted)
       experts, renormalized over choices predicted to survive capacity, so a
       rerouted token keeps a full-weight output instead of losing signal.

    A mild Switch load-balancing loss over all choices and the ST-MoE z-loss
    remain as regularizers. Loss terms use batch statistics, which is allowed
    (losses are not routing).
    """

    def __init__(self, cfg, aux_coef: float = 3e-3, z_coef: float = 1e-3,
                 bias_lr: float = 2e-3, bias_cap: float = 6.0,
                 reroute: bool = True):
        super().__init__()
        self.w_gate = nn.Linear(cfg.n_embd, cfg.n_expert, bias=False)
        nn.init.normal_(self.w_gate.weight, mean=0.0, std=0.02)
        self.register_buffer("expert_bias", torch.zeros(cfg.n_expert))
        self.aux_coef = float(aux_coef)
        self.z_coef = float(z_coef)
        self.bias_lr = float(bias_lr)
        self.bias_cap = float(bias_cap)
        self.reroute = bool(reroute)

    def _greedy_fill(self, idx, n_exp, cap):
        G, S, k = idx.shape
        oh = F.one_hot(idx, n_exp).float()                          # (G,S,k,E)
        rank = (oh.cumsum(dim=2) - oh).gather(3, idx.unsqueeze(-1)).squeeze(-1)
        fits = rank < cap                                           # (G,S,k)
        avail = cap - rank                                          # (G,S,k)
        t = torch.arange(S, device=idx.device, dtype=torch.float32).view(1, S, 1, 1, 1)
        earlier = (t - t.transpose(1, 2)).clamp_min(0.0)            # (G,S,S,1)
        used = (oh.unsqueeze(1) * earlier).sum(dim=2)               # (G,S,k,E)
        free = (avail.unsqueeze(-1) - used).clamp_min(0.0)          # (G,S,k,E)
        best = free.argmax(dim=-1)                                  # (G,S,k)
        return torch.where(fits, idx, best)

    def forward(self, x: torch.Tensor, spec: RouteSpec):
        n_exp = spec.n_expert
        k = spec.top_k
        cap = spec.capacity
        G, S, _ = x.shape
        logits = self.w_gate(x.float())                             # (G,S,E)
        probs = F.softmax(logits, dim=-1)
        biased = probs + self.expert_bias.float()
        topk_s, topk_i = biased.topk(k, dim=-1)                     # (G,S,k)
        if (not self.reroute) or cap * n_exp >= S * k:
            final = topk_i
        else:
            final = self._greedy_fill(topk_i, n_exp, cap)
        oh = F.one_hot(final, n_exp).to(probs.dtype)                # (G,S,k,E)
        rank = (oh.cumsum(dim=2) - oh).gather(3, final.unsqueeze(-1)).squeeze(-1)
        alive = rank < cap                                          # (G,S,k)
        w = probs.gather(-1, final) * alive.to(probs.dtype)
        gate = w / w.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        if self.training:
            target = (S * k) / n_exp
            err = oh.sum(dim=(0, 1, 2)) / target - 1.0                 # (E,)
            self.expert_bias.add_(torch.sign(err) * self.bias_lr)
            self.expert_bias.clamp_(-self.bias_cap, self.bias_cap)
        frac = oh.sum(dim=2).mean(dim=1) / k                        # (G,E)
        favg = probs.mean(dim=1)                                    # (G,E)
        aux = self.aux_coef * n_exp * (frac * favg).sum(dim=-1).mean()
        z = self.z_coef * (torch.logsumexp(logits, dim=-1) ** 2).mean()
        return final, gate, aux + z, None
