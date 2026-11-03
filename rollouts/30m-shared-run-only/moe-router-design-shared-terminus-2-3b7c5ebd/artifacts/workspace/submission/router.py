# Fine-grained MoE routing: aux-loss-free bias balancing (DeepSeek-V3 style),
# device-limited top-k (node-limited routing) with training-time router noise
# to spread device selection, small Switch aux + z-loss, and gate magnitude
# compensation at sub-1.0 (deployment) capacity factors.
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from moe_api import register, RouteSpec


@register("grained_bias")
class GrainedBiasRouter(nn.Module):
    """Token-choice top-k routing with:

    - selection scores = softmax probs + a learned-but-not-backpropped bias
      buffer, updated per training micro-batch by sign(load_error) so load
      stays balanced without a balance loss distorting the gates
      (aux-loss-free balancing, DeepSeek-V3);
    - device-limited routing: each token may only use the `device_limit`
      expert-parallel devices whose summed biased scores are highest, capping
      cross-device traffic at (device_limit - 1) visits per token;
    - training-time Gaussian noise on the logits (GShard-style): with the
      shared bias vector, device selection would otherwise herd onto the same
      devices within a micro-batch, exploding the busiest-device charge;
      noise makes the top-device pick a per-token lottery;
    - a small Switch-style sequence-scope balance loss and ST-MoE z-loss;
    - gates renormalized over the selected experts, scaled up by
      `deploy_gate_scale` at capacity factors below 1.0 to compensate for the
      larger fraction of choices dropped on overflow at deployment.
    """

    def __init__(self, cfg, aux_coef: float = 3e-3, z_coef: float = 1e-3,
                 bias_lr: float = 1e-3, device_limit: int = 2,
                 noise_std: float = 1.0, deploy_gate_scale: float = 1.25):
        super().__init__()
        self.w_gate = nn.Linear(cfg.n_embd, cfg.n_expert, bias=False)
        nn.init.normal_(self.w_gate.weight, mean=0.0, std=0.02)
        self.register_buffer(
            "expert_bias", torch.zeros(cfg.n_expert, dtype=torch.float32))
        self.aux_coef = aux_coef
        self.z_coef = z_coef
        self.bias_lr = bias_lr
        self.device_limit = device_limit
        self.noise_std = noise_std
        self.deploy_gate_scale = deploy_gate_scale

    def forward(self, x: torch.Tensor, spec: RouteSpec):
        E = spec.n_expert
        k = spec.top_k
        G, S, _ = x.shape

        logits = self.w_gate(x.float())                    # (G, S, E)
        if self.training and self.noise_std > 0:
            logits = logits + torch.randn_like(logits) * self.noise_std
        probs = F.softmax(logits, dim=-1)
        scores = probs + self.expert_bias

        dev = spec.ep_devices
        if dev is not None and dev > 1 and self.device_limit < dev \
                and E % dev == 0 and (E // dev) * self.device_limit >= k:
            per_dev = E // dev
            dev_scores = scores.view(G, S, dev, per_dev).sum(dim=-1)
            _, top_dev = dev_scores.topk(self.device_limit, dim=-1)
            dev_mask = F.one_hot(top_dev, dev).sum(dim=-2) > 0       # (G,S,dev)
            exp_mask = dev_mask.unsqueeze(-1).expand(G, S, dev, per_dev)
            exp_mask = exp_mask.reshape(G, S, E)
            scores = scores.masked_fill(~exp_mask, float("-inf"))

        _, topk_i = scores.topk(k, dim=-1)                 # (G, S, k)
        top_p = probs.gather(-1, topk_i)
        gate = top_p / top_p.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        if spec.capacity_factor < 0.999:
            gate = gate * self.deploy_gate_scale

        # Loss-side batch statistics only; never fed into routing decisions.
        frac_tokens = F.one_hot(topk_i, E).float().sum(dim=-2).mean(dim=1) / k
        frac_prob = probs.mean(dim=1)
        aux = self.aux_coef * E * (frac_tokens * frac_prob).sum(dim=-1).mean()
        z = self.z_coef * (torch.logsumexp(logits, dim=-1) ** 2).mean()

        if self.training:
            counts = F.one_hot(topk_i, E).float().sum(dim=(0, 1, 2))
            err = counts.mean() - counts
            self.expert_bias.add_(torch.sign(err) * self.bias_lr)

        return topk_i, gate, aux + z, None
