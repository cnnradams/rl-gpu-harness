# Routing design for the sparse MoE LM

## Experiments

Hypothesis: the fixed compute budget charges device traffic (~one target
expert pass per extra device per token) and device imbalance
(`16 * busiest - total`), so restricting each token's top-4 choices to
experts on one or two of the 16 expert-parallel devices (a small device
gate picks top-m devices; the expert gate picks all 4 experts inside
them, Switch-style balance losses at expert and device level, z-loss on
the masked logits) would cut the traffic charge and buy extra training
steps that outweigh the coarser effective granularity.

Evaluation: full xs-scale runs (identical budget, deployment capacity
0.75 evaluation), compared against the canonical baseline router trained
at the same scale:

| design | val_loss (xs) | ep_imbalance | ep_traffic_share | devices/token |
|---|---|---|---|---|
| baseline top-4 token choice | 4.9596 | 1.36 | 0.090 | 3.70 |
| device-pick m=1 (4 experts on 1 device) | 5.1526 | 1.82 | 0.000 | 1.00 |
| device-pick m=2 (8 experts on 2 devices) | 4.9853 | 1.64 | 0.033 | 1.98 |

Outcome: the hypothesis failed. Concentrating all k choices of a token on
one or two devices raised the busiest-device imbalance charge enough to
cancel most of the traffic saving (imbalance 1.36 -> 1.64-1.82), and the
loss of fine-grained expert choice (16/32 effective experts instead of
64) hurt held-out loss more than the extra steps helped. Both variants
were worse than the baseline at xs, with m=1 clearly worst. With the
remaining time too short to validate any further variant (shared
experts, capacity-factor tuning) at a meaningful scale, the submission
falls back to the tuned, proven baseline recipe.

## Submitted solution

The tuned baseline: top-4 token-choice routing over the reference 64
experts (expert_hidden 768), gates normalized over the k choices, the
Switch auxiliary load-balancing loss measured per sequence over all
choices (coefficient 1e-2), the ST-MoE router z-loss (coefficient 1e-3),
no shared experts, trained at the capacity-factor floor of 1.0 with
router_lr_mult 1.0, seed 1337. This recipe reproduces the reported
baseline validation reward of -3.847. Reproduce with
`SEED=1337 bash /workspace/baseline/baseline.sh` (writes this bundle)
and validate with `/workspace/validation/val.sh /workspace/submission`.
