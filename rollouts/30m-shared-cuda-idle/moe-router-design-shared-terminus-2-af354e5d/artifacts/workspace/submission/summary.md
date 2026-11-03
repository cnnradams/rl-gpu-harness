## Experiments

Held-out scoring rebuilds the trained model at a deployment capacity factor of
0.75 while training requires cf >= 1.0, so ~25% of all choices overflow at
scoring time. Baseline gates are normalized over all k choices, so any token
whose choices overflow silently loses that share of its output.

Approaches tried (xs scale, full schedule, evaluated at deployment cf):
1. Sigmoid gates + DeepSeek-V3-style aux-loss-free bias balancing + capacity-aware
   gate renormalization. Exported and trained, passed the causality probes, but
   routing collapsed to ~16 experts (util_entropy 0.333, drop_rate 0.94,
   ep_imbalance ~6x) at both weak (aux 3e-3, bias_lr 1e-3) and strong
   (aux 1e-2, bias_lr 5e-2) balancing settings. Gradients through renormalized
   sigmoid gates concentrate routing; the slow bias controller cannot stabilize
   it. Abandoned.
2. Final: keep the proven softmax top-k assignment with the published Switch
   aux (1e-2, per sequence over all choices) and ST-MoE z-loss (1e-3), and add
   capacity-aware gate renormalization only.

## Submitted solution

Top-4 token-choice routing over 64 experts (width 768), softmax gate, Switch
load-balancing loss and z-loss exactly as the tuned baseline, trained at cf 1.0.
Change from baseline: each token's gates are normalized over the choices
predicted to survive capacity at the ACTIVE capacity factor, using
moe_api.assign_slots (causal: a choice's slot depends only on earlier kept
choices, so routing stays causal and it is only a detached mask). At train cf
(1.0, near-uniform load) almost everything is kept and the model matches the
baseline; at the tighter deployment cf, tokens whose choices all survive keep
unit-sum outputs instead of losing magnitude, and overflowing choices keep
their softmax share as a fallback (fallback also covers the all-overflow case).
Router params/state unchanged (one nn.Linear), router compute overhead is a
few one_hot/cumsum ops, so the step count matches the baseline budget.
Reproduce: python3 /workspace/train.py --scale target --config
/workspace/submission/config.json --router-file /workspace/submission/router.py
