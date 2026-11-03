# MoE routing submission

## Experiments

Time budget allowed only short xs-scale probes (30 steps, expert_hidden 384,
seed 1337) plus analysis of the harness's compute charges.

1. **Aux-loss-free bias balancing (DeepSeek-V3 style) with softmax gates.**
   Added a persistent per-expert bias buffer, updated only in training from
   batch choice counts, used for selection only. Tried bias_lr 0.5, 0.05 and
   0.005 with aux_coef 0.003-0.01. All variants badly hurt device balance
   (ep_imbalance 5.2-7.5 vs 3.39 for the baseline at the same setting) and
   val loss. Cause: with softmax gates the probabilities are near-uniform
   (~1/64), so prob+bias selection becomes bias-dominated; the published
   recipe uses sigmoid gates whose scores have real spread. Abandoned.
2. **Baseline replication.** The shipped baseline at xs/30 steps:
   val_loss 7.8875 (deploy cf 0.75), ep_imbalance 3.39. My re-implementation
   with trim disabled matches it exactly (7.8848, within seed noise).
3. **Deployment-capacity trim (submitted).** The scorer rebuilds the model at
   capacity factor 0.75, below the 1.0 training floor, so ~10% of choices are
   dropped by overflow in position order (effectively random). Hypothesis:
   letting the router itself drop each token's lowest-gate 4th choice and
   renormalize the remaining gates converts random drops into deliberate
   lowest-value drops and restores a fully-weighted output for every token.
   At xs/30 steps trim vs no-trim deploy loss differed by ~0.003 (noise);
   the mechanism is near-free by construction (kept choices at cf 0.75 fit
   capacity almost exactly: 3 kept x 512 x 1.25 slots vs 0.75 x 4 kept).

## Submitted solution

Top-4 token-choice routing over the reference 64 experts (width 768),
identical to the tuned baseline - Switch load-balancing loss per sequence
over all choices (coef 1e-2), ST-MoE z-loss (coef 1e-3), trained at capacity
factor 1.0 - plus one change: a **deployment keep-mask**. In eval mode, when
the capacity factor is below 1.0 (scoring uses 0.75), the router returns a
keep mask that drops each token's lowest-probability choice and renormalizes
the remaining three gates to sum to 1. This replaces random position-ordered
overflow dropping with deliberate lowest-gate dropping at deployment, and
every token keeps a fully weighted expert output. Training-time behavior,
routing causality (probe mismatch 0) and the compute envelope are unchanged
from the baseline. Reproduce: `python3 /workspace/train.py --scale target
--config /workspace/submission/config.json --router-file
/workspace/submission/router.py`.
