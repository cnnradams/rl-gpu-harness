# Sparse MoE routing submission

## Experiments

Compute accounting in train.py drives the design: every top-k choice (kept or
dropped) pays device imbalance and traffic, padded capacity slots are charged
whether filled or not, and scoring happens at deployment capacity factor 0.75
while training requires >= 1.0.

Experiments run (xs scale, train.py with export + training causality probes):

1. Baseline recipe (top-4 token choice, 64 experts, Switch aux 1e-2, z-loss
   1e-3, cf 1.0), 1200-step: val_loss 5.540, ep_devices_per_token 3.72,
   ep_traffic_share 8.9%, ep_imbalance 1.49, drop_rate 0.175.
2. This design, moderate balancing (top-2, 48 experts + 2 shared, aux 5e-3,
   bias_lr 1e-3), 1200-step: +7.6% tokens trained, traffic 3.4%, devices/token
   1.96, but val_loss 5.656 and imbalance 1.63 - balancing too weak.
3. This design, stronger balancing (aux 1e-2, bias_lr 2e-3), full 2440-step xs
   run: val_loss 4.832 (indist 4.750 / ood 4.914), imbalance 1.37, traffic
   3.4%, causality probe mismatch 0. Chosen configuration.

Failed/limited: the weaker balancing variant (2) underperformed; short 200-step
smoke runs were dominated by early-training imbalance and not predictive. GPU
budget ran out before a full-length baseline comparison at xs; the submitted
design was chosen on its structural cost advantages (halved traffic, 87.5% vs
75% expert-mass retention at deployment capacity via shared experts) plus the
full-length xs result.

## Submitted solution

- config.json: top_k=2, n_expert=48 (multiple of 16 devices), expert_hidden=768,
  n_shared_expert=2, capacity_factor=1.0 (floor; padded slots are charged),
  router_lr_mult=1.0, seed=1337. Active params (2+2)*768 = 4*768 exactly;
  total sparse (48+2)*768 <= 64*768.
- router.py (`balanced_shared`): softmax probabilities over 48 experts; top-2
  selection on logits + a persistent balancing-bias buffer (DeepSeek-V3 style:
  bias affects selection only, gates are the selected softmax probabilities
  renormalized over the 2 choices). Aux objectives: Switch load-balancing loss
  per routing group over both choices (coef 1e-2) + ST-MoE z-loss (1e-3).
  Bias update: in training mode only, `bias -= bias_lr * sign(mean_load -
  uniform)` in-place on the buffer; frozen in eval so routing is a pure causal
  function of the token. Bias lives in the checkpoint (well under the 1%
  router-state cap).
- Shared experts (2 of width 768) carry half the active expert mass: never
  dropped at the 0.75 deployment capacity, no device traffic, no imbalance.

Reproduce: `python3 /workspace/validation/val.sh /workspace/submission`
(trains at target scale; scores at deployment capacity 0.75).
