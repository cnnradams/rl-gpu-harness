## Experiments

Evaluation is done at deployment capacity factor 0.75 while training uses at
least 1.0. At CF 0.75 about 16% of choices overflow capacity and are dropped;
the harness does not renormalize the dropped gates, so tokens routed to full
experts silently lose output magnitude (baseline xs: val_loss 4.9256 at train
CF vs 4.9555 at deploy CF, drop_rate 0.163).

Hypotheses tested at xs scale (4 layers x 384, 64 experts, top-4, CF 1.0,
seed 1337):

1. Baseline reproduction (token-choice, Switch aux 1e-2, z-loss 1e-3):
   val_loss 4.9555 at deploy CF (4.9256 at train CF).
2. Slot-aware gate normalization + device-limited routing (top-2 of 16
   devices, then top-4 experts): val_loss 4.9913. Traffic share fell from
   9.0% to 3.7% (+5% steps) but quality regressed at both train CF (4.9588)
   and deploy CF, so the device constraint hurt more than the saved compute
   bought; device limiting was dropped.
3. Slot-aware gate normalization alone: the router predicts causally (same
   prefix-count queue as the harness) which of each token's choices receive a
   capacity slot, and normalizes the softmax gates over only the choices that
   get a slot, at whatever capacity factor the spec carries (an xs ablation run of this variant alone did not complete before the time limit). At deploy CF the
   gates therefore renormalize over the delivered choices instead of losing
   the overflowed share; at CF 1.0+ it coincides with the baseline
   normalization when nothing overflows. See /workspace/exp for configs/logs.

## Submitted solution

Top-4 token-choice routing over 64 experts (width 768), Switch
load-balancing aux loss per sequence over all choices (coefficient 1e-2),
ST-MoE router z-loss (coefficient 1e-3), trained at capacity factor 1.0, no
shared experts - identical to the baseline except for slot-aware gate
normalization: gates are divided by the summed probability of the choices
that actually receive capacity slots (predicted causally, mirrors
moe_api.assign_slots; falls back to the full top-k sum if no choice gets a
slot). This is causal (depends only on the token and earlier positions),
pure in eval mode, traceable (only allowed aten ops), and changes nothing
outside the router module.

Reproduce: `python3 /workspace/train.py --scale target --config
/workspace/submission/config.json --router-file /workspace/submission/router.py`.
