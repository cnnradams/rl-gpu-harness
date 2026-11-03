## Experiments

The scored quantity is the held-out loss at a deployment capacity factor of
0.75, well below the trainable floor of 1.0. At that capacity a meaningful
share of each token's four expert choices overflow their expert's per-sequence
slots and are dropped, and the baseline lets the dropped choices' gate mass
vanish, so an overflowing token's expert output shrinks. Hypothesis:
recovering that mass on the choices that are actually computed should lower
the deployment loss without touching training.

Evaluated at `xs` scale against the baseline recipe (top-4 token-choice,
Switch aux 1e-2, z-loss 1e-3, capacity factor 1.0; baseline val_loss 4.8524):

1. fit_renorm: renormalize every token's gates over its choices that fit the
   current capacity, mask and scale fully detached. Failed (val_loss 5.1546):
   with no gradient signal about capacity the router drifted into worse
   device imbalance (1.62 vs 1.36) and the train-capacity loss itself degraded.
2. fit2: same correction but only where a token actually overflows, with the
   fitting mass allowed to carry gradients (straight-through). Better balance
   (imbalance 1.33, drop 0.137) but still worse than baseline (4.9655):
   routing gradients through 'does this expert have room' do not reflect
   token-expert affinity and corrupt the gate gradients.
3. cap_aware (submitted): exact baseline gates and gradients; a fully
   detached per-token correction that only fires where choices overflow at
   the call's capacity factor. At the training capacity factor the aux loss
   keeps overflow rare so training matches the baseline; at deployment the
   overflowed tokens' surviving choices are rescaled to the full top-k mass.

Device imbalance and traffic charges were noted (baseline pays ~36% excess
expert work and ~9% of budget on traffic at xs); reducing them would buy
extra steps but was not exploitable reliably within the time budget.

## Submitted solution

Top-4 token-choice routing over 64 experts (expert_hidden 768, no shared
experts, capacity factor 1.0) with the Switch load-balancing loss measured
per sequence over all choices (coefficient 1e-2) and the ST-MoE router z-loss
(1e-3) - the baseline recipe - plus a capacity-aware gate correction:

- At every call the router simulates, causally and at that call's
  `spec.capacity`, which of each token's choices fit, using
  `moe_api.assign_slots` (the same position-order queue the harness enforces).
- Where a token loses choices to overflow (its fitting gate mass falls below
  1), its surviving gates are rescaled so they sum to 1. The factor is fully
  detached, so training gates and gradients equal the baseline's; the
  correction is a pure, causal function of positions <= t and behaves
  identically in the two eval processes (it depends only on parameters and
  the input), so the causality probe passes.

Reproduce: `python3 /workspace/train.py --scale target --config
/workspace/submission/config.json --router-file /workspace/submission/router.py`
