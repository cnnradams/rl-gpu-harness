# Sparse MoE routing design

## Experiments
The fixed compute budget charges (a) padded expert slots, (b) expert-parallel
device imbalance (16 devices, busiest device sets the pace) and (c) device
traffic (~one expert-pass per device a token visits beyond the first, for every
choice, kept or dropped). At top_k=4 over 64 experts, a balanced router spreads
each token over ~2.6 extra devices on average, i.e. traffic costs roughly 65%
of the routed expert work, and any residual imbalance adds more. Hypothesis:
moving capacity from routed choices into always-on shared experts (DeepSeekMoE
style) cuts traffic and imbalance sharply, so the same budget buys
substantially more training tokens, while shared experts also stabilize the
common-case computation that the tight deployment capacity factor (0.75) makes
fragile. Concretely: top_k=2 with n_shared_expert=2 keeps the active-parameter
invariant (2+2)*768 = 4*768; the routed pool shrinks to 48 experts (still a
multiple of 16 devices; 3 experts/device), which lowers the expected number of
devices a token's choices touch to ~1.5 (vs ~2.6) and the per-expert slot count
stays 21 (cf 1.0) so capacity pressure per expert is unchanged. Aux-loss
coefficients were left at the published Switch/ST-MoE values (1e-2 / 1e-3),
which sweeps (per the baseline notes) found optimal; capacity slots are charged
as compute so training stays at the cf floor of 1.0. A short xs-scale run
validated that the design exports, passes the training-time causality probe,
and trains under the budget with the expected traffic/imbalance savings.

## Submitted solution
Top-2 token-choice routing over 48 experts of width 768 with 2 always-on
shared experts (iso-active with the baseline top-4/64-expert design), Switch
load-balancing loss over all choices (coef 1e-2, per sequence), ST-MoE z-loss
(1e-3), trained at capacity factor 1.0. The router code is the baseline
token-choice router (generic in top_k/n_expert), so routing behaviour, export
and causality properties are unchanged; only the shape config differs.
Reproduce: `python3 /workspace/train.py --scale target --config
/workspace/submission/config.json --router-file /workspace/submission/router.py`.
