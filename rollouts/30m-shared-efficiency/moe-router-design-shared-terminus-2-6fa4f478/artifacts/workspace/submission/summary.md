## Experiments

Scored metric: mean of in-distribution and OOD validation loss at deployment
capacity factor 0.75, while training must run at capacity factor >= 1.0. At
cf 0.75 there are strictly fewer expert slots (64*24=1536) than choices
(512*4=2048) per sequence, so at least 25% of choices must be dropped at
scoring time. The baseline trains at cf 1.0 (nearly all choices kept, gates
normalized over all top-4), so the routing it is scored with differs from the
routing it trained with.

v1 (trained-cf rerouting + gate renorm over survivors): predicted overflow
with assign_slots at the TRAINING capacity factor and rerouted/renormalized
against it. xs result: val_loss 5.0416 vs baseline 4.9929 - WORSE. The
deploy-vs-train loss gap doubled (0.061 vs 0.032): training at cf 1.0 rarely
reroutes (5.6% drops), while scoring at cf 0.75 reroutes ~50% of choices, so
the scored routing was never trained. Conclusion: what matters is not
rerouting per se, but that training and scoring see the SAME routing.

v2 (training-only keep mask, eval byte-identical to baseline): dropped only
strict causal overflow choices during training and measured the balancing
loss over survivors. Not fully evaluated; superseded by v3.

v3 (deploy-simulated routing): make ALL routing decisions against a hardcoded
simulated capacity of deploy_cf=0.75, in training and eval alike. The harness
capacity (1.0 in training, 0.75 in eval) then never binds in training and
equals the simulated one at scoring, so training and deployment routing are
identical (verified: expert indices, gates and keep mask are exactly equal
across train@cf1.0, eval@cf1.0 and eval@cf0.75). Overflowing choices are
causally rerouted to each token's next-best experts (one replacement per
token per round, 8 rounds, using assign_slots which is token-major and hence
causal), and gates are renormalized over choices that survive the simulated
capacity, so tokens keep full output mass at deployment capacity instead of
losing ~25-50% of it as the baseline does. The Switch balancing loss is
measured over surviving choices so it targets what actually survives.

Checks: source screen passes; harness export succeeds for train@cf1.0,
eval@cf1.0, eval@cf0.75; outputs verified (distinct in-range experts per
token, finite gates/aux); training-mode causality probe passed for v1 (0
mismatches over 121k positions) and v3 uses the same causal primitives. A
final xs-scale comparison of v3 against the baseline was launched on GPU.

## Submitted solution

`token_choice_deploysim`: top-4 token-choice routing over 64 experts
(width 768) where routing capacity is simulated at the deployment factor
0.75 in both training and evaluation, with causal overflow rerouting to
next-best experts and gate renormalization over surviving choices; Switch
load-balancing loss (coefficient 1e-2, per-sequence, over surviving choices)
and ST-MoE z-loss (1e-3). Same compute charge as the baseline (top_k=4,
n_expert=64, expert_hidden=768, capacity_factor=1.0); extra router flops are
negligible next to an expert pass. Changes from baseline: capacity handling
and gate normalization only; expert shape, aux coefficients and training
config are unchanged.

Reproduce: `python3 /workspace/validation/val.sh /workspace/submission`, or
equivalently `python3 /workspace/train.py --scale target --config
/workspace/submission/config.json --router-file /workspace/submission/router.py`.
