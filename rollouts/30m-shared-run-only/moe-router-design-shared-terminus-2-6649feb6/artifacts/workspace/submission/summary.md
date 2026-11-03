## Experiments
Hypothesis: the score is measured at a deployment capacity factor of 0.75,
tighter than any trainable factor, so top-4 token-choice routing loses ~25% of
its choices to overflow in position order at evaluation: some tokens lose their
strongest choice, and dropped choices take their gate share of the output with
them. An initial design additionally rerouted overflowing choices to next-best
experts via causal prefix-count slot prediction (moe_api.assign_slots); it
passed the source screen and export but produced degenerate outputs (tie
ranks / fully-dropped tokens) and was rejected by the step-0 validity check,
so it was abandoned. The simplified, robust design was validated end-to-end at
xs scale (64 experts, width 384): training, export in all three call modes and
the training-time causality probe (mismatch 0) all passed, and the
deployment-CF val_loss (7.4867) was only 0.0017 above the train-CF val_loss
(7.4851), versus the ~0.02-0.03 gap typical of plain overflow dropping, while
the training-time behavior is exactly the baseline recipe. Alternatives
considered: expert-choice routing (rejected: whole-sequence top-k breaks
causality), aux-free bias balancing and shared experts (rejected: no tuning
time within the session budget).

## Submitted solution
Adaptive top-4 token-choice router over 64 experts (width 768), no shared
experts, trained at capacity factor 1.0 with the Switch load-balancing loss
(coefficient 1e-2, per sequence, over kept choices) and the ST-MoE router
z-loss (1e-3) - identical to the tuned baseline during training. Changes from
baseline: (1) a capacity-adaptive keep mask keeps only each token's strongest
int(capacity_factor * top_k) choices, so at the deployment factor 0.75 every
token sheds its weakest (4th) choice - exactly matching the 0.75 slot budget
under a balanced load - instead of the harness dropping overflowed choices in
position order (which can drop a token's strongest choice); (2) gates are
renormalized over kept choices so a token keeps its full output magnitude
rather than losing the dropped choice's gate share. All routing decisions are
per-position and causal; the router holds no state beyond its gate matrix.
Reproduce: `python3 /workspace/train.py --scale target --config
/workspace/submission/config.json --router-file /workspace/submission/router.py`.
