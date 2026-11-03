## Experiments
Hypothesis: at the deployment capacity factor 0.75 the baseline top-4
token-choice router drops ~25% of choices and never renormalizes gates, so
tokens lose routed signal exactly when capacity is scarce. Alternatives
tried in a CPU harness with the shipped moe_api.assign_slots as ground
truth: (a) conservative candidate-listing capacity tests over top-8
candidates (starves late-sequence tokens: only 58% of choices kept at
training capacity); (b) multi-pass greedy fixpoint rerouting (residual
queue mispredictions, 0.3-6% of kept choices still dropped); (c) the
submitted exact scheme. Verified: kept choices are never dropped by the
harness at capacity factors 1.0 and 0.75, 98% of the theoretically
deliverable choices are delivered at 0.75 (vs ~71% for plain top-4),
gates renormalize to 1, routing is causal under suffix replacement, and
the file passes screen.py.

## Submitted solution
Rerouting router ("reroute") over the reference shape: top_k 4, 64 experts
of width 768, no shared experts, trained at capacity factor 1.0, seed 1337,
Switch load-balancing loss (coef 1e-2, per sequence, over kept choices) and
ST-MoE z-loss (1e-3). Mechanism: the harness queue for a plain top-k queue
is exactly and causally predictable (a choice of expert e is delivered iff
fewer than `capacity` earlier positions also picked e); the router computes
that prediction with a position-axis cumsum, drops overflowing choices
router-side (keep=False, gate 0) and reroutes each token's first
overflowing choice to its rank-(k+1) candidate when that expert provably
has a free slot (earlier kept choices plus earlier reroute listings below
capacity, a conservative causal count). Accepted reroutes are charged
against later keep decisions, so kept choices are guaranteed delivery at
any capacity factor. Gates are renormalized over kept choices only.
Reproduce: `bash /workspace/validation/val.sh /workspace/submission`
(trains at target scale, scores at deployment capacity 0.75).
