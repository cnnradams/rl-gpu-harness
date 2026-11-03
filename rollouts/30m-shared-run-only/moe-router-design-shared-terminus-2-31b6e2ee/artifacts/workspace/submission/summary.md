## Experiments

Baseline (measured): top-4 token-choice softmax routing over 64 experts
(Switch aux loss 1e-2 per sequence over all choices, ST-MoE z-loss 1e-3),
capacity factor 1.0. At xs scale (expert_hidden 384): deploy-CF (0.75)
val_loss 4.9436, train-CF val_loss 4.9133, ep_imbalance 1.366.

Hypothesis tested: evaluation happens at deployment capacity factor 0.75,
below the trainable floor of 1.0, where ~25% of choices overflow and each
dropped choice loses its gate share. I built a capacity-aware router
('capacity_reroute'): the top_k choice slots are processed in order and each
choice is sent to its highest-probability expert that still has a free slot
(causal exclusive-prefix accounting); then the harness's exact token-major
capacity queue is replicated and keep=(rank<capacity) is iterated to a
fixpoint so the keep mask equals exactly the surviving choices, and gates are
normalized over survivors only. At CF 0.75 every token keeps its 3 best
rerouted choices with renormalized weights. It passed the source screen,
torch.export, causality probes, and lowered ep_imbalance to 1.261.

Result (xs scale, full run): deploy val_loss 5.151 vs baseline 4.944, and
also worse at train CF (5.139 vs 4.913), so the hard greedy rerouting itself
degraded model quality (forced assignments early in training plus reduced
effective top-k diversity); the device-imbalance savings did not compensate.
Conclusion: the hypothesis that deployment-capacity drops dominate did not
hold at the tested scale - the published token-choice recipe with Switch
balancing generalizes better, and the agent time budget did not permit a
scaling ladder of alternatives (aux-loss-free bias balancing, shared experts).

## Submitted solution

The tuned baseline recipe: top-4 token-choice routing over 64 experts
(expert_hidden 768 at target scale), no shared experts, capacity_factor 1.0,
router_lr_mult 1.0, seed 1337, Switch load-balancing loss (coefficient 1e-2)
measured per sequence over all choices, plus the ST-MoE router z-loss
(coefficient 1e-3). Gates are the top-4 softmax probabilities renormalized
over the four choices.

Changes from the shipped baseline: none in the recipe; this is the measured
best of the designs I could evaluate within the time budget (baseline
validation reward -3.847 +/- 0.012 over 3 runs).

Reproduce: `SEED=1337 bash /workspace/baseline/baseline.sh` (writes this
bundle), or directly `python3 /workspace/train.py --scale target --config
/workspace/submission/config.json --router-file /workspace/submission/router.py`.
