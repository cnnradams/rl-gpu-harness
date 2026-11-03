## Experiments
Hypothesis tested: at the deployment capacity factor 0.75 (tighter than the
training floor of 1.0), the baseline token-choice router drops ~25% of
choices and each dropped choice takes its gate weight with it, shrinking the
MoE output of affected tokens. I built a capacity-aware router ("capaware")
that predicts causally and exactly which choices survive capacity (rank of
each choice among earlier same-expert choices in token-major order, matching
moe_api.assign_slots; verified equal to assign_slots at capacity factors
0.75/1.0/1.25/2.0) and renormalizes gates over surviving choices. It passed
the source screen, torch.export, and the training causality probe (0
mismatches), and its train->deploy loss gap was small (5.0169 -> 5.0513 at
xs). However, in a controlled xs-scale A/B (same config, same seed: top-4,
64 experts, expert_hidden 384, cf 1.0, Switch aux 1e-2, z-loss 1e-3) the
baseline router reached val_loss 4.8926 (gap 0.033) versus 5.0513 (gap
0.034) for the capacity-aware variant: the gate renormalization distorted
router learning far more than the ~0.03-nat deployment-capacity effect it
was meant to recover. No other variants (bias balancing, shared experts,
granularity) were reachable in the time budget.

## Submitted solution
The tuned canonical baseline: top-4 token-choice routing over the reference
64 experts (expert_hidden 768), Switch load-balancing loss measured per
sequence over all choices (coefficient 1e-2) plus ST-MoE router z-loss
(coefficient 1e-3), capacity factor 1.0, seed 1337, no shared experts.
Chosen because it empirically beat the capacity-aware alternative at xs
scale and is the known-competent recipe (measured validation reward
-3.847). Reproduce with: SEED=1337 bash /workspace/baseline/baseline.sh
then python3 /workspace/train.py --scale target --config
/workspace/submission/config.json --router-file /workspace/submission/router.py
