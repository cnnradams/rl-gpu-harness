# Sparse MoE routing submission

## Experiments
The dominant scoring lever is the deployment capacity factor of 0.75 (tighter
than the 1.0 training floor): the evaluator rebuilds the trained model at CF
0.75, so load balance directly controls how many choices overflow and lose
signal. Device imbalance (ep_excess) and device traffic also consume the fixed
compute budget, cutting the number of training steps.

Baseline (shipped): top-4 token-choice over 64 experts (width 768), Switch
load-balancing loss (coef 1e-2, per-sequence scope) + ST-MoE z-loss (1e-3), CF
1.0. Clean reruns: xs/250 -> val_loss 6.603 (ep_imbalance 2.04, entropy 0.998);
s/250 -> val_loss 6.681 (198 steps).

Attempt 1 - aux-loss-free bias (DeepSeek-V3 style), WRONG sign/target. A
persistent per-expert bias added only to the scores used to SELECT top-k (gates
from the clean softmax), updated in place in training mode after selection.
Using a per-sequence target (counts - counts.mean) gave POSITIVE feedback: load
collapsed (entropy 0.56, ep_imbalance 7.5), the imbalance charge exhausted the
budget at step 153/250, val_loss 6.87. Reverted. (Causality probe passed,
mismatch 0 - the buffer-update mechanism itself is valid.)

Attempt 2 - corrected negative feedback toward a uniform global target
(bias -= lr * (count - target)/target), Switch aux retained. Swept bias_lr at
matched step counts:
  * lr=2e-4  xs/210: val_loss 6.669 (worse than baseline 6.603) - too strong,
    it distorts the gate's preferred assignment.
  * lr=5e-5  xs/210: val_loss 6.578 (better than baseline 6.603, entropy 0.998,
    ep_imbalance 2.03) - a mild win at xs.
  * lr=5e-5  s/198:  val_loss 6.688 (vs baseline 6.681) - the xs gain did NOT
    transfer to s; difference is within seed noise.

Conclusion: the gentle aux-loss-free bias is roughly neutral (small win at xs,
small loss at s, both near noise). It does not reliably beat the tuned baseline,
and findings at small scales are known not to transfer consistently to target.

## Submitted solution
Top-4 token-choice routing over 64 experts (expert_hidden 768), the Switch
auxiliary load-balancing loss over all choices (coefficient 1e-2, per-sequence
scope), the ST-MoE router z-loss (coefficient 1e-3), trained at capacity factor
1.0, no shared experts, router_lr_mult 1.0, seed 1337. This is the tuned
baseline recipe; our controlled experiments at xs and s showed no reliable
improvement from an aux-loss-free balancing bias, so we submit the proven
configuration rather than risk a worse or unstable variant at target scale.
Reproduce with `bash /workspace/baseline/baseline.sh`.
