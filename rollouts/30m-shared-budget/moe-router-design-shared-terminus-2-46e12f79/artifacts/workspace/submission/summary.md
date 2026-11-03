## Experiments

Starting point is the shipped baseline: top-4 token-choice over 64 experts,
Switch aux loss (1e-2) and ST-MoE z-loss (1e-3), trained at capacity factor
1.0 and scored at deployment capacity 0.75.

Hypotheses tested at xs scale (64 experts, top-4, width 384, 600 steps):

1. **Train/eval capacity mismatch dominates.** Training at cf=1.0 drops almost
   no choices, but scoring rebuilds the model at cf=0.75 where ~25% of choices
   overflow and the baseline loses the dropped gate mass outright (no
   renormalization). Our router simulates the harness slot assignment at the
   deployment capacity with `moe_api.assign_slots`, drops overflowing choices
   itself via the `keep` mask, and renormalizes each token's gates over the
   choices that survive, so training matches scoring exactly.
2. **Loss-free load balancing.** A per-expert bias buffer (DeepSeek-V3 style),
   updated in-place during training only toward uniform kept load, balances
   routing without the gradient interference of a large Switch aux loss, so
   the aux coefficient can be cut from 1e-2 to 2e-3. The bias is a persistent
   buffer, so the learned balance carries into evaluation.

Ablations compared the baseline router against this design under the same
compute charge at xs scale, measuring val loss at the deployment capacity
(`val_loss`) reported by `train.py`.

## Submitted solution

Top-4 token-choice over softmax probabilities shifted by a trainable-free
load-balancing bias buffer (persistent, updated only while `self.training`).
Before returning, the router runs `assign_slots` at the deployment capacity
(cf=0.75), marks overflowed choices `keep=False` and renormalizes gates over
surviving choices with a floor of 0.25 (clamped at 4) to bound the output of
tokens whose probability mass mostly overflowed. Aux losses: small Switch
load-balancing term (2e-3) over kept choices per routing group plus ST-MoE
z-loss (1e-3). Config: reference shape (64 experts x 768, top-4, no shared
experts), trained at cf=1.0 (the floor; slots are paid as compute), seed 1337,
router_lr_mult 1.0.

Reproduce: `python3 /workspace/train.py --scale target --config
/workspace/submission/config.json --router-file /workspace/submission/router.py`.
