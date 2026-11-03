## Experiments

All experiments at xs scale (4 layers, width 384, 64 experts, top-4, cf 1.0),
scored at deployment capacity 0.75. Baseline measured there: val_loss 4.8327
(ep_imbalance 1.336, traffic share 9.0%, drop ~0.19).

1. Sigmoid gates + DeepSeek-V3-style aux-loss-free bias balancing +
   deployment-survivor gate renormalization (choices whose queue rank exceeds
   the 0.75 capacity carry zero gate; survivors renormalized). Hypothesis:
   training the scored function and skipping the aux loss should help.
   Result: val_loss 5.154 — clearly WORSE. The sigmoid/renormalized gates and
   zeroed choices slowed optimization; balance itself was perfect
   (entropy 1.0, traffic 1.7%), so balance alone does not set the loss.
2. Softmax top-k (baseline formulation) + reduced Switch aux (3e-3) + bias
   balancing (sign update 1e-3). Hypothesis: let the bias do the balancing and
   shrink the aux loss that steals LM quality. Result: val_loss 4.949 — better
   than (1) but still behind the baseline. Balance was near-perfect
   (ep_imbalance 1.237, traffic 1.7%) yet the loss did not improve, so at this
   scale the full Switch aux coefficient genuinely helps training.

Conclusion: with the limited validation budget, no variant beat the tuned
baseline recipe at xs; the bias balancer is kept because it improves device
balance (hence compute budget / steps) at no cost to the loss formulation,
with the published Switch (1e-2) and ST-MoE z-loss (1e-3) coefficients intact.

## Submitted solution

Top-4 token-choice softmax routing over 64 experts (expert_hidden 768), with:
  * Switch load-balancing aux loss (per-sequence scope, coefficient 1e-2),
  * ST-MoE router z-loss (coefficient 1e-3),
  * an aux-loss-free per-expert bias buffer, updated only in training mode by
    sign(target load - actual load) * 1e-3 per step; the bias shifts expert
    selection only (gates are the plain softmax probabilities), nudging the
    assignment toward perfect device balance, which lowers the device-imbalance
    and traffic compute charges and buys extra training steps.
Trained at capacity factor 1.0, router_lr_mult 1.0, seed 1337.
The router passes /workspace/screen.py, exports cleanly under torch.export,
behaves as a pure function in eval mode (bias updates are training-only), and
showed 0 causality mismatches in train.py's training-mode probe.
Reproduce: python3 /workspace/train.py --scale target --config
/workspace/submission/config.json --router-file /workspace/submission/router.py
