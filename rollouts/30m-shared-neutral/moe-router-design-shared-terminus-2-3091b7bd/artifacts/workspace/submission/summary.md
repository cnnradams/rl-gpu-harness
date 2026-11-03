# Sparse MoE routing submission

## Experiments

All experiments at `xs` scale with `--max-steps 400` (LR schedule compressed;
runs stop at the compute budget, ~341-344 steps), evaluated at deployment
capacity factor 0.75:

| design | val_loss@CF0.75 | ep_imbalance | drop_rate |
|---|---|---|---|
| baseline token-choice (Switch aux 1e-2 + z 1e-3) | 6.3264 | 1.846 | ~0.28 |
| bias-balance, bias_lr 1e-3, no reroute | 6.3418 | 1.731 | 0.279 |
| bias-balance + reroute, bias_lr 1e-3 | 6.3292 | 1.400 | - |
| bias-balance + reroute, bias_lr 4e-3 | 6.2949 | 1.356 | - |

Hypotheses tested:
1. Aux-loss-free balancing (DeepSeek-V3 style per-expert bias updated by
   sign(load error) in training only) avoids load-balancing gradients
   interfering with the LM loss. Works: faster bias adaptation (4e-3) balances
   loads early and improves both loss and device-imbalance charges.
2. Causal overflow rerouting: at the deployment capacity factor 0.75, ~28% of
   choices overflow and are dropped. Predicting drops causally
   (moe_api.assign_slots) and substituting each token's best backup expert
   that still has a free slot (prefix counts of earlier primary choices)
   recovers that signal without breaking causality (probe mismatch 0).

## Submitted solution

`router.py` registers `bias_balance`:
- scores = sigmoid(W x); selection over score + bias. `bias` is a persistent
  buffer (init 0) updated only when `self.training`:
  `bias += 0.004 * sign(1/E - observed_share_of_choices)` per micro-batch.
- top-2k candidates; primary top-k; choices predicted to overflow capacity are
  replaced, per choice position in order, by the token's best backup expert
  with a free slot and not already used by that token (causal: depends only on
  earlier positions' primary choices).
- gates = unbiased sigmoid scores of the chosen experts, normalized to sum 1.
- aux loss = z-loss only (coef 1e-3). No load-balancing loss.
- config.json: reference shape (top_k 4, n_expert 64, expert_hidden 768,
  n_shared_expert 0), capacity_factor 1.0, router_lr_mult 1.0, seed 1337.

Reproduce: `/workspace/validation/val.sh /workspace/submission` or
`python3 /workspace/train.py --scale target --config
/workspace/submission/config.json --router-file /workspace/submission/router.py`.
