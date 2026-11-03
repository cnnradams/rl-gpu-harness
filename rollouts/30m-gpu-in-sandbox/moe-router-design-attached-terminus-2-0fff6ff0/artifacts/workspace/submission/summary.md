# MoE routing design: aux-loss-free balanced top-4 routing

## Experiments
All comparisons at xs scale (4 layers, width 384, 64 experts, top-4, capacity
factor 1.0, seed 1337), scored on val_loss at the deployment capacity factor
0.75, exactly as the evaluator scores.

1. Re-derived baseline (softmax top-4, Switch aux 1e-2 per sequence, z-loss
   1e-3): val_loss 5.0197 (indist 4.9214, ood 5.1180), drop 0.160,
   ep_imbalance 1.354.
2. Sigmoid gates + DeepSeek-V3-style aux-loss-free balancing bias (per-expert
   bias buffer added to scores for selection only, updated in training by
   sign(mean_load - load) * 1e-3 per routing call) + z-loss 1e-3 + small
   sequence-scope Switch aux 3e-3: val_loss 4.9670 (indist 4.8746,
   ood 5.0595), drop 0.174, ep_imbalance 1.253.
3. Same as (2) but fully aux-free (aux_coef 0): val_loss 4.9439
   (indist 4.8526, ood 5.0351), drop 0.219, ep_imbalance 1.315. Best.
4. Same as (3) but also without z-loss (z_coef 0): val_loss 5.0227,
   ep_imbalance 1.446. Failed: sigmoid gates still need the logit-magnitude
   penalty of the z-loss for stability and balance.

Conclusions: the balancing bias removes the need for the Switch auxiliary loss
entirely (which otherwise trades against LM quality through the gate
gradients), while the z-loss remains necessary. The higher drop rate of the
aux-free variant is a compute charge, not a score term, and the loss improves
anyway at deployment capacity 0.75.

## Submitted solution
Router `sig_bfree` in router.py: token-choice top-4 routing over 64 experts
(width 768). Scores are sigmoid(w_gate x); a per-expert bias buffer is added to
the scores used for top-k selection only, while gates are the raw sigmoid
scores normalized over the k choices. During training (only when
self.training) the bias is updated in place by
sign(mean_load - load) * bias_lr, computed over the routing call's choices;
eval programs are pure functions of the input, so causality holds (verified by
the training-loop probe: 0 mismatches). Aux losses: only the ST-MoE z-loss at
1e-3; no load-balancing loss. Config: top_k 4, n_expert 64, expert_hidden 768,
n_shared_expert 0, capacity_factor 1.0, router_lr_mult 1.0, seed 1337.
Changes from baseline: softmax -> sigmoid gates; Switch aux removed in favor of
the balancing bias buffer. Reproduce:
python3 /workspace/train.py --scale target --config
/workspace/submission/config.json --router-file /workspace/submission/router.py
