# Bias-balanced token-choice MoE router

## Experiments

Hypotheses considered and evaluated by reading the harness and short smoke runs:

1. **Baseline re-derivation** (top-4 token-choice, Switch aux 1e-2 + z-loss 1e-3,
   capacity factor 1.0): the baseline trains at CF 1.0 but is scored at the
   deployment CF 0.75, where any residual imbalance turns into dropped choices.
   Its strong aux loss also pulls gates toward uniform, which costs LM loss.
2. **Higher training capacity factor (1.25)**: spends ~25% of the fixed compute
   budget on padded buffer slots that carry no extra signal, at a budget where
   dropping is already rare at CF 1.0 with a well-balanced router. Rejected.
3. **Aux-loss-free bias balancing (submitted)**: selection uses softmax probs
   plus a per-expert bias buffer updated in training only by the sign of the
   per-batch load error (DeepSeek-V3 style). Gates come from the unbiased
   probabilities. This directly minimizes imbalance (hence eval-time drops at
   CF 0.75) without distorting the gates. A reduced per-sequence Switch aux
   (3e-3) and the ST-MoE z-loss (1e-3) are kept for early-training balance and
   logit stability.

Validation performed (agent time was limited to 30 minutes, so no full target
run was possible): `screen.py` passes; `torch.export` of the router for all
three call modes (training, eval at train CF, eval at deployment CF) succeeds
via `train.export_routers`, and the exported programs load and validate
(`load_router_export`); in-place buffer updates appear only in the training
program, satisfying the pure-eval rule.

## Submitted solution

Top-4 token-choice over the reference 64 experts (width 768), trained at
capacity factor 1.0. Selection scores are softmax probabilities plus a
persistent per-expert bias buffer; each training micro-batch updates the bias
by `-bias_lr * sign(load_e - mean_load)` in place (never in eval, so eval is a
pure function of the input and the saved state_dict). Gates are the unbiased
probabilities of the chosen experts, normalized over the k choices. Auxiliary
objectives: per-sequence Switch load-balancing loss at 3e-3 (batch statistics
feed only the loss, never routing, so causality holds) plus an ST-MoE z-loss
at 1e-3. Compared to the baseline: bias-based selection for near-zero drops at
the tight deployment capacity, and a 3x smaller aux coefficient to reduce gate
distortion. Reproduce: `python3 /workspace/validation/val.sh /workspace/submission`
(config.json pins seed 1337, top_k 4, n_expert 64, expert_hidden 768,
n_shared_expert 0, capacity_factor 1.0, router_lr_mult 1.0).
