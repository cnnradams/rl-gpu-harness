## Experiments
Hypotheses tried, evaluated with full xs-scale runs (same seed, full LR
schedule, scored at the deployment capacity factor 0.75):

1. Aux-loss-free bias balancing (DeepSeek-style per-expert selection bias,
   training-mode-only sign updates) + reduced Switch aux (3e-3) + z-loss +
   gate renormalization over capacity-surviving choices (causal prediction via
   moe_api.assign_slots): val_loss 5.1094 (indist 5.0049 / ood 5.2139).
   Balance metrics were excellent (util_entropy 0.9999, max_expert_share
   1.64%, ep_imbalance 1.370) but the loss was clearly WORSE than the tuned
   baseline recipe.
2. The tuned baseline recipe (top-4 token choice, Switch aux 1e-2 per
   sequence over all choices, z-loss 1e-3, CF 1.0): val_loss 4.9827
   (indist 4.8848 / ood 5.0806), util_entropy 0.9993, drop_rate 0.159,
   ep_imbalance 1.365.

Conclusion: at this scale the full-strength Switch loss does not meaningfully
interfere with the LM objective, and neither removing it in favor of bias
updates nor compensating dropped choices' gate mass helped. Given a strict
time budget that precluded a scaling-ladder study, the submitted design is
the recipe the evidence supports. The renorm and bias mechanisms also
remained validated (clean torch.export, zero causality mismatches in a
200-step training-mode probe), so they can be re-enabled trivially.

## Submitted solution
`token_choice` router: top-4 token-choice routing over 64 experts of width
768, no shared experts. Scores from a single linear gate (std 0.02 init) in
float32; top-4 experts by softmax probability; gates normalized over the four
choices (a dropped choice loses its share). Aux objectives: Switch
load-balancing loss measured per routing group (sequence) over all four
choices at coefficient 1e-2, plus the ST-MoE router z-loss at 1e-3. Trained
at capacity factor 1.0, router_lr_mult 1.0, seed 1337. This matches the
published/tuned baseline mechanism, re-derived and verified with a full
xs-scale training run (val_loss 4.9827, zero causality mismatches, clean
export). Reproduce: `python3 /workspace/train.py --scale target --config
/workspace/submission/config.json --router-file
/workspace/submission/router.py`.
