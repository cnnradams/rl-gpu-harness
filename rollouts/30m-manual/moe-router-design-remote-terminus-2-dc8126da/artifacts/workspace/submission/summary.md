## Experiments
Hypothesis: the scored deployment capacity factor (0.75) is tighter than any
trainable capacity factor (>=1.0), so a plain token-choice router has ~25% of
its expert choices overflow-dropped at evaluation: dropped choices lose their
gate share and, during any training at tight capacity, their signal. Fix the
router, not the capacity: (a) "elastic keep" - keep only the top
floor(capacity*n_expert/seq_len) choices by gate rank per token (4 at CF>=1.0,
3 at deploy CF 0.75) and renormalize gates over kept choices, so a fully-routed
token keeps a total gate weight of 1 at deployment; (b) causal overflow
rerouting of kept-but-over-capacity choices to each token's next-best experts
with free slots (3 rounds, using moe_api.assign_slots, token-major order keeps
it causal); (c) DeepSeek-V3-style aux-loss-free balancing: a persistent bias
buffer added to probs for expert selection only (gates remain true probs),
updated in training mode only by bias_lr*(1/E - load), plus a reduced Switch
aux loss 0.005 and ST-MoE z-loss 0.001.
Evaluation: full evaluator (validation/score.py) at xs scale, 64 experts of
width 384, seed 1337, vs the shipped baseline router with identical envelope.
Results (reward = -(loss_indist+loss_ood)/2, higher better):
- baseline token_choice (Switch 0.01 + z 0.001, CF 1.0): reward -5.0362
  (indist 4.9344, ood 5.1380, drop_rate 0.159)
- submitted router: reward -4.9758 (indist 4.8810, ood 5.0706, drop_rate 0.082,
  util_entropy 0.99997, max_expert_share 0.0162)
=> +0.0604 nats over baseline at xs, valid run, causality probe 0 mismatch.
What failed: an earlier test bundle was invalid only for a missing summary.md;
no router-level invalidity observed in any run.

## Submitted solution
Token-choice top-4 over the reference 64 experts (expert_hidden 768), trained
at capacity factor 1.0 with:
- elastic keep mask matching choice count to granted capacity (3-of-4 at the
  0.75 deployment factor), gates renormalized over kept choices;
- causal overflow rerouting to next-best experts with free capacity;
- bias-buffer load balancing (train-only buffer updates, persisted in the
  state_dict) + Switch aux 0.005 + z-loss 0.001.
Changes from baseline: keep/reroute logic, bias-based balancing, halved aux
coefficient. Reproduce: python3 /workspace/train.py --scale target --config
/workspace/submission/config.json --router-file /workspace/submission/router.py
