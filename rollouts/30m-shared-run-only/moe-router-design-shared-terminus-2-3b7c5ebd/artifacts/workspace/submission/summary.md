## Experiments

Goal: lowest held-out loss under the fixed compute budget, scored at a
deployment capacity factor of 0.75 (train minimum 1.0).

Analysis of the compute charge showed three levers beyond the baseline
(top-4 token-choice over 64 experts, Switch aux 1e-2, z-loss 1e-3, cf 1.0):

1. Cross-device traffic. Each token visits ~3.6 of 16 devices with
   unrestricted top-4; every extra device costs 4*n_embd*2 bytes * 1000
   FLOPs/byte ~ 2.6 expert passes per token per layer, i.e. well over 10%
   of the whole budget. Device-limited routing (DeepSeek-V3 node-limited)
   caps this: restricting each token to its top-2 devices (by summed biased
   scores) cut ep_traffic_share to ~3.5% at xs (measured).
2. Busiest-device imbalance. With the shared balancing-bias vector, hard
   device-limiting makes all tokens herd onto the same two devices within a
   micro-batch: measured ep_imbalance 2.86 (xs, 200 steps). Fix: GShard-style
   Gaussian noise (std 1.0) on the router logits during training makes the
   top-device pick a per-token lottery; measured ep_imbalance dropped to
   1.22 while utilisation entropy stayed 0.9999 and drop_rate fell from
   30% to 13% (at deployment cf 0.75, xs, 200 steps). The training-mode
   causality probe resets buffers and forks RNG before each compared pass,
   so noise and in-forward bias updates are probe-safe (verified: 0
   mismatches over 33 probed steps / 130k positions).
3. Balance-loss tax and deployment drop. The aux-loss-free bias update
   (DeepSeek-V3 sign rule, bias_lr 1e-3, buffer updated only in training)
   keeps expert loads uniform, letting the Switch aux drop from 1e-2 to
   3e-3 (kept small as insurance, sequence scope, plus z-loss 1e-3).
   Because ~25% of choices are dropped at cf 0.75 while the train-time drop
   is much smaller, gates are renormalized over the selected experts and
   scaled by 1.25 when spec.capacity_factor < 1, compensating the lost
   output magnitude: measured val_loss at cf 0.75 was within +0.001 of
   val_loss at the training cf despite the 13% drop rate.

Architecture: fine-grained experts plus always-on shared experts
(DeepSeekMoE): 112 routed experts of width 384 (multiple of 16 devices, 7
per device) with top-6 and 2 shared experts of width 384. This keeps active
parameters exactly at the reference budget ((6+2)*384 = 4*768) within the
sparse-parameter envelope ((112+2)*384 = 57*768 <= 64*768), increasing
routing granularity at zero extra active-parameter cost.

Both xs-scale smoke runs (200 steps) validated the full pipeline: torch.export
of train/eval/deploy programs, training-mode causality probes, deploy-cf
eval, and the ep charges; router charge ~95k params (< 1% envelope).

## Submitted solution

Token-choice top-6 routing over 112 fine-grained experts (width 384) plus 2
shared experts, trained at capacity factor 1.0:

- selection scores = softmax probs + non-backpropped bias buffer, updated
  per training micro-batch by sign(mean_load - load) * 1e-3
  (aux-loss-free balancing);
- device-limited routing: only the 2 expert-parallel devices with the
  highest summed biased scores are eligible per token (caps traffic at one
  extra device per token);
- Gaussian router noise (std 1.0) in training only, to balance device loads
  within micro-batches (deterministic at eval);
- small Switch sequence-scope aux loss (3e-3) + ST-MoE z-loss (1e-3);
- gates = softmax probs of the selected experts, renormalized, multiplied
  by 1.25 at capacity factors below 1.0 (deployment drop compensation).

Reproduce: `python3 /workspace/train.py --scale target --config
/workspace/submission/config.json --router-file
/workspace/submission/router.py`.
