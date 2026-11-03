## Experiments

Hypothesis: at top-4 over 64 experts on 16 expert-parallel devices, a token's
choices reach ~3.75 devices on average. Each device beyond the first is charged
4*n_embd*2 bytes at 1000 FLOP/byte (~one target expert pass), so the baseline
spends ~9.1% of its whole compute budget on all-to-all traffic (measured at xs
scale: ep_traffic_share 0.091, ep_devices_per_token 3.75). Limiting each token
to 2 devices (DeepSeek-V3 node-limited routing, adapted to the harness' devices)
caps traffic at 1 extra device per token; the freed budget buys ~5% more
training steps (all tokens see the full LR cosine, so extra steps directly
lower loss).

Evaluation at xs scale (4 layers, width 384, 64 experts, top-4, cf 1.0, seed
1337), scored at deployment capacity factor 0.75:

| run | val_loss@0.75 | traffic share | devices/token | steps |
|-----|---------------|---------------|---------------|-------|
| baseline token-choice (aux 1e-2, z 1e-3) | 5.0000 | 9.1% | 3.75 | 2197 |
| device-limited n_dev=2 (aux 1e-2, z 1e-3) | 4.9633 | 3.5% | 2.00 | 2316 |

Device-limited routing improved val loss by 0.037 at identical seed and budget.
The training-mode causality probe reported 0 mismatch. Drop rate at deploy cf
also fell slightly (0.147 vs 0.162). A follow-up run with aux_coef 0.005 did not
finish before the wall-time limit, so the submitted coefficients are the
validated baseline ones (1e-2, 1e-3).

Other designs considered but not run for lack of wall time: aux-free bias
balancing (DeepSeek-V3 style), n_dev=3, shared experts, training cf > 1.0.

## Submitted solution

token-choice top-4 routing over 64 experts (width 768), restricted so each
token's 4 choices come from at most 2 of the 16 expert-parallel devices:
device score = sum of the token's top-2 (top_k/n_dev) softmax probabilities
within the device; the 2 best devices are kept, all other experts masked to
zero, and the top-4 experts and gates are taken from the masked distribution
(gates renormalized to sum 1). Switch per-sequence load-balancing loss
coefficient 1e-2 (still over all 64 experts, keeping utilization uniform,
util_entropy 0.999) and ST-MoE z-loss 1e-3, trained at capacity factor 1.0.
Changes from baseline: the device limit only; everything else matches the
baseline recipe.

The router is causal (per-token function of its own hidden state), stateless in
eval, traceable by torch.export (topk/sum/scatter/expand only), and its only
parameter is the gate linear (49152 params, well under the 1% budget).

Reproduce: `python3 /workspace/train.py --scale target --config
/workspace/submission/config.json --router-file /workspace/submission/router.py`.
The xs validation numbers above reproduce with the same files and an xs-shaped
config (expert_hidden 384) as in /workspace/work/cfg_*.json.
