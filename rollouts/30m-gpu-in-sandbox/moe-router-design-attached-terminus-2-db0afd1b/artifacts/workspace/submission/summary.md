# Routing design for the sparse MoE

## Submitted solution

`devaware_tc`: top-4 token-choice routing over the reference 64 experts at
training capacity factor 1.0, with the Switch load-balancing aux loss
coefficient 1e-2 measured per sequence over all choices, and the ST-MoE z-loss
at 1e-3 - the baseline's proven balancing recipe, unchanged.

The one change from the baseline is in the gates: after top-k selection, a
choice whose expert-parallel device (expert e on device e // 4 of 16) is not
touched by any other choice of the same token is discounted before the gates
are normalized over the k choices (`selected = p * (1 - gamma * present)`,
gamma = 1, then renormalize). Discrete routing, the aux loss, balance and
capacity behaviour are identical to the baseline; only the mixture weights
shift toward choices that share a device with the token's other choices, so
cross-device choices must earn their weight. The rule is per-token, hence
causal, and adds only small (G,S,16,16) matmuls to the router program.

Reproduce: `python3 /workspace/train.py --scale target --config /workspace/submission/config.json --router-file /workspace/submission/router.py`
(config: top_k 4, n_expert 64, expert_hidden 768, n_shared_expert 0,
capacity_factor 1.0, router_lr_mult 1.0, seed 1337.)

## Experiments

All experiments at xs scale with an xs-shaped config (expert_hidden 384),
scored by val_loss at the deployment capacity factor 0.75.

1. Aux-loss-free bias balancing (DeepSeek-V3 style, sigmoid scores + sign-of-
   load-error bias buffer updates): FAILED badly at this scale (val 5.39 vs
   4.99). Sigmoid compression of logits made the bias dominate selection;
   routing collapsed (util_entropy 0.36, drop 0.93, ep_imbalance 5.9x). The
   mechanism is causality-safe (the training probe restores buffers before each
   compared forward; 0 mismatches), but it needs a stronger aux scaffold than
   the literature suggests at this size; abandoned.
2. Baseline recipe + device-aware gate discount (submitted): val 4.998,
   util_entropy ~1.0, drop_rate 0.138 at deployment cf, ep_imbalance 1.40,
   traffic share 8.9%, causality probes 0 mismatches. Matches baseline-quality
   balancing; gate discount is the only delta.
3. (2) + keep-mask dropping of each token's relatively weakest choices (below
   0.6x the token's mean selected prob) with gates renormalized over kept
   choices: val 5.029, slightly worse, with no charge savings (imbalance and
   traffic unchanged - the aux loss already keeps load uniform). Reverted.

The submission is experiment 2: the baseline recipe with the device-aware gate
discount, the most reliable recipe found within the time budget.
