# Capacity-aware top-k routing for the deployment capacity factor

## Experiments
Hypotheses and approaches tried, all evaluated at xs scale (4 layers, width 384,
64 experts, top-4, 512-token groups, 16 devices) with the harness's own
training loop, which scores at deployment capacity factor 0.75:

1. Baseline (top-4 token choice, Switch aux loss 1e-2 over all choices per
   sequence, ST-MoE z-loss 1e-3, trained at capacity factor 1.0): gates are
   normalized over all k=4 choices, so at deployment capacity 0.75 every
   dropped choice (about a quarter of them) silently removes its share of the
   token's MoE output. Held-out loss therefore degrades exactly where the
   model is scored.
2. Capacity-aware gating (submitted): the router replicates the harness's
   causal slot assignment (moe_api.assign_slots semantics) on its own choices
   at the capacity factor of the current call, and normalizes gates over only
   the choices that survive capacity; dropped choices get gate 0. At training
   capacity 1.0 this changes little (few drops), while at deployment 0.75
   each token keeps full output mass over its ~3 surviving choices. A first
   implementation used -inf masking and failed the training-mode causality
   probe: tokens whose 4 choices all overflow produced NaN gates, and
   NaN != NaN was counted as a routing change. Masking with -1e9 (softmax
   underflows to exact 0 for dropped choices whenever any choice survives,
   uniform finite gates when none does) fixed it: 0 prefix mismatches on the
   replicated probe.
3. Device-concentration bonus (router kwarg dev_bonus, adds a bias toward
   experts on the top-1 expert's device to cut the device-traffic charge):
   implemented but disabled in the final config; at xs the traffic share is
   only ~8% of budget and early runs did not show a clear win once the
   resulting extra drops at deployment capacity were accounted for. Left as
   an option, default 0.
4. Auxiliary loss normalization: the Switch aux value at uniform routing is
   top_k / (n_expert * group_size); measuring the loss as excess over this
   floor (clamped at 0) keeps the published coefficient 1e-2 meaningful at
   the deployment group statistics. Kept, as it is strictly closer to the
   intended Switch penalty.

## Submitted solution
Top-4 token-choice routing over the 64 reference experts (width 768 at target
scale), no shared experts, trained at capacity factor 1.0 with the Switch
load-balancing loss (coefficient 1e-2, per routing group over all choices,
excess-over-uniform form) and the ST-MoE z-loss (coefficient 1e-3) — the
baseline recipe — with one change: gates are normalized over the choices that
survive capacity at the capacity factor of the current call, computed with a
causal in-router replica of the harness's slot assignment (prefix counts per
expert within each sequence). The router is deterministic, keeps no buffers,
updates state only never (it is stateless), and behaves as a pure function of
its input in eval mode, so both causality probes pass. Reproduce with:

    python3 /workspace/train.py --scale target \
      --config /workspace/submission/config.json \
      --router-file /workspace/submission/router.py
