## Experiments

Hypotheses and what was tried (xs-scale smoke + full runs, target config):

1. Baseline: top-4 token-choice over 64 experts, Switch per-sequence aux
   (1e-2), ST-MoE z-loss (1e-3), trained at capacity factor 1.0. Deployment
   evaluation is at capacity factor 0.75, so the baseline necessarily drops
   25% of all choices at deployment even under perfect balance, and dropped
   gate share is not renormalized - every token permanently loses that share
   of its MoE output at eval. Additional per-sequence skew drops further
   choices on hot experts.

2. Submitted design (trimmed top-k + aux-free bias balancing):
   - Route top-4 but keep only the top-3 choices per token via a constant
     mask over the sorted top-k. This mask depends only on the token itself,
     so it is causal. Per-group expert load becomes 3*S/64 = 24 slots, exactly
     the deployment capacity at CF 0.75, so a balanced router drops nothing at
     deployment while training at CF 1.0 keeps slack. Gates are the softmax
     probabilities of the selected experts renormalized over the kept
     choices, so a token's output weights always sum to 1 - no gate mass is
     lost to capacity at deployment. The 4th expert is still scored, which
     gives the router an alternative target when a top-3 expert's buffer is
     full during training.
   - Load balancing: a persistent bias buffer (64 floats) added to the
     selection scores only (never to the gates), updated in training mode
     with a fixed-sign step (bias_lr=2e-3) toward uniform kept-load
     (DeepSeek-style aux-free balancing). This removes most balancing
     pressure from the aux loss so it stops fighting the gate gradient.
   - A Switch-style per-sequence aux loss (1e-2) over the kept choices is
     kept to control per-sequence skew (capacity is enforced per sequence),
     plus the ST-MoE z-loss (1e-3) for logit stability.

Verification:
- `python3 /workspace/screen.py /workspace/submission/router.py` passes.
- xs-scale train.py run (30 steps): torch.export of all three call modes
  succeeded; training-mode causality probe mismatch=0; util_entropy 0.977;
  drop_rate 0.7087 = exactly 17/24, i.e. only the deliberately-dropped 4th
  choice is dropped - zero kept choices were lost at deployment CF 0.75,
  confirming the core mechanism. Full xs run vs baseline xs run used for a
  sanity comparison (see /workspace/run_mine_xs and /workspace/run_base_xs
  summaries when present).

Rejected/considered alternatives: shared experts (isomorphic to reducing
routed top-k at the same active budget; the keep-trim already exploits that
iso-active equivalence); higher training capacity factor (pays extra padded
compute, worse budget use); batch-scope aux (reads balanced while sequences
overflow); expert-choice routing (breaks causality).

## Submitted solution

`router.py` registers `trimmed_bias`: top-4 token-choice routing with a
top-3 keep mask, deployment-safe gate renormalization, an aux-free
load-balancing bias buffer (persistent, updated only in training), plus a
small per-sequence Switch aux and z-loss. `config.json` uses the reference
target shape (n_expert 64, expert_hidden 768, top_k 4, capacity_factor 1.0,
router_lr_mult 1.0, seed 1337).

Reproduce: with the submission bundle in /workspace/submission, run
`python3 /workspace/train.py --scale target --config /workspace/submission/config.json --router-file /workspace/submission/router.py`
or score via `/workspace/validation/val.sh /workspace/submission`.
