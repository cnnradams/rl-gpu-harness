## Experiments
Experiments were run at xs scale (64 experts, top-4, expert_hidden 384,
trained at CF 1.0, scored at deployment CF 0.75) with a full 800-step cosine
schedule, seed 1337. Baseline recipe (top-4 token choice, Switch aux loss
1e-2 per sequence over all choices, ST-MoE z-loss 1e-3) reached val_loss
5.8869 with deployment drop_rate 0.188.

Hypotheses tried (all against the baseline at identical scale/schedule):
1. Hard keep-3: pre-drop every token's 4th choice router-side (keep mask),
   aux loss over kept choices only, gates renormalized. val_loss 5.8890,
   drop_rate 0.190 - no gain: training with only 3 routed experts/token costs
   as much as the tighter capacity fit saves, since the aux loss rebalances
   the kept choices uniformly and overflow persists at CF 0.75.
2. Probability-threshold dropping: keep the 4th choice only when its softmax
   probability >= 0.10. val_loss 5.9072, drop_rate 0.335 - clearly worse:
   dropping weak choices only shrinks the kept set the aux loss balances,
   and the dropped tokens lose output signal.
3. Choice-order rotation: move the weakest (k-th) choice to the front of each
   token's listed choices so harness overflow drops the weakest choice.
   val_loss 5.9324 - worse at this scale; not adopted.

The tuned baseline recipe proved robust: none of the causal router-side
dropping or reordering variants improved on it within the compute budget.

## Submitted solution
The submitted bundle is the baseline recipe: top-4 token-choice routing over
64 experts (expert_hidden 768), Switch load-balancing aux loss (coefficient
1e-2, per routing group over all choices) plus ST-MoE router z-loss
(coefficient 1e-3), trained at the capacity-factor floor 1.0, no shared
experts, seed 1337, router_lr_mult 1.0. Reproduce with
bash /workspace/baseline/baseline.sh   (writes this same bundle), or
python3 /workspace/train.py --scale target --config
/workspace/submission/config.json --router-file /workspace/submission/router.py
