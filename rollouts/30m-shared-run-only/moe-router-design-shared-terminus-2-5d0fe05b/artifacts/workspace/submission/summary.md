# Routing design for the sparse MoE LM

## Experiments

All experiments were run at xs scale (4 layers, width 384, 64 experts of width
384, top-4, capacity factor 1.0 training, deployment evaluation at cf 0.75),
comparing `val_loss` (mean of in-distribution and OOD loss at deployment
capacity). Baseline at this scale: val_loss 4.9234 (val_loss_at_train_cf
4.8916, i.e. a ~0.032 deployment-capacity penalty).

1. **Capacity-aware gate renormalization.** Hypothesis: at deployment cf 0.75,
   ~25% of choices overflow even under perfect balance; renormalizing gates
   over the choices predicted (causally, via `moe_api.assign_slots`) to survive
   capacity should recover the lost signal. Result: val_loss 5.0091, worse even
   at training cf (4.9761). The served/dropped indicator couples each choice's
   gate gradient to the counts of later-serving choices and adds gradient
   variance; the deploy-cf gap was unchanged (0.033). Rejected.
2. **Fine-grained experts (DeepSeekMoE-style): 128 experts of half width,
   top-8.** Same total-sparse and active-parameter envelopes. Result: val_loss
   5.0844, much worse. Top-8 reaches ~6.6 devices/token and the device-traffic
   charge consumed 17% of the compute budget (vs 9% at top-4), cutting training
   tokens far more than finer granularity repaid. Rejected.
3. **Stronger load balancing (aux_coef 0.03).** Hypothesis: more uniform load
   would cut overflow at the tight deployment capacity. Result: val_loss
   5.0074, worse; the stronger auxiliary loss hurt the language-modeling
   objective more than the (unchanged ~0.03) deploy-cf penalty. Rejected.

The shipped baseline recipe was not beaten by any of these variations, so the
submission reproduces it exactly.

## Submitted solution

Top-4 token-choice routing over the reference 64 experts (expert_hidden 768),
softmax gates over the token hidden state (linear router, std-0.02 init,
fp32 logits), gates normalized over the 4 chosen experts, Switch
load-balancing auxiliary loss (coefficient 1e-2) measured per routing group
(sequence) over all 4 choices, and the ST-MoE router z-loss (coefficient
1e-3). Trained at capacity factor 1.0 (the floor), no shared experts,
router_lr_mult 1.0, seed 1337. This is identical in math and configuration to
the tuned canonical baseline; the router class is re-registered under the name
`token_choice_final` to avoid registry collisions. Reproduce with

    bash /workspace/baseline/baseline.sh            # writes an equivalent bundle
    python3 /workspace/validation/val.sh /workspace/submission
