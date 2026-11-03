## Experiments

All experiments at xs scale (4x384, 64 experts, top-4, 16 devices), trained with
`train.py` and scored at the deployment capacity factor 0.75, seed 1337.

Baselines and ablations (xs val_loss, deploy CF; lower is better):

| run | steps | val_loss | at train CF |
|-----|-------|----------|-------------|
| Switch baseline (aux 0.01, z 0.001, CF 1.0) | 200 | 6.8010 | 6.7958 |
| + DeepSeek-V3 sign-rule bias (bias_lr 0.002, aux 0.01) | 200 | 6.8272 | 6.8271 |
| + causal greedy overflow reroute (aux 0.003) | 200 | 6.8997 | 6.8997 |
| reroute only (bias off, aux 0.01) | 200 | 6.7580 | 6.7533 |
| reroute + gentle bias (bias_lr 0.0005) | 200 | 6.8989 | 6.8986 |
| baseline, full length | 2198 | 4.9832 | 4.9534 |
| reroute only, full length | 2198 | 5.0421 | 5.0096 |
| submitted config (reroute off, bias off), parity check | 200 | 6.8030 | 6.7985 |

Hypotheses tried:
1. Aux-loss-free bias balancing (DeepSeek-V3 sign rule on a persistent
   per-expert bias buffer, updated only in training; the harness restores
   buffers per causality-probe pass, so this is causal and valid). Failed at
   both 0.002 and 0.0005: constant-magnitude sign updates add top-k selection
   noise and underperform the plain Switch aux loss at every horizon tested.
2. Causal greedy overflow rerouting at tight capacity (prefix-count based, in
   the style of the harness's own assign_slots): choices whose expert is full
   are reassigned, in position order, to the expert with the most remaining
   room, and gates are renormalized over choices predicted to survive. Won at
   200 steps (+0.043 over baseline) but LOST at full length (-0.059): once the
   router is well balanced, deploy-time overflow is rare and rerouting tokens
   to arbitrary high-capacity experts with near-uniform gates injects noise
   that dropping avoids. Findings at 200 steps did not transfer to the
   converged regime, and target trains 5x longer than full xs, so the reroute
   was disabled in the final config.
3. Fine-grained experts / shared experts were rejected on cost grounds:
   with 16 devices and traffic charged per extra device per token (~one expert
   pass each), splitting 4x768 active width into more, narrower experts
   raises devices/token well above the current 3.69 and spends budget on
   traffic instead of training steps.

## Submitted solution

Top-4 token-choice routing over the reference 64 experts (expert_hidden 768),
Switch load-balancing loss (coefficient 0.01) measured per sequence over all
choices, ST-MoE z-loss (0.001), capacity factor 1.0, no shared experts, seed
1337. The router (`balanced_reroute`) is a superset of the baseline that also
implements optional bias balancing (`bias_lr`) and optional causal overflow
rerouting (`reroute`); both are disabled in the submitted config because
full-length xs experiments showed they do not beat the tuned baseline at
convergence. Gates are softmax probabilities of the selected experts,
renormalized over choices predicted to survive capacity. Reproduce: train with
`python3 /workspace/train.py --scale target --config config.json
--router-file router.py` from the submission directory.
