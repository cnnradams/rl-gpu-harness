# Giving agents access to GPUs

We tested whether agents can use less GPU time without doing worse work by accessing a separate GPU machine instead of having a GPU in their own machine.

All agents used Kimi-K3 through the Terminus-2 harness, with H100 GPUs and independent grading.

We measured **performance** and **allocated GPU time** separately. Allocated time includes an idle GPU; it excludes machine startup, grading and model inference. We did not measure how much of that time the GPU was actively computing.

## What the agent had to do

**Matrix multiplication:** given a working PyTorch program that multiplies a large input matrix by learned weights, adds bias, and scales and adds the result, write a faster replacement. The agent could develop and benchmark GPU kernels repeatedly. It submitted Python code; the grader checked numerical correctness and measured speedup over the original program.

**Layer normalization:** given a PyTorch program that normalizes each input by its mean and variance, write a faster GPU implementation with the same output. The agent submitted replacement code. As with matrix multiplication, incorrect output scored zero; correct code was scored by speedup.

**Mixture-of-experts training:** given a language-model training codebase, redesign the router—the code deciding which specialist neural networks process each token, how their outputs are combined, and how training encourages balanced use. The agent could run training experiments, then submitted its router and configuration. The grader trained that design under a fixed compute allowance and measured prediction error on held-out text. Lower loss is better; this tests a training-method design, not a submitted set of trained weights.

**Math fine-tuning:** given the Qwen3-1.7B base language model and evaluation tools, choose training data and a training method, run fine-tuning, and save the best trained model. The goal was to improve its answers to GSM8K grade-school math word problems. The grader loaded the submitted model and measured the fraction of held-out questions it answered correctly. Test questions could not be used for training.

**Model serving:** given a working SGLang server running Qwen3-1.7B, implement a policy that discards less useful cached attention information while retaining useful earlier context. The agent submitted a server-code patch and experiment summary. The grader applied the patch to a clean server and tested generation speed and answer accuracy at fixed cache limits. A valid patch earned a positive score only if it improved throughput; accuracy loss reduced the reward.

## Experiment 1: where the GPU lives

Each setup had eight attempts on each task, with up to one hour for the agent:

- **GPU in the agent's machine.** Ordinary shell commands can use the GPU. It stays allocated while the machine runs.
- **Separate GPU, manual files.** The agent works on a CPU machine, starts a separate GPU machine, and explicitly copies files to and from it.
- **Separate GPU, shared files.** The agent works on a CPU machine. GPU machines share its `/workspace` folder through a Modal Volume. File changes can be synchronized explicitly.

### Performance

All tables show **means [95% confidence intervals]**. The intervals show uncertainty in each average; wide intervals mean the ordering is uncertain. Higher speedup and accuracy are better; lower loss is better.

| Task | GPU in agent machine | Separate GPU, manual files | Separate GPU, shared files |
| --- | ---: | ---: | ---: |
| Matrix multiplication speedup | 4.32× [3.97, 4.67] | 4.14× [3.95, 4.33] | 4.53× [4.04, 5.03] |
| Layer normalization speedup | 25.72× [23.57, 27.87] | 28.21× [26.37, 30.05] | 26.88× [26.34, 27.41] |
| Mixture-of-experts loss ↓ | 3.7809 [3.7666, 3.7952] | 3.8003 [3.7259, 3.8747] | 3.7911 [3.7606, 3.8216] |
| Math fine-tuning accuracy | 21.4% [0.9, 42.0] | 14.9% [0.0, 33.1] | 21.8% [2.9, 40.8] |

**Serving did not produce a reliable quality comparison.** Most patches were slower than the reference. Even the three with positive rewards scored only 38–68% on the grader’s accuracy-retention measure, so calling them “successes” would overstate the result. Full serving measurements remain in the appendix.

### Allocated GPU time

Allocated minutes per attempt; lower is better.

| Task | GPU in agent machine | Separate GPU, manual files | Separate GPU, shared files |
| --- | ---: | ---: | ---: |
| Matrix multiplication | 28.1 [21.8, 34.3] | 23.4 [17.4, 29.3] | 30.5 [21.5, 39.4] |
| Layer normalization | 39.8 [23.9, 55.7] | 32.9 [18.6, 47.2] | 27.9 [16.0, 39.7] |
| Mixture-of-experts training | 39.3 [32.0, 46.5] | 38.5 [34.7, 42.4] | 34.2 [25.3, 43.1] |
| Math fine-tuning | 54.9 [52.5, 57.3] | 53.9 [50.8, 57.0] | 48.1 [42.5, 53.6] |
| Model serving | 57.0 [54.3, 59.8] | 44.7 [39.2, 50.2] | 37.6 [25.7, 49.5] |

### Findings

- **No statistically significant performance degradation was detected with GPU access through a tool.** Serving remains inconclusive for the reasons above.
- **A GPU tool alone does not ensure efficiency.** Agents often kept machines alive with `sleep infinity` or another long sleep, then used `gpu exec` for work. That leaves the GPU allocated between commands, much like keeping it in the agent's own machine.

## Experiment 2: how the separate GPU is used

This experiment used shared files throughout. Each setting had six attempts per task, with a **30-minute agent limit**.

Normally, `gpu run` starts a GPU machine, `gpu exec` runs additional commands on it, and `gpu stop` stops it. We compared:

- **Agent manages the GPU:** no efficiency instruction or enforced time budget.
- **Ask for efficiency:** add only “Use GPU resources efficiently.”
- **Limit GPU time:** enforce a total allocation allowance, then end GPU access while allowing CPU work to continue. The allowance was set to half the corresponding pilot median; exact values are in the appendix.
- **Release unused GPUs:** stop a machine after five minutes without a GPU-using (CUDA) process. Such a process can exist without actively computing.
- **Run one program per machine:** allow `gpu run` and `gpu stop`, but no `gpu exec`. The machine stops when its main program exits. Files persist in the shared workspace.

### Matrix multiplication

Mean allocation and speedup, with 95% confidence intervals. Incorrect submissions count as zero speedup.

| Setting | GPU minutes ↓ | Speedup ↑ | Correct submissions |
| --- | ---: | ---: | ---: |
| Agent manages the GPU | 18.4 [7.4, 29.5] | 3.86× [2.87, 4.84] | 6 of 6 |
| Ask for efficiency | 20.8 [10.4, 31.3] | 3.96× [3.67, 4.25] | 6 of 6 |
| Limit GPU time | 9.5 [5.9, 13.0] | 2.89× [0.92, 4.86] | 5 of 6 |
| Release unused GPUs | 24.1 [20.4, 27.9] | 4.23× [4.03, 4.43] | 6 of 6 |
| Run one program per machine | 5.2 [1.9, 8.4] | 3.22× [1.39, 5.05] | 5 of 6 |

### Mixture-of-experts training

Mean allocation and validation loss, with 95% confidence intervals. All submissions were valid.

| Setting | GPU minutes ↓ | Validation loss ↓ |
| --- | ---: | ---: |
| Agent manages the GPU | 10.3 [3.0, 17.5] | 3.7885 [3.7755, 3.8016] |
| Ask for efficiency | 13.2 [5.7, 20.8] | 3.8003 [3.7789, 3.8218] |
| Limit GPU time | 8.0 [5.5, 10.4] | 3.7883 [3.7656, 3.8111] |
| Release unused GPUs | 14.3 [9.3, 19.2] | 3.7954 [3.7718, 3.8190] |
| Run one program per machine | 9.5 [1.3, 17.7] | 4.4131 [2.7989, 6.0273] |

### Findings

- **Asking for efficiency did not save GPU time in these samples.** Mean allocation was higher for both tasks.
- **Every budgeted training agent switched GPUs on and off across several jobs while continuing to work.** Their mean loss was similar to the control. The kernel failure involved a workspace rollback, so it is not a clean measure of the budget’s effect on quality.
- **The unused-GPU release policy did not save allocation in either task.** Both observed means were higher than the control.
- **One-program machines had the lowest matrix-multiplication allocation.** Five of six kernels were correct. Training's worse mean loss came from one poor submission, examined below.

The confidence intervals remain wide. None of these allocation differences passed the correction for multiple comparisons.

## What the traces explain about failures

In the one-program comparison, one bad submission in each task drives most of the mean performance gap. Reviewing these attempts and the failed budget attempt revealed both interface problems and submission mistakes.

### Problems directly connected to the interface

- **A shared-workspace rollback lost a replacement solution.** In the budgeted kernel attempt, the agent wrote new code after GPU access ended. `gpu sync` reported success, but the old broken file reappeared and was graded. The exact rollback mechanism is unresolved; the replacement was never GPU-tested.
- **Repeated startup reduced time available for experiments.** Matrix-multiplication agents spent about four minutes waiting for fresh GPU machines, versus 17 seconds in the control. This is an observed cost of the setup; its effect on final correctness is not isolated.
- **Two different identifiers confused agents.** Five of six training agents passed a request identifier where a command identifier was expected and received “Unknown command ID for this session.” The worst attempt interpreted that as “The box is gone,” although its training job was still running. It then stopped the machine, losing a baseline comparison. This directly connects the interface to wasted work, though not to the final model's bad training objective.

### Failures that look like run-to-run variation, without an established interface cause

- **The failed matrix-multiplication kernel omitted bias.** Its CPU fallback included bias, hiding the error in CPU checks. The agent pursued precision and environment explanations instead: “maybe box environments differ.” It tested a correct fallback but never made it the submitted file before the deadline. One incorrect kernel pulled down the average; the trace does not show that run-only access caused its bug.
- **The bad training design overwhelmed its main learning objective.** Its code amplified a term intended to balance expert use by a factor of 8,192. The grader log reported “loss 7.3969 aux 487.8959.” The agent submitted after only a short test, saying “The design is sound,” despite having time remaining. This is evidence of a faulty objective and inadequate validation. The lost baseline experiment may have contributed, but the traces do not establish that connection. The remaining submissions trained normally.
- **A slower correct kernel chose a slower algorithm.** It performed three separate matrix multiplications and voluntarily stopped optimizing. A control submission chose a similarly conservative method. That strategy was not specific to one-program machines.

These are diagnostic comparisons, not exclusions: the reported means retain every failure. The practical findings are to fix the workspace rollback, reduce startup overhead and remove identifier ambiguity; a rerun would test whether those changes improve outcomes. The present traces do not justify treating the entire gap as either an interface effect or noise.

## Shared files and manual copying

The 30-minute follow-up traces show a practical advantage for shared storage in training. Two manual-copy attempts ran out of time while transferring their trained models. Another spent roughly ten minutes compressing and downloading its checkpoint.

Shared storage removes that transfer step. The reviewed run-only agents launched programs against `/workspace` without explicit pushes or pulls, making files available across fresh GPU machines. However, shared storage also produced refresh conflicts and the file rollback described above.

This supports shared storage for large training artifacts. It does not establish an agent preference or isolate the benefit for run-only access: the manual-copy comparison also allowed `gpu exec`, so both file handling and command access differed.

## Evidence and limits

Confidence intervals use Student’s t distribution and describe each mean separately; they are not adjusted for multiple comparisons. Bounds are clipped to the metric’s possible range. Missing grades are excluded from performance averages, not replaced by zero. Similar averages do not establish equal performance. [Detailed measurements](results-details.md) include exact GPU-time allowances and supporting trace observations.

We also piloted allowing one versus four GPU machines. That was a small test of whether agents would use the higher limit: most did not use machines concurrently, and none used four at once. It does not establish a scaling benefit, so it is excluded from the main comparisons.

The tables describe the completed one-hour interface and half-hour policy experiments. A separate expanded study matches all five tasks at 30 minutes; it is not pooled into these tables. Its collected episodes and snapshot coverage are in [rollouts](../rollouts/).

To inspect the collected trajectories and terminal recordings:

```bash
uv run --extra harbor harbor view rollouts --jobs
```
