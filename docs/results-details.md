# Detailed measurements and pilot results

[Read the main report](results.md). This appendix preserves the full measurements, confidence intervals and quantity pilot.


We tested whether giving an agent a separate GPU tool reduces allocated GPU time while preserving the quality of its work. Agents used Terminus-2 with Kimi-K3 and H100 GPUs. Each attempt was graded independently after the agent finished.

Allocation and performance are separate measurements. The tables report task-specific differences; they do not estimate a single effect across tasks.

## 1. Where should the agent's GPU live?

We compared three setups on five tasks, with **eight attempts per setup per task and a 60-minute agent limit**:

- **GPU-in-sandbox:** the agent's own machine has a GPU. Ordinary shell commands can use it, and it remains allocated while that machine runs.
- **Manual file transfer:** the agent works on a CPU machine and runs GPU commands on a separate machine. It explicitly pushes input files and pulls outputs.
- **Shared workspace:** the agent works on a CPU machine. Separate GPU machines mount the same `/workspace` Volume. The agent starts a GPU program with `gpu run`, can run additional commands with `gpu exec`, and can explicitly synchronize files with `gpu sync`.

matrix multiplication and LayerNorm tasks optimize kernels; performance is speedup over the reference implementation. mixture-of-experts training trains a mixture-of-experts model; lower validation loss is better. Fine-tuning trains a model to answer GSM8K math questions; higher accuracy is better. Serving modifies a KV-cache pruning implementation; a positive score requires a valid implementation and a throughput improvement.

Each cell gives **mean allocated GPU minutes / median performance**.

| Task | GPU-in-sandbox | Manual GPU: push/pull | Shared Volume: run/exec |
| --- | ---: | ---: | ---: |
| matrix multiplication speedup | 28.05 / 4.20× | 23.35 / 4.20× | 30.46 / 4.43× |
| LayerNorm speedup | 39.82 / 26.49× | 32.89 / 27.47× | 27.85 / 27.09× |
| mixture-of-experts training validation loss | 39.29 / 3.7812 | 38.55 / 3.7741 | 34.23 / 3.7825 |
| GSM8K fine-tuning accuracy | 54.87 / 15.62% | 53.88 / 8.20% | 48.08 / 8.79% |
| Serving: positive-score outcomes | 57.02 / 2 of 8 | 44.73 / 1 of 8 | 37.57 / 0 of 8 |

Serving had a median score of zero in every setup, so its table reports successful outcomes instead. One LayerNorm grade and two fine-tuning grades are missing; they are not counted as zero. Only 20 of the 24 fine-tuning attempts passed every benchmark audit check, so those accuracy figures need caution.

Manual serving used 12.29 fewer GPU minutes than GPU-in-sandbox (95% interval: 6.51–18.07 fewer; adjusted p=0.0076). This was the only interface allocation difference that passed correction for multiple comparisons. Its quality was not shown to be equivalent.

## 2. Can resource policies make the shared workspace more efficient?

We compared the following changes against shared-workspace access without a resource policy. Each had **six attempts per task and a 30-minute agent limit**, on matrix multiplication and mixture-of-experts training:

- **No resource policy:** the agent chooses how long to keep GPU machines running.
- **Efficiency request:** the prompt adds only “Use GPU resources efficiently.” There is no enforced time budget.
- **GPU-time budget:** GPU access ends after exactly **663.793 allocated GPU-seconds for matrix multiplication** or **650.346 for mixture-of-experts training**. These allowances were half the corresponding pilot median. CPU work can continue until the agent deadline.
- **Release after five minutes without CUDA:** a GPU machine is stopped after five minutes with no CUDA process. This detects process presence, not actual GPU utilization.
- **Run-only commands:** `gpu run` starts a GPU machine for one program; the machine stops when that program exits. `gpu exec` is unavailable. The shared workspace carries files between runs.

Values are **means [95% confidence intervals]**. Compare these rows with each other; the preceding interface experiment allowed twice as much agent time.

| Shared-workspace setup | Agent limit; n/task | matrix multiplication GPU min | Speedup | Correct | mixture-of-experts training GPU min | mixture-of-experts training loss |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| No resource policy | 30 min; 6 | 18.44 [7.40, 29.48] | 3.86 [2.87, 4.84]× | 6/6 | 10.26 [3.04, 17.48] | 3.7885 [3.7755, 3.8016] |
| Efficiency request | 30 min; 6 | 20.84 [10.38, 31.30] | 3.96 [3.67, 4.25]× | 6/6 | 13.24 [5.68, 20.80] | 3.8003 [3.7789, 3.8218] |
| GPU-time budget | 30 min; 6 | 9.46 [5.91, 13.00] | 2.89 [0.92, 4.86]× | 5/6 | 7.96 [5.52, 10.40] | 3.7883 [3.7656, 3.8111] |
| Release after five minutes without CUDA | 30 min; 6 | 24.14 [20.40, 27.88] | 4.23 [4.03, 4.43]× | 6/6 | 14.27 [9.32, 19.22] | 3.7954 [3.7718, 3.8190] |
| Run-only commands | 30 min; 6 | 5.15 [1.86, 8.44] | 3.22 [1.39, 5.05]× | 5/6 | 9.46 [1.26, 17.66] | 4.4131 [2.7989, 6.0273] |

The GPU budget and run-only setup used less matrix multiplication allocation in the observed means, but neither allocation difference passed correction for multiple comparisons. Wide performance intervals mean we cannot conclude that either preserves quality. Six successes out of six also have substantial uncertainty: the 95% correctness interval is approximately 61–100%.

### What the traces explain

- **The budget encouraged real on/off use in mixture-of-experts training.** All six budget agents used 2–7 separate GPU allocations. They spent an average of 22.69 elapsed minutes working, versus 19.74 without a policy, while allocating less GPU time. Savings were not simply earlier agent termination.
- **The budget kernel failure includes a workspace rollback.** The original implementation had a dtype error, but the agent later wrote a replacement. After a successful sync, the original file reappeared and was graded. See the reconstruction below; attributing the failure only to the agent’s original bug is incorrect.
- **The run-only matrix multiplication failure reached the agent deadline with a broken submission.** It omitted bias. A fallback passed local checks at 1.07× but was never copied into the submitted `solution.py`.
- **One poor mixture-of-experts training submission drove up the run-only mean loss.** Its loss was 7.55; the median was 3.78232, close to 3.78285 without a policy. Its training log reported “loss 7.3969 aux 487.8959.”
- **Run-only avoided sleeping GPU keepers in all 12 attempts.** Agents used programs as the machine's main command. Two workspace refresh conflicts occurred and both recovered.

## Quantity pilot: evidence about uptake, not established scaling

This smaller pilot allowed either one or four machines, with three attempts per cell and a 60-minute agent limit. Each task used the same policy on both sides. Fine-tuning used an earlier idle-release policy that also reset its clock on GPU command activity. Values are **median GPU minutes / median performance**.

| Task / policy | Up to one box | Up to four boxes |
| --- | ---: | ---: |
| matrix multiplication / efficiency request | 27.77 / 4.31× | 33.78 / 4.21× |
| LayerNorm / GPU budget | 18.88 / 27.86× | 17.81 / 26.32× |
| mixture-of-experts training / GPU budget | 10.86 / loss 3.7978 | 7.92 / loss 3.7814 |
| Fine-tuning / idle release | 45.43 / 42.97% | 35.20 / 62.89% |
| Serving / no resource policy | 44.17 / 0.565× | 43.09 / 0.653× |

The pilot holds policy fixed within each task, but three attempts per setting and little actual concurrency are insufficient to establish a benefit from more machines. Policies differ across tasks, so pooling tasks would also obscure which policy was tested. None used four ready machines concurrently; only four of fifteen attempts in the four-machine condition overlapped two or more. All serving submissions scored zero because their throughput remained below the reference.

## Reading the measurements and reviewing attempts

GPU minutes count allocated time, including idle time. They exclude provisioning, independent grading and model inference. We did not systematically measure CUDA-active time, so lower allocation alone does not demonstrate better utilization. Confidence intervals describe uncertainty in individual means; they do not prove equivalent quality.

The interface comparison uses gVisor for GPU-in-sandbox and CPU VMs for the separate-GPU setups. These are comparisons of complete agent setups.

The tables above cover 180 attempts from the completed interface, resource-policy and run-only comparisons, plus the separate quantity pilot. The expanded comparison uses a matched 30-minute limit across all five tasks. Its collected episodes are in [rollouts](../rollouts/); the folder's README records snapshot coverage. They are not pooled into the tables above.

To review trajectories, terminal recordings, grading evidence and GPU events:

```bash
uv run --extra harbor harbor view rollouts --jobs
```

## Trace investigation: one-program machines

The comparison below retains every assigned outcome. Looking at medians and correct submissions is a diagnostic for why means differ, not a reason to exclude failures.

### Matrix multiplication

Mean speedup was 3.21970× for one-program machines and 3.85544× for ordinary shared-file access. Medians were 4.13179× and 4.17633×. The five correct one-program submissions averaged 3.86364×; the sixth scored zero.

The failed attempt (`0a59da9f`) omitted bias from its optimized GPU path while retaining it in the CPU fallback. It suspected precision and fresh environments instead: “Since each 'gpu run' is a separate box, maybe box environments differ.” A fallback passed five local correctness checks but was never promoted to the submitted file before the deadline. Both setups had two agent timeouts, so more timeouts do not explain the difference.

Four one-program submissions achieved speedups above four. Its slower correct submission (`51b4bc61`) used three separate matrix multiplications and voluntarily stopped optimizing. The control also had a slow submission using a similar strategy. This supports variation in algorithm choice rather than a universal optimization limitation.

There was a measured startup cost: one-program agents launched 9–20 machines each. Their recorded request-to-ready delays totaled 235 seconds per attempt on average, versus 16.5 seconds in the control. This consumes the agent deadline while remaining outside allocated GPU time. The traces do not separately measure compilation time or establish that startup delays caused the incorrect submission.

### Mixture-of-experts training

Mean loss was 4.4131 with one-program machines and 3.7885 in the control. Median loss was 3.782316 and 3.782848. One valid but poor model (`9282b9ac`) scored 7.55217; the remaining five averaged 3.785268. That five-model average is only a sensitivity check, not the study result.

The poor submission's saved training code amplifies its balancing-gradient term by a factor of 8,192 relative to the baseline normalization. Its grader training log reports “loss 7.3969 aux 487.8959.” This is a concrete algorithmic defect consistent with poor training, but no ablation establishes that it is the sole cause.

Five of six one-program training agents received “Unknown command ID for this session,” compared with none of the six controls. They used a request identifier where a command identifier was required. In the poor attempt, repeated failures led the agent to believe the machine had died; it then stopped the live baseline training job. The four other affected attempts still produced ordinary losses. Thus the interface confusion is systematic, while its responsibility for the one bad final model remains unproven. Controls had their own file and workspace-refresh errors.

These observations support fixing the demonstrated identifier confusion before repeating a matched comparison. They do not establish an inherent training penalty from ephemeral GPU machines.

Another training attempt (`dc92e6a7`) compared two designs against the baseline and saw both lose, then submitted a third design after only a short test. Its trace said “still worse than baseline at xs” before proposing a “New final design.” The final checks established that the code ran, not that it trained better. This is evidence of weak experiment selection as well as the separate identifier friction.

## Serving: what a positive reward means

A positive reward is not an accuracy-preserving success. Reward combines speed gains with retained accuracy. The two positive GPU-in-agent-machine submissions retained 68% and 38% of reference accuracy; the positive manual-file submission retained 64%. Counting those as three successes hides substantial accuracy loss. Of the 24 submissions, one shared-file submission was invalid; the other 23 had valid native grades. Most valid submissions were slower than the reference.

Below are native means [95% confidence intervals], including the grader's zero values for the invalid submission. Throughput speedup and accuracy retention are distinct from the composite reward and should be read together.

| Setup | Throughput speedup | Reference accuracy retained | Composite reward |
| --- | ---: | ---: | ---: |
| GPU in agent machine | 0.92× [0.71, 1.13] | 57.6% [40.8, 74.5] | 0.045 [0.000, 0.116] |
| Separate GPU, manual files | 0.71× [0.52, 0.91] | 55.2% [45.6, 64.7] | 0.023 [0.000, 0.076] |
| Separate GPU, shared files | 0.63× [0.39, 0.88] | 43.5% [20.7, 66.3] | 0.000 [0.000, 0.000] |

The main report uses means with confidence intervals. The older interface table above retains medians as a secondary summary; do not mix its values with the main report’s means.

## Interface performance comparisons

Exploratory two-sided Welch tests compared each separate-GPU setup with GPU-in-sandbox on each of the five tasks, with Holm correction across ten comparisons. None detected a significant performance difference, including degradation. This is not an equivalence or noninferiority result: eight assigned attempts per cell leave wide uncertainty, some grades are missing, and the fine-tuning audit limitations still apply. Serving uses its composite reward and has few positive scores.

## Performance sample counts

The GPU-in-agent-machine setup has seven grades for layer normalization and math fine-tuning; manual-file fine-tuning also has seven. Other performance cells have eight. Missing grades are not counted as zero. Four fine-tuning attempts failed benchmark audit checks, so those accuracy comparisons need caution.

## Budget kernel failure: corrected trace attribution

In attempt `a7f0eec9`, the original 1,770-byte `solution.py` failed with “self and mat2 must have the same dtype.” After the budget expired, trajectory step 22 wrote a 4,438-byte replacement and `ls` confirmed that size. The following `gpu sync` reported both agent commit and reload completed. Step 26 then printed the original implementation again, with no intervening explicit rewrite in the recorded commands. The saved artifact and grader’s solution hash match that original file.

The original SHA256 is `0d8084310be22b5ef11be146383adf0d10d5fd4838f97abc14164c92153e1840`; the replacement reconstructed from the actual write command is `7f35f597ec6dfe16e3444c84ed5270bd2f24e7df599fc8ffa79636d7b8cb7935`. The original was pushed onto the GPU earlier in the attempt. Existing workspace metadata validates as a Modal Volume v2.

This proves an observed file rollback before grading, not merely a grader collecting the wrong file. Exact attribution to a particular background commit, shutdown commit or reload requires reproduction; the trace does not capture per-file storage commit history. The untested replacement’s correctness is unknown. The native zero remains in the results, with this corrected explanation.
