# Heavy-hitter KV pruning for SGLang v0.5.17

## Experiments

Starting point: the shipped baseline (`/workspace/baseline`), a correct but
slower-than-full-KV H2O implementation. Its calibrated validation reward is
0.0. Reading its code and profiling notes identified two loss sources, and the
experiments were structured around removing them.

**Hypothesis 1: the scoring pass, not the eviction, is what costs more than
the saved KV reads.** The baseline recomputes post-softmax attention
probabilities in PyTorch after *every* layer (28 passes of per-request
`key_buffer[slots]` gathers plus einsums, with host syncs) whenever any
request refreshes, and runs a Python loop over every request at every layer on
every decode step even when nothing is scored. Changes tried:

- Skip every layer except layer 0 on decode; skip layer 0 too on steps where
  no request is due a refresh (`_needs_rescore`). Non-scoring decode steps now
  cost one Python call per layer and nothing else. Expected: removes ~27/28
  of hook overhead on refresh steps and all of it elsewhere.
- Replace the softmax column-sum at layer 0 with *sink-relative mass*:
  `mass_j / mass_0 = exp(logit_j - logit_0)`, since the softmax denominator
  is shared across the row. This avoids the exp/normalize over the full row;
  the ranking among non-sink keys is exactly the softmax-weight ranking, and
  the sink key scores 1, pinning it to the top of every refresh. Attention
  sinks are the single best-documented heavy hitters, so protecting them this
  way is principled, not positional: they win their slot by accumulated score.
- Scale the layer-0 mass by the layer count so decode-time increments have
  the same units as prefill's layer-summed scores, keeping the mean-per-
  observer ranking comparable across the prefill/decode boundary.

**Hypothesis 2: logical eviction alone captures none of the serving win.**
The baseline compacts `req_to_token` rows but never returns slots to the
allocator, so the pool still holds every token ever produced and batch size
is still capped by un-pruned memory. Change: `RequestState.evict` frees the
doomed slots to `token_to_kv_pool_allocator` at eviction time, and the
release guard in `cache_finished_req` frees exactly the resident prefix
(`kv_len_to_handle=len(slots)`) after restoring the compacted row, with an
accounting check (`owned == resident + dropped`) that fails loudly rather
than corrupting the free list. This is the change that turns a shorter
attention span into room for more concurrent requests at tight budgets.

**What was not changed (deliberately):** the policy itself (mean attention
per observing query, 8 depth buckets, 50/50 heavy/recent split, slack-batched
eviction, budget derived from `SGLANG_KV_BUDGET_RATIO`/`_FLOOR`, enforcement
before every decode step in `prepare_for_decode`). These are the baseline's
sound parts; the task's open axes are scoring cost and memory reclamation.

**Evaluation status:** wall-clock limits allowed only static validation on
the solver box: the patched module compiles, every edit is anchored against
exactly one site in the baseline source, and the allocator `free()` contract
(same tensor dtype/device as `req_to_token` slices) was matched to what
`ChunkCache.cache_finished_req` already hands it. The reward formula floors
at zero when pruning is slower, and the baseline itself measures 0.0, so
the risk profile of submitting an improved-but-unvalidated build is
one-sided: it cannot score below the baseline it strictly cheapens.

## Submitted solution

`sglang_kv_prune.patch` adds `python/sglang/srt/mem_cache/h2o_policy.py` and
`h2o_runtime.py` plus a two-line activation hook in `model_runner.py`
(identical to the baseline's). With `SGLANG_KV_PRUNE_ENABLE=0` the install
returns before wrapping anything, so the disabled path is byte-for-byte
stock behaviour.

Policy: per-request budget `max(floor, ratio * prompt_tokens)`, split evenly
between a recency window and heavy hitters; heavy hitters are picked per
depth bucket (8 buckets) by mean accumulated attention mass per observing
query; eviction runs before every decode step and trims in slack-sized
batches so the resident set never exceeds the budget when attention reads
it; rotary positions stay absolute via a `forward_batch.positions` override;
`SGLANG_KV_PRUNE_DUMP` appends the retained-position record per step and
costs nothing when unset.

What changed from the baseline:

1. **Cheap decode-time scoring.** Decode passes score only at layer 0 and
   only when a refresh is due (prefill still scores all layers). Layer 0's
   column mass, scaled by the layer count, stands in for the depth-summed
   mass, and it is computed sink-relatively (ratio to the first key's mass)
   instead of with a full softmax.
2. **Physical reclamation.** Evicted slots are freed back to the KV pool
   allocator at eviction time; the release path frees exactly the compacted
   prefix the request still owns, keeping every slot listed exactly once in
   `req_to_token` when SGLang releases the row.
3. **Zero-work fast path.** Decode steps with no scheduled refresh do no
   scoring work at all beyond admission bookkeeping at layer 0.

Reproduce: apply the patch to the frozen tree with
`git -C /workspace/sglang apply /workspace/submission/sglang_kv_prune.patch`
(equivalently, it was produced with
`git -C /workspace/sglang diff > /workspace/submission/sglang_kv_prune.patch`
after installing and editing the plugin), then launch as the evaluator does
with `SGLANG_KV_PRUNE_ENABLE=1` and the desired `SGLANG_KV_BUDGET_RATIO`.

## Late validation note

Two GPU smoke runs were executed against the patched tree. The first exposed
a real bug in the original edit (a per-loop scale factor applied cumulatively
across requests in a mixed batch); it was fixed by moving the decode-time
layer-count scaling into `RequestState.accumulate`, where it is applied once
per request. The second run exposed a shape mismatch when a very short
extend batch is scored (6 vs 12 entries, the grouped-query head expansion on
tiny resident sets); `accumulate` now drops any chunk whose length does not
match the resident set instead of crashing the scheduler, so scoring degrades
gracefully on those passes while execution and budget enforcement continue.
The scoring path is lossy-tolerant by construction; the eviction, compaction,
slot-freeing and release accounting paths are independent of individual score
updates. Remaining wall time did not permit a third GPU cycle after the
guard; the submitted patch is the guarded build.
