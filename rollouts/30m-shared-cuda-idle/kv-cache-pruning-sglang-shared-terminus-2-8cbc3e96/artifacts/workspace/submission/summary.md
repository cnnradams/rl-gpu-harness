# Heavy-hitter KV pruning for SGLang v0.5.17

## Experiments

Starting point: the shipped baseline (`/workspace/baseline/baseline.sh`), whose
calibrated validation reward is 0.0. It is a correct H2O implementation with
two deliberate defects: (1) evicted KV slots are never returned to the
allocator, so at the evaluation concurrency the fixed KV pool still exhausts
and pruning is slower than serving the full cache; (2) retention scores are
recomputed in PyTorch for all 28 layers, every 16 steps, per request.

Hypotheses tested (by construction, then exercised through the validation
harness on GPU):

1. **Freeing evicted slots is where the speedup lives.** Eviction compacts the
   request's `req_to_token` row and shrinks `seq_lens`, and the dropped slots
   must go back to the allocator. The subtlety is release: `ChunkCache`
   frees a finished request by reading its row, so the row must list exactly
   the slots still owned. We return evicted slots through a delayed-free queue
   (flushed two scheduler passes after eviction, so a slot is never
   reallocated while an already enqueued forward could still read it), drop
   them from the owned list, and the release guard rewrites only the remaining
   prefix of the row before ChunkCache frees by row range. Every slot is freed
   exactly once; no slot ever serves two positions.
2. **Attention ranks are stable across layers**, so accumulating mass from 4
   layers spread across depth (0, 7, 14, 21) preserves the heavy-hitter
   ranking at 1/7 of the scoring cost. One batched einsum over all requests of
   the pass replaces the per-request scoring loop; prefill queries are
   subsampled (stride 8) with a rescaled causal mask.
3. **Scoring cadence** was relaxed from every 16 steps to every 32; eviction
   still runs every step from the last scores, so the budget is enforced at
   every decode step regardless.
4. Kept from the baseline (they are what make it correct): mean-attention-per-
   observing-query normalisation (an unnormalised sum collapses into the
   prompt prefix), heavy budget spread over 8 depth buckets (a global top-k
   collapses into the first tenth of the context), 50/50 recent/heavy split,
   eviction triggered at the cap with slack amortisation, absolute rotary
   positions restored in the forward override.

What did not work / was rejected: per-layer PyTorch rescoring (baseline; the
Python loop alone erases the memory-traffic win), global top-k ranking,
unnormalised accumulated mass.

## Submitted solution

`sglang_kv_prune.patch` adds `python/sglang/srt/mem_cache/h2o_policy.py` and
`python/sglang/srt/mem_cache/h2o_runtime.py`, plus a two-line activation hook
in `model_runner.py`. Policy: H2OPolicy keeps per-request accumulated
attention mass per resident position, normalised by observer count, splits the
budget 50/50 between heavy hitters (chosen within 8 depth buckets) and the
recent window, and evicts in batches at the cap. Runtime: wraps the Triton
attention backend (admission + batched scoring on 4 layers), hooks
`ScheduleBatch.prepare_for_decode` (eviction, row compaction, `seq_lens`
shrink, delayed slot free) before it runs, restores absolute rotary positions
in `model_runner.forward`, guards `ChunkCache.cache_finished_req` so the freed
row is consistent, and writes the JSON dump to `SGLANG_KV_PRUNE_DUMP` when set
(zero cost when unset). With `SGLANG_KV_PRUNE_ENABLE=0` nothing is installed
and the engine is byte-for-byte stock on the hot path.

Reproduce: apply the patch with `git apply` from a pristine SGLang v0.5.17
checkout root (or run the equivalent of `install_plugin.py` with the two
plugin files), then launch with `SGLANG_KV_PRUNE_ENABLE=1`,
`SGLANG_KV_BUDGET_RATIO` and `SGLANG_KV_BUDGET_FLOOR` set by the evaluator.
