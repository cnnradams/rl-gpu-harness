## Experiments

Starting point: the shipped baseline (`/workspace/baseline`), a naive H2O
implementation that keeps per-request Python score lists, recomputes attention
probabilities in PyTorch every 16th decode step, compacts each request's
`req_to_token` row to enforce the budget logically, but never returns evicted
KV slots to the allocator. Its own notes identify the lost speedup in two
places: (1) logical-only eviction frees no memory, so the KV pool still holds
every token and batching is unchanged; (2) rescoring walks every layer with a
device-to-host sync.

Hypotheses / approaches tried:

- H1: Freeing evicted slots at eviction time is the dominant win. With
  `ChunkCache` (radix cache disabled, as the evaluator fixes it), a finished
  request is freed by reading its `req_to_token` row, so freeing at eviction
  is safe iff the row the release path reads still lists every slot the
  request owns exactly once. The baseline already rewrote the row at release
  (restoring all owned slots); with early freeing that rewrite must restore
  only the slots not already freed, otherwise slots are double-freed and two
  live positions share one slot (an invalid run). Implemented: eviction
  returns the dropped slot ids, `apply_eviction` batches them into one
  allocator `free` per decode step, and the release guard rewrites the row
  with `owned - freed`.
- H2 (baseline's, kept): mean attention per observing query as the retention
  score, heavy budget spread over 8 depth buckets, 50/50 recent/heavy split,
  batched eviction with slack, rescore every 16 steps. These are accuracy-side
  choices; the retained dump and accuracy suite in validation check them.

Evaluation: `/workspace/validation/val.sh` (public seed, ratios 0.3/0.2/0.1,
floor 128) runs the budget, retention, denominator-integrity and fidelity
gates and measures decode speedup vs the same build with
`SGLANG_KV_PRUNE_ENABLE=0`.

## Submitted solution

`sglang_kv_prune.patch` adds `python/sglang/srt/mem_cache/h2o_policy.py` and
`h2o_runtime.py` plus a two-line activation hook in `model_runner.py`.

How it works: per request, a policy object accumulates attention mass per
resident absolute position (summed over layers, heads, queries) and normalises
by the number of observing queries. The budget is
`max(SGLANG_KV_BUDGET_FLOOR, floor(SGLANG_KV_BUDGET_RATIO * prompt_tokens))`,
split evenly between a recency window and heavy hitters chosen per depth
bucket. Eviction runs in `ScheduleBatch.prepare_for_decode` (before lengths
advance), compacts the request's `req_to_token` row, shrinks `seq_lens` /
`seq_lens_cpu`, and — the change from the baseline — immediately returns the
evicted slots to `token_to_kv_pool_allocator`, so the KV pool and per-step
attention reads actually shrink. A release-time guard rewrites the row to
exactly the slots still owned so SGLang frees each slot exactly once. Rotary
positions stay absolute via a `positions` override in `model_runner.forward`.
Scores refresh every 16th decode step by recomputing post-softmax mass for the
resident keys; eviction runs every step off the last scores, so the budget
holds at every decode step. `SGLANG_KV_PRUNE_DUMP` writes one JSON record per
request per decode step and costs nothing when unset. With
`SGLANG_KV_PRUNE_ENABLE=0` the install hook returns before touching anything,
so the disabled arm is byte-for-byte stock behaviour.

Reproduce: `git apply` the patch onto a pristine v0.5.17 tree (it contains the
full diff, generated with `git -C /workspace/sglang diff -- python/sglang`),
then serve with the evaluator's fixed launch and the env vars above.
