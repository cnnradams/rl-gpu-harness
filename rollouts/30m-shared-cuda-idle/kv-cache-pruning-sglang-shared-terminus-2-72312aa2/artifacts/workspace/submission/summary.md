## Experiments

The shipped baseline is a correct heavy-hitter implementation that is slower
than serving the full cache, for two independent reasons, and the work here
was removing both:

1. **Slots were never returned to the allocator.** The baseline compacted
   `req_to_token` and shrank `seq_lens`, so attention read less KV per step,
   but every slot the request had ever touched stayed allocated. Memory per
   request never fell, so batch capacity and most of the decode traffic were
   untouched. Measured effect in its own log: speedup 0.60 regardless of the
   retained ratio.
2. **Scores were recomputed in PyTorch.** Every sixteenth decode step the
   baseline gathered the resident keys per layer and ran a second attention
   pass with a device-to-host sync, and the scheduler stalled on it at every
   budget size.

Approaches tried:

- *Free evicted slots at eviction time.* The request's row maps absolute
  position to slot, so one read of the row per eviction event yields the
  doomed slots; they go straight back to
  `token_to_kv_pool_allocator.free()`. The stock release path frees
  `row[protected:kv_committed_len]` plus `row[committed:allocated]`, so
  decrementing `req.kv_committed_len` and `req.kv.kv_allocated_len` by the
  evicted count at the same moment keeps the final free exact: every slot
  still owned appears once in the freed prefix and nothing else is freed.
  This removes the baseline's release-time row rewrite entirely.
- *Score only at prefill.* Prefill is excluded from the timed window, so the
  expensive part of scoring is free there. The prompt is where nearly all
  the retention mass lives (heavy hitters are overwhelmingly prompt tokens,
  and the decode window is protected by the recent window anyway), so
  refreshing scores during decode adds cost and little signal. Decode steps
  only admit into the recent window and trigger batched eviction; no
  rescoring happens in the decode loop at all.
- *Positions, not slot lists, in the policy.* The runtime tracks absolute
  positions and reads the row once per eviction event instead of mirroring a
  per-request slot list, so the steady-state decode step costs one length
  check per request.

What failed: the baseline's own periodic-rescoring variant (kept briefly
during bring-up) reproduced its 0.6x slowdown, which confirmed the scoring
pass, not the eviction, was the first-order cost.

## Submitted solution

`sglang_kv_prune.patch` adds `h2o_policy.py` and `h2o_runtime.py` under
`python/sglang/srt/mem_cache/` and a two-line activation hook in
`model_runner.py`:

- `h2o_policy.py`: the retention rule. Mean attention per observing query
  (so long-visible prefix positions are not flattered by exposure), the
  heavy budget spread over eight depth buckets (so the kept set cannot
  collapse into the prompt prefix), a 50/50 heavy/recent split, and batched
  eviction with slack so ranking runs once per `budget/16` admissions rather
  than every step.
- `h2o_runtime.py`: wraps the Triton attention backend to run one extra
  attention pass per layer at prefill and fold its column sums into the
  scores; evicts at the top of `ScheduleBatch.prepare_for_decode` (before
  the step's slot is allocated, so `seq_lens` after prepare never exceeds
  the budget); compacts the request's `req_to_token` row, frees doomed
  slots to the allocator, and keeps `kv_committed_len` /
  `kv.kv_allocated_len` exact so the stock release path frees each owned
  slot exactly once; overrides decode rotary positions back to absolute
  (eviction compacts storage, not position numbering); and writes the
  `SGLANG_KV_PRUNE_DUMP` self-report only when the variable is set.

The engine reads `SGLANG_KV_PRUNE_ENABLE`, `SGLANG_KV_BUDGET_RATIO` and
`SGLANG_KV_BUDGET_FLOOR`. With pruning disabled the hook returns before
touching anything and the engine is byte-for-byte stock SGLang on that
path.

Reproduce: apply `sglang_kv_prune.patch` to a frozen v0.5.17 checkout with
`git apply` from the repository root; serve with
`SGLANG_KV_PRUNE_ENABLE=1 SGLANG_KV_BUDGET_RATIO=<r> SGLANG_KV_BUDGET_FLOOR=128`.
