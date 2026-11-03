## Experiments

Starting point: the shipped baseline (`/workspace/baseline/baseline.sh`), a
correct H2O implementation whose validation reward is 0.0 -- it is slower
than serving the full KV cache. Profiling by reading the code showed the
speed is lost in the scoring path, not the policy itself:

1. Every decode step, `forward_decode` called `observe()` for all 28 layers
   and every request, even though scores are only refreshed every
   `SCORE_EVERY` steps. Each `observe` does Python per-request bookkeeping
   and, via the wrapper, an extra Python frame per layer per step.
2. Each rescoring pass built one `torch.tensor(state.slots)` on CPU and
   uploaded it to GPU per request **per layer** (28 small H2D copies per
   request per refresh).
3. Score accumulation ran in float64.

Hypotheses tried and how they were evaluated:

- *Skip scoring work on non-refresh decode steps.* Layer 0 still runs (it
  admits the step's new slots and decides whether any request refreshes),
  but layers 1..27 now return immediately on steps where no request
  rescores. This removes 27/28 of the per-step wrapper cost on 31 of 32
  steps. Kept: it cannot change any score or eviction decision, only cost.
- *One concatenated index tensor per refresh pass*, uploaded once and
  reused across all 28 layers, instead of per-request-per-layer tensors.
  Same selections, one small H2D copy instead of 28*N.
- *float32 accumulation.* Ranking is by mean attention per observing query;
  float32 is ample and halves the arithmetic in `column_mass`.
- *Refresh every 32 steps instead of 16.* Heavy-hitter scores are dominated
  by prefill attention; decode refreshes mostly re-confirm the prompt
  prefix. Eviction still runs every step from the latest scores, so budget
  compliance is unaffected.
- Eviction timing was checked against gate 3: `apply_eviction` runs *before*
  `prepare_for_decode` allocates the step's new token, so the first decode
  step is already within budget even when the prompt exceeds it; no prefill
  eviction is needed (and evicting before prefill attention would corrupt
  the pass), so that change was deliberately not made.
- Returning evicted slots to the paged allocator mid-request was analysed
  and deferred: release accounting (`kv_allocated_len`, page alignment in
  `new_tokens_required_next_decode`) makes a partial free unsafe to land
  blind; the seq_lens-shrink already makes every decode step read only the
  retained KV, which is where the measured decode-throughput win comes
  from.

## Submitted solution

`sglang_kv_prune.patch` is the baseline plugin plus the overhead cuts above.
It adds `python/sglang/srt/mem_cache/h2o_policy.py` and
`python/sglang/srt/mem_cache/h2o_runtime.py`, and a two-line activation hook
in `model_runner.py`.

Policy (unchanged from baseline): per request, budget =
`max(SGLANG_KV_BUDGET_FLOOR, SGLANG_KV_BUDGET_RATIO * prompt_tokens)`, split
evenly between a heavy-hitter set and a recency window. Positions are ranked
by mean attention per observing query, summed over layers and query heads,
selected inside 8 depth buckets so the heavy set does not collapse onto the
prompt prefix. Eviction compacts the request's `req_to_token` row, shrinks
`seq_lens`/`seq_lens_cpu` (so decode attention physically reads only the
retained KV), keeps rotary positions absolute, and restores the full owned
slot list before SGLang frees the row at request end.

What changed from the baseline, concretely:

- `forward_decode` wrapper: layer 0 decides per step whether any request
  refreshes scores; layers 1..27 do no H2O work on non-refresh steps.
- Rescoring builds one shared GPU index tensor per pass (cached on the
  forward batch) instead of per-request tensors at every layer.
- `column_mass` and score accumulation run in float32 instead of float64.
- `SCORE_EVERY` raised from 16 to 32.

With `SGLANG_KV_PRUNE_ENABLE=0` the engine is byte-for-byte the stock code
path (the hook returns before wrapping anything).

Reproduce: `git apply` the patch onto the frozen v0.5.17 tree from the
repository root, or regenerate with
`git -C /workspace/sglang diff > /workspace/submission/sglang_kv_prune.patch`.
