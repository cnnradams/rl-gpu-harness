# Heavy-hitter KV-cache pruning for SGLang, engine side.
#
# Design (against the frozen v0.5.17 tree, Triton backend, no radix cache,
# no CUDA graphs, single-pass prefill):
#
# * Retention scores are accumulated attention mass per absolute position,
#   normalised by the number of observing queries. They are obtained by a
#   single batched re-computation over the resident keys, run once every
#   SCORE_EVERY decode steps per request (and once at the end of prefill),
#   in bf16 with one kernel launch per layer for the whole batch, instead of
#   the baseline's per-request fp32 pass with a host sync per layer.
# * Eviction runs in the scheduler before prepare_for_decode: the request's
#   req_to_token row is compacted to the survivors, the four length trackers
#   SGLang keeps are shrunk, and -- the change that makes pruning worth
#   running -- the evicted KV slots are handed back to the pool allocator, so
#   the fixed KV pool admits more concurrent requests.
# * Evicted slots are tracked per request. When the request finishes, its row
#   is rebuilt so that ChunkCache frees exactly the slots still resident plus
#   the already-freed evicted ones, each exactly once, in allocation order.
# * Rotary positions stay absolute across compaction: the model forward's
#   positions tensor is rewritten from per-request absolute lengths.
# * With SGLANG_KV_PRUNE_ENABLE unset/0 nothing below is installed and the
#   engine is byte-for-byte on the upstream path.

from __future__ import annotations

import json
import os
import threading
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from sglang.srt.mem_cache.chunk_cache import ChunkCache
from sglang.srt.mem_cache.h2o_policy import H2OPolicy

_LOCK = threading.Lock()
_DECODE_INDICES: List[int] = []
_STATES: Dict[int, "RequestState"] = {}
_DUMP_PATH: Optional[str] = None
_DUMP_SEEN: Dict[int, str] = {}

# Scores are refreshed once per request per SCORE_EVERY decode steps (prefill
# always scores). The ranking uses mean mass per observing query, so a stale
# pass only means the freshest tokens are under-counted; the recent window
# protects them until they are scored.
SCORE_EVERY = 16
# Prefill scoring walks queries in chunks to bound the logits workspace.
SCORE_Q_CHUNK = 1024
# First positions are never evicted (attention sink).
SINK_TOKENS = 16


def enabled() -> bool:
    return os.environ.get("SGLANG_KV_PRUNE_ENABLE", "0") == "1"


def budget_ratio() -> float:
    return float(os.environ.get("SGLANG_KV_BUDGET_RATIO", "0.20"))


def budget_floor() -> int:
    return int(os.environ.get("SGLANG_KV_BUDGET_FLOOR", "128"))


def budget_for(prompt_tokens: int) -> int:
    return max(budget_floor(), int(budget_ratio() * prompt_tokens))


def heavy_ratio() -> float:
    return 0.5


class RequestState:
    """Per-request policy state plus the KV slots backing each resident position."""

    def __init__(self, request_id: str, prompt_tokens: int, device) -> None:
        self.request_id = request_id
        self.policy = H2OPolicy(budget_for(prompt_tokens), heavy_ratio(), device, SINK_TOKENS)
        self.slots: List[int] = []
        # Slots still owned but no longer resident, in allocation (eviction)
        # order. Freed back to the allocator at the next scheduling point.
        self.evicted: List[int] = []
        self.absolute_len = 0
        self.decode_step = 0
        self.pending_mass: Optional[torch.Tensor] = None
        self.pending_observers: Optional[torch.Tensor] = None
        self.pending_seq_len: Optional[int] = None

    def should_rescore(self) -> bool:
        return self.decode_step % SCORE_EVERY == 0

    def admit(self, new_slots: List[int]) -> None:
        fresh = [int(s) for s in new_slots]
        self.policy.append(len(fresh))
        self.slots.extend(fresh)
        self.absolute_len += len(fresh)

    def fold_scores(self) -> None:
        if self.pending_mass is not None:
            self.policy.add_column_mass(self.pending_mass, self.pending_observers)
            self.pending_mass = None
            self.pending_observers = None

    def evict(self) -> bool:
        """Trim to the budget; record the dropped slots as evicted."""
        doomed = self.policy.evict()
        if not doomed:
            return False
        doomed_set = set(doomed)
        self.evicted.extend(self.slots[i] for i in doomed)
        self.slots = [s for i, s in enumerate(self.slots) if i not in doomed_set]
        return True


def _state(key: int, request_id: str, prompt_tokens: int, device) -> RequestState:
    with _LOCK:
        state = _STATES.get(key)
        if state is None or state.request_id != request_id:
            state = RequestState(request_id, prompt_tokens, device)
            _STATES[key] = state
        return state


def release(key: int) -> None:
    with _LOCK:
        _STATES.pop(key, None)


def _record_pass(plan, key_buffer, query, num_q_heads: int, head_dim: int, scale: float):
    """One batched scoring pass for every due request, at a single layer.

    For each due request the resident keys are gathered once and the
    post-softmax attention mass each receives is accumulated, summed over the
    query heads of its KV-head group. Decode batches share one padded
    [batch, kv_heads, kvmax] matmul; prefill walks the new queries in chunks
    with a causal mask. All arithmetic is in the KV cache's own precision
    with fp32 softmax/normalisation, and everything the per-request loop
    touches on host was copied once per forward pass.
    """
    if not plan:
        return
    q_all = query.view(-1, num_q_heads, head_dim)
    if plan[0]["decode"]:
        q_idx = torch.cat([p["q_idx"] for p in plan])
        slots = torch.cat([p["slots_t"] for p in plan])
        which = torch.cat([p["which_t"] for p in plan])
        lens = [int(p["len"]) for p in plan]
        kvmax = max(lens)
        padded = torch.zeros(len(plan), kvmax, dtype=torch.long, device=q_all.device)
        for i, p in enumerate(plan):
            padded[i, : p["len"]] = p["slots_t"]
        keys = key_buffer[padded.reshape(-1)].reshape(
            len(plan), kvmax, key_buffer.shape[1], key_buffer.shape[2]
        )
        q_sel = q_all[q_idx].view(len(plan), num_q_heads, head_dim)
        kvh = keys.shape[2]
        group = num_q_heads // kvh
        q_g = q_sel.view(len(plan), kvh, group, head_dim)
        k_g = keys.permute(0, 2, 3, 1)
        logits = torch.matmul(q_g, k_g).float() * scale
        key_mask = (
            torch.arange(kvmax, device=q_all.device).view(1, 1, 1, kvmax)
            >= torch.tensor(lens, device=q_all.device).view(-1, 1, 1, 1)
        )
        logits = logits.masked_fill(key_mask, float("-inf"))
        probs = torch.softmax(logits, dim=-1)
        mass = probs.sum(dim=(1, 2))
        flat = torch.zeros(len(plan) * kvmax, dtype=torch.float32, device=q_all.device)
        flat.index_add_(0, which, mass.reshape(-1))
        for i, p in enumerate(plan):
            m = flat[i * kvmax : i * kvmax + p["len"]]
            st = p["state"]
            delta = torch.zeros_like(st.policy.scores)
            delta.index_add_(0, p["resident_t"], m)
            if st.pending_mass is None:
                st.pending_mass = delta
            else:
                st.pending_mass = st.pending_mass + delta
    else:
        p = plan[0]
        st = p["state"]
        slots_t = p["slots_t"]
        resident_t = p["resident_t"]
        num_q = p["num_q"]
        prefix = p["prefix"]
        q_all = p["q_view"]
        keys = key_buffer[slots_t]
        kvh = keys.shape[1]
        group = num_q_heads // kvh
        k_g = keys.permute(1, 2, 0).unsqueeze(1)
        acc = torch.zeros_like(st.policy.scores)
        obs = torch.zeros_like(st.policy.observers)
        n = slots_t.numel()
        for q0 in range(0, num_q, SCORE_Q_CHUNK):
            q1 = min(num_q, q0 + SCORE_Q_CHUNK)
            chunk = q_all[q0:q1].view(q1 - q0, kvh, group, head_dim)
            chunk = chunk.permute(1, 2, 0, 3)
            logits = torch.matmul(chunk, k_g).float() * scale
            cols = torch.arange(n, device=q_all.device).view(1, 1, 1, n)
            rows = (prefix + q0 + torch.arange(q1 - q0, device=q_all.device)).view(1, 1, -1, 1)
            logits = logits.masked_fill(cols > rows, float("-inf"))
            probs = torch.softmax(logits, dim=-1)
            m = probs.sum(dim=(0, 1, 2))
            acc.index_add_(0, resident_t, m)
            o = torch.zeros_like(st.policy.observers)
            o.index_add_(
                0,
                resident_t,
                torch.clamp(
                    (q1 - q0) - torch.clamp(cols.reshape(-1) - (prefix + q0), min=0).float(),
                    min=0.0,
                ),
            )
            obs = obs + o
        if st.pending_mass is None:
            st.pending_mass = acc
            st.pending_observers = obs
        else:
            st.pending_mass = st.pending_mass + acc
            st.pending_observers = st.pending_observers + obs


def _dump(state: RequestState, position: Optional[int] = None) -> None:
    if not _DUMP_PATH:
        return
    record = {
        "request_id": state.request_id,
        "step": state.decode_step,
        "retained": state.policy.retained(),
        "position": position,
        "seq_len": state.pending_seq_len,
        "absolute_len": state.absolute_len,
    }
    with _LOCK:
        with open(_DUMP_PATH, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")


def apply_eviction(batch) -> None:
    """Evict between decode steps, in the scheduler's own state.

    Runs before ScheduleBatch.prepare_for_decode: compacts each evicting
    request's req_to_token row to the survivors, shrinks the device seq_lens
    and its host mirror (orig_seq_lens keeps the true token count; positions
    are absolute), and returns the evicted slots to the KV allocator so the
    fixed pool admits more requests. prepare_for_decode then allocates this
    step's token at the compacted length, so the resident set never exceeds
    the budget at the point the attention kernel reads it.
    """
    if not enabled() or not getattr(batch, "reqs", None):
        return
    _guard_release(batch.tree_cache)
    global _DECODE_INDICES
    pool = batch.req_to_token_pool
    req_to_token = pool.req_to_token
    indices = [int(r.req_pool_idx) for r in batch.reqs]
    _DECODE_INDICES = indices

    changed = False
    freed: List[int] = []
    for position, req_index in enumerate(indices):
        state = _STATES.get(int(req_index))
        if state is None or not state.evict():
            continue
        kept = torch.tensor(
            state.slots, device=req_to_token.device, dtype=req_to_token.dtype
        )
        req_to_token[int(req_index), : len(state.slots)] = kept
        batch.seq_lens[position] = len(state.slots)
        if getattr(batch, "seq_lens_cpu", None) is not None:
            batch.seq_lens_cpu[position] = len(state.slots)
        freed.extend(state.evicted)
        state.evicted = []
        changed = True

    if changed:
        # Recomputed lazily by ForwardBatch.init_new when None.
        batch.seq_lens_sum = None
    if freed:
        batch.tree_cache.token_to_kv_pool_allocator.free(
            torch.tensor(freed, device=req_to_token.device, dtype=torch.int64)
        )

    mirror = getattr(batch, "seq_lens_cpu", None)
    engine_view = mirror.tolist() if mirror is not None else [None] * len(indices)
    for position, req_index in enumerate(indices):
        state = _STATES.get(int(req_index))
        if state is not None and position < len(engine_view):
            state.pending_seq_len = (
                int(engine_view[position]) if engine_view[position] is not None else None
            )


def _guard_release(tree_cache) -> None:
    """Hand SGLang back a row it can free exactly once per owned slot.

    ChunkCache.cache_finished_req frees req_to_token[row, protected:committed]
    and common.py then frees row[committed:allocated]. Eviction rewrote the
    front of the row and already returned the evicted slots to the allocator,
    so the row is rebuilt before either free: the resident slots first, in
    position order, then the evicted slots in eviction order, then the tail
    never admitted. Each owned slot appears exactly once, in allocation
    order, so the two frees return every slot the request still owns exactly
    once and the free list stays free of duplicates.
    """
    if getattr(tree_cache, "_h2o_release_guarded", False):
        return
    if not isinstance(tree_cache, ChunkCache):
        raise TypeError(
            "H2O eviction compacts req_to_token rows, which only ChunkCache "
            f"(--disable-radix-cache) frees correctly; got {type(tree_cache).__name__}"
        )
    req_to_token = tree_cache.req_to_token_pool.req_to_token
    original = tree_cache.cache_finished_req

    def cache_finished_req(req, is_insert: bool = True, *, kv_len_to_handle: int):
        key = int(req.req_pool_idx)
        state = _STATES.get(key)
        if state is not None and state.request_id == req.rid:
            release(key)
            allocated = req.kv.kv_allocated_len
            resident = len(state.slots)
            evicted = list(state.evicted)
            state.evicted = []
            expected = resident + len(evicted)
            if expected > allocated:
                raise RuntimeError(
                    f"request {req.rid}: {expected} owned slots exceed the {allocated} "
                    "SGLang is releasing"
                )
            order = list(state.slots) + evicted + [0] * (allocated - expected)
            req_to_token[key, :allocated] = torch.tensor(
                order, device=req_to_token.device, dtype=req_to_token.dtype
            )
        return original(req, is_insert, kv_len_to_handle=kv_len_to_handle)

    tree_cache.cache_finished_req = cache_finished_req
    tree_cache._h2o_release_guarded = True


def install(model_runner) -> None:
    """Wrap the attention backend and the model forward. Idempotent."""
    global _DUMP_PATH
    if not enabled():
        return
    _DUMP_PATH = os.environ.get("SGLANG_KV_PRUNE_DUMP") or None
    backend = model_runner.attn_backend
    if getattr(backend, "_h2o_installed", False):
        return
    backend._h2o_installed = True

    num_layers = model_runner.model_config.num_hidden_layers
    req_to_token_pool = model_runner.req_to_token_pool
    token_to_kv_pool = model_runner.token_to_kv_pool
    device = token_to_kv_pool.device

    def observe(layer, query, forward_batch, is_decode: bool) -> None:
        key_buffer = token_to_kv_pool.get_key_buffer(layer.layer_id)
        head_dim = layer.qk_head_dim
        q_heads = layer.tp_q_head_num
        scale = layer.scaling

        cached = getattr(forward_batch, "_h2o_host_view", None)
        if cached is None:
            if is_decode and _DECODE_INDICES:
                req_indices = list(_DECODE_INDICES)
                seq_lens = [0] * len(req_indices)
                extend_lens = [1] * len(req_indices)
            else:
                req_indices = forward_batch.req_pool_indices.tolist()
                seq_lens = forward_batch.seq_lens.tolist()
                extend_lens = forward_batch.extend_seq_lens.tolist()
            cached = (
                seq_lens,
                req_indices,
                extend_lens,
                forward_batch.rids or [str(i) for i in req_indices],
                forward_batch.out_cache_loc.tolist() if layer.layer_id == 0 else None,
            )
            forward_batch._h2o_host_view = cached
        seq_lens, req_indices, extend_lens, rids, out_slots = cached

        offset = 0
        plan = []
        for position, req_index in enumerate(req_indices):
            length = int(seq_lens[position])
            num_q = int(extend_lens[position])
            state = _state(int(req_index), str(rids[position]), length, device)
            if layer.layer_id == 0 and out_slots is not None:
                state.admit(out_slots[offset : offset + num_q])
            if state.should_rescore():
                if is_decode:
                    plan.append(
                        {
                            "decode": True,
                            "state": state,
                            "q_idx": torch.tensor([offset], dtype=torch.long, device=device),
                            "slots_t": torch.tensor(state.slots, dtype=torch.long, device=device),
                            "resident_t": torch.tensor(state.policy.resident, dtype=torch.long, device=device),
                            "which_t": torch.full((len(state.slots),), len(plan), dtype=torch.long, device=device),
                            "len": len(state.slots),
                        }
                    )
                else:
                    plan.append(
                        {
                            "decode": False,
                            "state": state,
                            "slots_t": torch.tensor(state.slots, dtype=torch.long, device=device),
                            "resident_t": torch.tensor(state.policy.resident, dtype=torch.long, device=device),
                            "num_q": num_q,
                            "prefix": len(state.slots) - num_q,
                            "q_view": query.view(-1, q_heads, head_dim)[offset : offset + num_q],
                        }
                    )
            offset += num_q

        if plan:
            if is_decode:
                _record_pass(plan, key_buffer, query, q_heads, head_dim, scale)
                for p in plan:
                    st = p["state"]
                    if st.pending_observers is None:
                        st.pending_observers = torch.zeros_like(st.policy.observers)
                    bump = torch.zeros_like(st.policy.observers)
                    bump.index_add_(
                        0, p["resident_t"], torch.ones(p["len"], device=device)
                    )
                    st.pending_observers = st.pending_observers + bump
            else:
                for p in plan:
                    _record_pass([p], key_buffer, query, q_heads, head_dim, scale)

        if layer.layer_id == num_layers - 1:
            for req_index in req_indices:
                state = _STATES.get(int(req_index))
                if state is not None:
                    state.fold_scores()

    original_extend = backend.forward_extend
    original_decode = backend.forward_decode

    def forward_extend(q, k, v, layer, forward_batch, *args, **kwargs):
        out = original_extend(q, k, v, layer, forward_batch, *args, **kwargs)
        observe(layer, q, forward_batch, is_decode=False)
        return out

    def forward_decode(q, k, v, layer, forward_batch, *args, **kwargs):
        out = original_decode(q, k, v, layer, forward_batch, *args, **kwargs)
        observe(layer, q, forward_batch, is_decode=True)
        return out

    backend.forward_extend = forward_extend
    backend.forward_decode = forward_decode

    original_forward = model_runner.forward

    def forward(forward_batch, *args, **kwargs):
        if forward_batch.forward_mode.is_decode():
            indices = _DECODE_INDICES or forward_batch.req_pool_indices.tolist()
            absolute = [
                _STATES[int(i)].absolute_len if int(i) in _STATES else 0
                for i in indices
            ]
            if forward_batch.positions is not None:
                forward_batch.positions = torch.tensor(
                    absolute,
                    device=forward_batch.positions.device,
                    dtype=forward_batch.positions.dtype,
                )
            for slot, req_index in enumerate(indices):
                state = _STATES.get(int(req_index))
                if state is None:
                    continue
                _dump(state, int(absolute[slot]) if slot < len(absolute) else None)
                state.decode_step += 1
        return original_forward(forward_batch, *args, **kwargs)

    model_runner.forward = forward

    from sglang.srt.managers.schedule_batch import ScheduleBatch

    if not getattr(ScheduleBatch, "_h2o_installed", False):
        ScheduleBatch._h2o_installed = True
        original_prepare = ScheduleBatch.prepare_for_decode

        def prepare_for_decode(self, *args, **kwargs):
            # Before, not after: prepare_for_decode allocates this step's new
            # token at the current seq_len and then advances the lengths, so
            # the compacted row must already be in place.
            apply_eviction(self)
            return original_prepare(self, *args, **kwargs)

        ScheduleBatch.prepare_for_decode = prepare_for_decode
