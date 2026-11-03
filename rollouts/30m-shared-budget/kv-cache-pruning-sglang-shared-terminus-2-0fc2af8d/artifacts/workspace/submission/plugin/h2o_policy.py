# Heavy-hitter KV-cache pruning for SGLang: retention policy bookkeeping.
#
# One H2OPolicy per request tracks, on device, the accumulated attention mass
# and observing-query count for every absolute position produced so far, the
# list of positions still resident, and the KV slots backing them. Ranking is
# mean attention per observing query inside eight depth buckets, with the
# opening sink positions protected; see h2o_runtime.py for the engine side.

from __future__ import annotations

from typing import List, Optional

import torch


def window_sizes(budget: int, heavy_ratio: float) -> tuple:
    heavy = int(budget * heavy_ratio)
    heavy = max(0, min(int(budget), heavy))
    return heavy, int(budget) - heavy


class H2OPolicy:
    """Heavy-hitter retention over one request's absolute KV positions."""

    def __init__(self, budget: int, heavy_ratio: float, device, sink: int = 16) -> None:
        self.budget = int(budget)
        self.heavy_ratio = float(heavy_ratio)
        self.heavy, self.recent = window_sizes(self.budget, self.heavy_ratio)
        # Depth buckets the heavy budget is spread across; one bucket is a
        # global ranking and collapses into the prompt prefix on this model.
        self.buckets = 8
        # Attention sink: the first tokens absorb a large share of the mass on
        # every later query, and dropping them corrupts the softmax far beyond
        # what their rank suggests. They are kept out of the evictable pool.
        self.sink = max(0, int(sink))
        # Eviction runs when the resident set reaches the budget and trims back
        # by `slack`, so per-step Python only pays for a ranking every `slack`
        # admissions rather than on every decode step.
        self.slack = max(1, min(64, self.budget // 16))
        self.device = device
        self.resident: List[int] = []
        self.scores = torch.zeros(0, dtype=torch.float32, device=device)
        self.observers = torch.zeros(0, dtype=torch.float32, device=device)

    def append(self, num_new: int) -> None:
        num_new = int(num_new)
        if num_new <= 0:
            return
        base = len(self.resident)
        self.resident.extend(range(base, base + num_new))
        self.scores = torch.cat(
            [self.scores, torch.zeros(num_new, dtype=torch.float32, device=self.device)]
        )
        self.observers = torch.cat(
            [self.observers, torch.zeros(num_new, dtype=torch.float32, device=self.device)]
        )

    def add_column_mass(self, mass: torch.Tensor, observers: torch.Tensor) -> None:
        """Accumulate one scoring pass, indexed by absolute position."""
        if mass.numel() != self.scores.numel():
            raise ValueError(
                f"mass has {mass.numel()} entries, position table has {self.scores.numel()}"
            )
        self.scores = self.scores + mass.to(torch.float32)
        self.observers = self.observers + observers.to(torch.float32)

    def evict(self) -> List[int]:
        """Trim the resident set to below the budget; return evicted RESIDENT idxs.

        Ranking is mean accumulated attention per observing query, compared
        inside depth buckets so the heavy budget is spent across the context
        rather than collapsing into the prefix. The sink prefix and the recent
        window are not evictable.
        """
        if len(self.resident) < self.budget:
            return []
        target = max(1, self.budget - self.slack)
        overflow = len(self.resident) - target
        protected_from = max(0, len(self.resident) - self.recent)
        overflow = min(overflow, protected_from)
        if overflow <= 0:
            return []
        evictable_hi = protected_from
        evictable_lo = min(self.sink, evictable_hi)
        keep_quota = max(0, evictable_hi - evictable_lo - overflow)
        doomed = self._bucketed_victims(evictable_lo, evictable_hi, keep_quota)
        doomed_set = set(doomed)
        self.resident = [p for i, p in enumerate(self.resident) if i not in doomed_set]
        return doomed

    def _bucketed_victims(self, lo: int, hi: int, keep_quota: int) -> List[int]:
        if hi <= lo:
            return []
        if keep_quota <= 0:
            return list(range(lo, hi))
        span = hi - lo
        buckets = min(self.buckets, span)
        pos = torch.tensor(self.resident[lo:hi], device=self.device, dtype=torch.long)
        mean = self.scores[pos] / self.observers[pos].clamp(min=1.0)
        means = mean.tolist()
        survivors = set()
        for index in range(buckets):
            a = lo + (index * span) // buckets
            b = lo + ((index + 1) * span) // buckets
            if b <= a:
                continue
            share = round((index + 1) * keep_quota / buckets) - round(
                index * keep_quota / buckets
            )
            if share <= 0:
                continue
            ranked = sorted(range(a, b), key=lambda i: (means[i - lo], self.resident[i]), reverse=True)
            survivors.update(ranked[:share])
        return [i for i in range(lo, hi) if i not in survivors]

    def retained(self) -> List[int]:
        return list(self.resident)
