# Review the 30-minute comparisons

From the repository root:

```bash
uv run --extra harbor harbor view rollouts --jobs
```

This snapshot contains 169 actual native rollouts of 210 assigned slots; 41 remain pending. Each of seven jobs expects six trials for each of five tasks. Compare within a task: its metrics differ from those of other tasks. Failed, timed-out and preempted outcomes remain included. No pending episodes are invented.

Agent trajectories, terminal recordings, verifier diagnostics and submission source retain their original bytes. Offline curation removes training datasets, model weights, caches and duplicated runner source; every included/omitted path has a SHA256, size and reason in `provenance.json`, with the original archive pointer. No agent-facing limit is introduced.

Kernel scalar rewards are viewer projections; `result.native.json` preserves the original result whenever projected. Explicit recovered-grade overlays carry their separate evidence and do not change native outcomes. `selection-input.json` pins the exact comparison selection.
