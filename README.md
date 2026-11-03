# GPU harness

GPU tools for agents, backed by a shared HTTP service on Modal. A trusted
launcher creates a session for each rollout; the agent uses the `gpu` CLI or
HTTP API to run work on native Modal GPU Sandboxes.

```text
trusted launcher ──Modal RPC──▶ session + service URL
agent + gpu client ──HTTP─────▶ shared service ──▶ GPU Sandboxes
```

Each session sets its GPU type, resources, box count, lifetime and optional
GPU-time or idle policy. Files can move explicitly or live on a shared
`/workspace` Volume. In run-only sessions, `gpu run` starts a box for its
program and stops the box when that program exits.

```bash
uv sync --frozen
uv run modal deploy -m gpu_harness.app
uv run python -m gpu_harness.session --gpu H100 --output artifacts/session.json
```

Deploy the service once; mint sessions with different settings for individual
rollouts. The issuer returns the deployed URL, so runners do not hardcode it.

- [Usage](docs/usage.md): sessions, credentials, commands and workspace behavior.
- [Harbor](docs/harbor.md): optional upstream integration and local trace review.
- [Benchmark adapters](benchmarks/README.md): task preparation and optional runners.
- [Results](docs/results.md): comparison tables, trace findings and archived evidence.
- [Rollouts](rollouts/): trajectories and terminal recordings, ready for `harbor view rollouts --jobs`.

Run local tests with `uv run --extra harbor python -m pytest -q`.
