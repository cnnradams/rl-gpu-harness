# Benchmark adapters

These optional adapters prepare iterative GPU tasks for upstream Harbor, with
separate final verification. The GPU service also works with other tasks and
runners through its CLI and HTTP API.

| Adapter | Pinned source | Metric |
| --- | --- | --- |
| KernelBench matmul and LayerNorm | `423217d9fda91e0c2d67e4a43bf62f96f6d104f1` | Correctness-gated speedup |
| RSI nanoGPT, MoE routing and SGLang KV pruning | `781a8091723b7a3651a3599e6b6d6063c82182bf` | Native task score |
| PostTrainBench | See [setup and handoff](posttrain.md) | Native evaluation and audits |

Prepare from clean source checkouts:

```bash
uv run --extra harbor python -m benchmarks.prepare --source /path/to/KernelBench --kernels fused-matmul layernorm
uv run --extra harbor python -m benchmarks.prepare_nano_gpt --source /path/to/rsi-benchmark
uv run --extra harbor python -m benchmarks.prepare_rsi --source /path/to/rsi-benchmark --task moe-router-design
uv run --extra harbor python -m benchmarks.prepare_rsi --source /path/to/rsi-benchmark --task kv-cache-pruning-sglang
```

Generated tasks live under `artifacts/tasks/`. Preparation records source and
verifier hashes, supplies terminal tooling and adapts workspace/timer setup.
RSI's native training, data builders and scoring remain unchanged. KernelBench
uses its task shapes, FP32 inputs and numerical checks; development feedback
uses 5 correctness/20 timing trials and final scoring uses 5/100.

## Optional runners

Run a scripted solution with a separate native verifier:

```bash
uv run --extra harbor modal run benchmarks/run.py --task kernelbench-layernorm \
  --interface shared --agent oracle --proxy-secret your-proxy-secret
```

For Terminus-2, use `--agent terminus-2 --agent-secret your-agent-secret`.
The trusted model Secret supplies `AGENT_API_KEY`, `AGENT_API_BASE` and
`AGENT_MODEL`. `--verifier-secret` supplies any credentials required by native
judges. These credentials remain in the trusted runner.

Interfaces are `attached` for a GPU in the agent Sandbox, `remote` for explicit
file transfer, and `shared` for a mounted workspace. `--seconds` sets the agent
horizon; the task retains its native verifier timeout. `--gpu-image` reuses a
built task image. Effective task settings, images and sources accompany the
native result. Outputs download under `artifacts/benchmark-results/`.

For repeated trials, [build a manifest](manifest.md) and use the
[cohort runner](study.md). Both runners use the same Harbor trial implementation.
