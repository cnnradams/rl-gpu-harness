# PostTrainBench adapter

Prepare the pinned GSM8K / Qwen3-1.7B-Base task:

```bash
uv run --extra harbor python -m benchmarks.prepare_posttrain \
  --source /path/to/PostTrainBench --hours 1
```

The generated task uses `/workspace` while retaining the original path as an
image alias. Preparation adjusts file locations and hardware descriptions,
replaces unused agent CLIs with terminal tools, and retains the native judge
and evaluator code. File hashes record these boundaries. The original
`test.sh` is retained behind a model-collection wrapper.

Use Harbor or the [optional runner](README.md) with an agent horizon matching
`--hours`. Native judges require their verifier credentials, including
`OPENAI_API_KEY`; the trusted runner accepts them through `--verifier-secret`.
The model provider uses a separate `--agent-secret`. Allow enough controller
lifetime for the task's native verifier.

For attached/manual access, native collection copies the final model into a
per-trial model Volume. For shared access, the verifier mounts the committed
session workspace after agent cleanup and runs the same collection code there.
This reads final GPU outputs without reloading the agent's former view.

`benchmarks.posttrain.add_verification_hook` supplies the model Volume and
agent-log handoff through Harbor hooks. For Terminus-2 it passes the native
ATIF to the judge's existing text fallback. Model weights stay on their
recorded Volume; native results and logs remain ordinary Harbor outputs.
