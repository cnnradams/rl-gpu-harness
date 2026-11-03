# Harbor integration

`gpu_harness.harbor:GPUEnvironment` extends upstream Harbor's Modal Environment.
Harbor creates the trial, runs the chosen agent, collects artifacts and performs
separate verification. The extension installs the HTTP GPU client and mints
one GPU-access session for that agent.

```text
trusted Harbor runner ──Modal RPC──▶ session issuer
        │                            returns service URL + credential
        ▼
CPU agent Sandbox ──HTTP──▶ shared GPU service ──▶ GPU Sandboxes
        └─ submitted files ──▶ Harbor's separate verifier
```

```bash
uv sync --extra harbor --frozen
uv run modal deploy -m gpu_harness.app
```

Select the Environment in a Harbor job or trial config. The image is the GPU
task image; Harbor still prepares the CPU agent from the task definition.

```json
{
  "environment": {
    "import_path": "gpu_harness.harbor:GPUEnvironment",
    "kwargs": {
      "gpu_proxy_secret": "your-proxy-secret",
      "gpu_limits": {
        "image_id": "im-your-task-image",
        "gpu": "H100",
        "max_boxes": 1,
        "box_seconds": 3600,
        "shared_workspace": true,
        "run_only": true
      }
    }
  }
}
```

Omit `run_only` for the standard run/exec/transfer interface. Set
`agent_runtime: "vm"` in environment kwargs to choose Modal's VM runtime for
the CPU agent; otherwise the provider default applies. GPU workers and the
verifier retain their own resources and runtime.

## Credentials and workspace

The trusted runner uses operator Modal credentials. `gpu_proxy_secret` names a
Secret containing `MODAL_PROXY_TOKEN_ID` and `MODAL_PROXY_TOKEN_SECRET`.
Outbound header replacement keeps real HTTP credentials outside the agent.
For file delivery, select `gpu_proxy_file` instead: a trusted-runner JSON file
with `Modal-Key` and `Modal-Secret`. The Environment uploads its scoped session
config to `/run/secrets/gpu-harness.json` with mode 0600. This mode makes those
HTTP credentials readable by the agent. It also supports deployments whose
network allowlist requires file-based delivery.

Each shared session owns a Volume. The Environment mounts it at `/workspace`
on the CPU agent, seeds the task files and binds the agent Sandbox before GPU
commands start. GPU boxes mount the same session Volume. Reload and command
lifetime behavior are described in [usage](usage.md).

A custom runner can use the same issuer and bind operation:

```python
import modal
from gpu_harness import APP_NAME
from gpu_harness.session import create_session

session = create_session(limits, run_id=rollout_id)
# Mount session["workspace_volume"] when creating the CPU agent.
modal.Function.from_name(APP_NAME, "bind_agent").remote(session["token"], agent.object_id)
```

The agent image needs Python 3 and pip; setup adds `httpx` and the standalone
`gpu` client. Use `gpu_harness.harbor.instructions(limits)` as Harbor
`extra_instructions` to describe the configured interface. The Environment
also saves this text to `gpu-harness/instructions.md`; it does not rewrite the
task's prompt. Harbor's agent implementation, terminal handling and compaction
remain upstream.

At stop, the Environment closes GPU access, stops owned boxes and collects
committed GPU logs plus session metadata under the trial's `gpu-harness/`.
Harbor's separate verifier receives its task resources and no agent GPU-tool
credentials. The integration targets direct Linux Modal Sandboxes and pins
Harbor 0.23.0 through the optional dependency.

Run the scripted integration check with:

```bash
uv run --extra harbor python examples/harbor_smoke.py --proxy-secret your-proxy-secret
```

## Review local trajectories

View the recorded comparisons directly from the repository root:

```bash
uv run --extra harbor harbor view rollouts --jobs
```

Prepare completed native trial directories for Harbor's viewer:

```bash
uv run --extra harbor python examples/review_trials.py path/to/trials \
  --output artifacts/local-review/exported-trials
uv run --extra harbor harbor view artifacts/local-review --jobs
```

The output must be new and outside the source. The helper copies each selected
trial; it leaves source files unchanged. For a correct KernelBench result,
speedup is also exposed as the viewer reward. Its original `result.json` is
retained as `result.native.json`, with hashes in `review.json`. Other native
rewards, missing grades and exceptions remain unchanged. Local review needs
neither Modal credentials nor a hosted viewer.
