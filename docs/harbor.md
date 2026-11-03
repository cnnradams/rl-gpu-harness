# Run Harbor with GPU tools

## One-time setup

From this repository, with Modal credentials configured (`uv run modal setup`
if needed):

```bash
uv sync --extra harbor --frozen
export MODAL_ENVIRONMENT=<your-environment>
uv run modal deploy -m gpu_harness.app
uv run python examples/setup_proxy.py --secret rl-gpu-harness-proxy
```

The last command creates a proxy token, grants it access to this environment,
and stores its credentials in a Modal Secret named `rl-gpu-harness-proxy`.
Reuse that Secret for subsequent runs. If it already exists, skip that command.

## Run a task

Use the same Modal environment as setup; Harbor looks up the deployed service
and named Secret there. Configure your agent's model credentials as usual, then:

```bash
export MODAL_ENVIRONMENT=<your-environment>
uv run --extra harbor harbor run -p path/to/task -a terminus-2 -m your/model \
  -e gpu_harness.harbor:GPUEnvironment \
  --verifier gpu_harness.harbor:GPUVerifier \
  --ek gpu_proxy_secret=rl-gpu-harness-proxy \
  --extra-instruction "If /etc/gpu-harness.md exists, it describes GPU access for this task."
```

`gpu_proxy_secret` is the **name of the Secret**, not a token. The service is
shared across runs; each trial gets its own GPU session and limits.

The default is **run-only with shared `/workspace`**: the agent runs on CPU,
`gpu run -- COMMAND` starts a GPU box, and that box stops when the command exits.

## Task compatibility and grading

GPU task images need Python 3.10+ with `venv` and pip 22.3+. During image build,
the adapter installs the client into a private virtual environment; task Python
packages remain unchanged. Installation needs build-time network access, not
sandbox runtime internet access. Unsupported images fail with a setup error.

**CPU tasks pass through unchanged** to Harbor's normal Modal environment:
no GPU-tool session, client installation, or proxy credentials are needed.

For GPU tasks, the adapter supports Linux single-container environments and
both verifier modes. `GPUVerifier` delegates separate verifiers to Harbor as
usual. For a shared verifier, it closes the agent's GPU session, snapshots the
CPU filesystem, and starts a GPU verifier with that filesystem and the same
committed workspace Volume. The original task tests run unchanged, outside the
agent's GPU budget. Installed packages and files outside `/workspace` are
included in the filesystem snapshot.

**Snapshot verification preserves disk state, not running processes, open
connections, or GPU memory.** Tasks whose tests require an already-running
service are not equivalent under this approach. Multi-step tasks with shared
verifiers are rejected; transferring verifier changes back to subsequent agent
steps is not implemented. The `--verifier` option is required because Harbor
0.23 has no environment callback at the start of grading.

Harbor collects the task's declared artifacts, restores them at their original
paths, and runs its tests in the task's verifier image with the verifier's GPU
resources. The verifier receives no agent GPU-tool credentials or budget.
For shared `/workspace` files, collection uses a fresh read-only CPU mount so
GPU-written submissions do not depend on the agent's last `gpu sync`.
Paths outside `/workspace` remain local to their machine.

The launch command appends the tool description through Harbor's native
`--extra-instruction` option; it does not edit the task or replace the agent.

## Modes and limits

The task's `[environment]` GPU count and `gpu_types` select the GPU workers;
Harbor resource overrides apply first. The agent runs on CPU. Workers use the
task image (Dockerfile or configured registry image) and its CPU/memory request.
The separate verifier retains its own task resources.

Select an interface with `--ek mode=...`:

| Mode | Behavior |
| --- | --- |
| `run-only` (default) | Shared `/workspace`; `gpu run` starts a GPU for one program and stops it when that program exits. No `gpu exec`. |
| `shared` | Shared `/workspace`; `gpu exec` can run additional commands on a live GPU box. |
| `manual` | Separate filesystems; the agent transfers files with `gpu push` and `gpu pull`. Run and exec are available. |

Resource policies are independent of the interface. Add either or both:

```bash
--ek gpu_seconds=600        # 600 allocated GPU-seconds across all workers
--ek cuda_idle_seconds=300  # Stop a box after five minutes without a CUDA process
```

With no policy options, there is no session GPU-time budget or idle-release
policy. A box still has a maximum lifetime (two hours by default). For example,
a shared run/exec interface with a ten-minute GPU budget adds:

```bash
--ek mode=shared --ek gpu_seconds=600
```

Advanced overrides use `gpu_limits`, for example:

```bash
--ek 'gpu_limits={"max_boxes":4,"box_seconds":14400,"image_id":"im-custom-image"}'
```

An explicit `gpu_limits.gpu` / `gpus_per_box` overrides the task's GPU choice.
Tasks without a GPU request pass through; use Harbor's GPU resource override to change that request. If a task lists
several GPU types, the first is used, matching Harbor's Modal environment.
Set `--ek agent_runtime=vm` to select Modal's VM runtime for the CPU agent.

## Credentials and workspace

The trusted runner uses operator Modal credentials. `gpu_proxy_secret` names a
Secret containing `MODAL_PROXY_TOKEN_ID` and `MODAL_PROXY_TOKEN_SECRET`.
Outbound header replacement keeps real HTTP credentials outside the agent.
Modal cannot combine header injection with a network allowlist. For restricted
network tasks, select `gpu_proxy_file` instead: a trusted-runner JSON file
with `Modal-Key` and `Modal-Secret`. The Environment uploads its scoped session
config to `/run/secrets/gpu-harness.json` with mode 0600. This mode makes those
HTTP credentials readable by the agent. It also supports deployments whose
network allowlist requires file-based delivery.

Each shared session owns a Volume. The Environment mounts it at `/workspace`
on the CPU agent, seeds the task files and binds the agent Sandbox before GPU
commands start. GPU boxes mount the same session Volume. Reload and command
lifetime behavior are described in [usage](usage.md).

The agent image needs Python 3 and pip; setup adds `httpx` and the standalone
`gpu` client. The adapter writes the selected interface and policy to
`/etc/gpu-harness.md` and to the trial's `gpu-harness/instructions.md`.
Harbor's agent implementation, terminal handling and compaction remain upstream.

At stop, the Environment closes GPU access, stops owned boxes and collects
committed GPU logs plus session metadata under the trial's `gpu-harness/`.
Harbor's separate verifier receives its task resources and no agent GPU-tool
credentials. The integration targets direct Linux Modal Sandboxes and pins
Harbor 0.23.0 through the optional dependency.

Run a short scripted GPU check (no model credentials needed) with:

```bash
uv run --extra harbor python examples/harbor_smoke.py --proxy-secret rl-gpu-harness-proxy
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

## GPU time

After a trial finishes:

```bash
uv run python examples/harbor_usage.py jobs/<job>/<trial>
```

This reports allocated GPU seconds, agent wall time, their ratio, and GPU startup
time. Allocation includes idle time; it is not CUDA utilization. Verifier phase
time is listed separately and excluded from the agent's GPU usage. The source
 records are saved in each trial's `gpu-harness/session.json`.
