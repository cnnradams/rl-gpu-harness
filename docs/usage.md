# Sessions and GPU commands

Deploy the service with `uv run modal deploy -m gpu_harness.app`. A trusted
launcher then mints a separate session for each rollout:

```bash
uv run python -m gpu_harness.session --gpu H100 --max-boxes 1 \
  --box-seconds 7200 --output artifacts/session.json
```

This command calls the deployed `mint_session` Modal function; it does not
deploy a service or allocate a GPU. It saves the discovered HTTP URL, session
credential and settings in a new mode-0600 file. Programmatic launchers use
`gpu_harness.session.create_session(Limits(...), run_id=...)`.

`--image-id` selects a task image; the default is a minimal Python image.
`--gpus-per-box`, `--cpu` and `--memory-mib` set resources. `--box-seconds` bounds
each box's lifetime. `--gpu-hours` enables a session-wide allocation budget.

## Credential delivery

The client accepts a config file, environment variables or caller-supplied HTTP
headers. Operator Modal credentials belong to the trusted launcher.

A client config can contain `url`, `token`, and a `proxy` object with `Modal-Key`
and `Modal-Secret`. Its default path is `/run/secrets/gpu-harness.json`;
`--config PATH` or `GPU_HARNESS_CONFIG` selects another path. Keep a real
credential file mode 0600 and outside the shared workspace. Environment-based
clients use `GPU_HARNESS_URL`, `GPU_HARNESS_TOKEN`, `GPU_HARNESS_PROXY_KEY` and
`GPU_HARNESS_PROXY_SECRET`.

Modal launchers can instead pass `credential_policy(session, "your-proxy-secret")`
when creating the agent Sandbox. That Secret contains `MODAL_PROXY_TOKEN_ID`
and `MODAL_PROXY_TOKEN_SECRET`; the policy injects real authentication headers
outside the agent, for the service hostname. The agent's client can send dummy
credentials. Run-only client configuration also carries the nonsecret
`run_only: true` flag, either directly or inside `limits`.

Create and allow an endpoint proxy token in the deployment environment:

```bash
modal workspace proxy-tokens create --name gpu-harness --json
modal workspace proxy-tokens allow <token-id> <environment>
```

## Commands

```bash
gpu run --detach -- python3 /workspace/train.py
gpu status COMMAND
gpu wait COMMAND
gpu stop BOX
```

`run` starts a GPU box with the supplied program as its entrypoint. The box
stops when that program exits. Foreground commands stream output and return
the program's exit code; `--detach` returns box and command IDs. `status` and
`wait` take the **command ID**, while `stop` takes the **box ID**. CPU work uses
the agent's ordinary shell.

Remote commands receive EOF on stdin; local pipes and heredocs are not forwarded.

Standard sessions also provide `gpu exec BOX -- PROGRAM ...`, `gpu push BOX
LOCAL REMOTE`, `gpu pull BOX REMOTE LOCAL`, and `gpu refresh BOX`. Exec and
transfers require a live box; files outside a shared Volume are local to it.
Transfers stream through HTTP and native Sandbox stdin/stdout, with integrity
checks and no whole-file buffer or total file-size quota.

Successful command responses show usable box/command IDs. If the run/exec
response is unavailable, the client prints a retry command using `--key ID`.
Reuse that key for the same invocation; `GET /requests/ID` exposes its recorded
outcome. An ambiguous request is not automatically repeated.

`--timeout` explicitly bounds a remote command; exec has no implicit command
timeout. Box lifetime and resource policies remain separate. A request beyond
the session's box limit fails without allocating or queuing another box.

## Shared workspace and run-only access

`--shared-workspace` creates a session Volume mounted at `/workspace` on GPU
boxes. A custom launcher mounts the returned `workspace_volume` on its CPU
agent and calls the trusted `bind_agent` RPC once, before GPU work. The
[Harbor Environment](harbor.md) performs this setup automatically.

`gpu sync` commits and reloads the CPU agent's view. `gpu refresh BOX` reloads a
live GPU box's view. Reload requires no open files or process working
directories on the mount. The Harbor CPU agent starts in `/`; GPU commands default
to `/workspace`. Failed sync reports which steps completed. Same-file
concurrent edits use last-writer-wins semantics.

Add `--run-only` alongside `--shared-workspace`, or set
`Limits(shared_workspace=True, run_only=True, ...)`. In that mode, CPU workspace
writes are committed before each GPU run, and the new box mounts them. GPU
writes commit in the background and on shutdown; `gpu sync` refreshes CPU
outputs. The client offers run, stop, status, wait, policy and sync; the service
denies per-box exec, transfer and refresh operations. This mode does not make
separate mounted views a live shared filesystem.

## Policies and lifecycle

GPU-time budgets count recorded allocation after readiness, including idle
time and all GPUs; provisioning is excluded. Unobserved intervals are flagged
with `gpu_time_accounting_complete: false`. This is allocation accounting,
not provider billing.

`--idle-seconds 300 --idle-probe cuda` stops a box after five minutes without an
observed CUDA-attached process. CPU tool activity does not reset that clock.
Probes run about every ten seconds; failed probes do not trigger release.
`--idle-probe commands` instead measures time without a running tool command.
`gpu policy` reports the session's settings and recorded allocation.

The trusted launcher closes the session and stops its boxes at rollout end;
`gpu_harness.lifecycle.close_and_export` also collects committed logs and
credential-free session metadata. Failed boxes lose their running processes;
work is not automatically restarted.
