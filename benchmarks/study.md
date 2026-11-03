# Running a recorded cohort

The optional cohort runner executes manifest rows as upstream Harbor trials.
Modal's function queue bounds concurrency; each trusted CPU controller runs
setup, the agent, cleanup and separate verification.

```bash
uv run --extra harbor modal run --detach -m benchmarks.study \
  --manifest-path artifacts/interface-manifest.json \
  --agent-secret your-agent-secret --proxy-secret your-proxy-secret
```

The manifest supplies the agent model and `kwargs.api_base`; the model Secret
supplies `AGENT_API_KEY`. Oracle trials need no model Secret. Add
`--verifier-secret` when native judges require separate credentials.

Before starting a row, the runner checks recorded task/source hashes, image
settings and required credentials. It preserves the row's instructions,
resource settings and verifier horizon. An atomic Dict claim prevents a
retried controller invocation from starting a replacement rollout; interrupted
work is not automatically resumed. Controller timeouts are deployment settings,
separate from the agent horizon and session GPU policies.

Each task records its agent and verifier base images. The Environment records
the derived runtime images and cleanup timestamps. GPU access uses a session
minted for that trial on the shared HTTP service.

The command submits asynchronous calls and returns. It saves the manifest and
each row's FunctionCall ID under `artifacts/scientific-results/<cohort>/`.
`--detach` keeps the app running after the submitting process exits. Retrieve a
call's outcome with `modal.FunctionCall.from_id(ID).get()`; failures remain
exceptions. The `gpu-harness-validation-results` Volume retains native trial
directories and `.exports/<cohort>/` archives; model weights remain on their
recorded Volumes. Download completed archives and use the
[local viewer helper](../docs/harbor.md#review-local-trajectories) to review them.
