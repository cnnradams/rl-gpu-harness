"""Native PostTrainBench model handoff and trusted judge-trace delivery."""

import hashlib
import json
import shutil
import uuid

import modal
from harbor.trial.hooks import TrialEvent


async def model_volume():
    name = "gpu-harness-model-" + uuid.uuid4().hex
    volume = modal.Volume.from_name(name, create_if_missing=True, version=2)
    await volume.hydrate.aio()
    return name


def add_verification_hook(trial, *, interface, agent):
    workspace = None

    async def remember_workspace(event):
        nonlocal workspace
        if interface == "shared":
            workspace = trial.agent_environment.gpu_session["workspace_volume"]

    async def before_verification(event):
        trace = None
        if agent == "terminus-2":
            source = trial.paths.agent_dir / "trajectory.json"
            if not source.is_file():
                raise RuntimeError("PostTrainBench judges require the complete Terminus ATIF")
            target = trial.paths.artifacts_dir / "logs/artifacts/agent_logs/terminus-2.txt"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            with source.open("rb") as stream:
                trace = {
                    "sha256": hashlib.file_digest(stream, "sha256").hexdigest(),
                    "bytes": source.stat().st_size,
                }
        if interface == "shared":
            # Only the verifier is created after this hook. The agent already
            # has its own native mount and its GPU session is closed by Harbor.
            if workspace is None:
                raise RuntimeError("Shared PostTrainBench workspace was not initialized")
            trial.config.environment.kwargs["volumes"] = {"/mnt/ptb-workspace": workspace}
            trial.config.verifier.env["PTB_SHARED_WORKSPACE"] = "/mnt/ptb-workspace"
        trial.config.verifier.env["PTB_AGENT_CONFIG"] = trial.config.agent.model_name or agent
        (trial.paths.trial_dir / "posttrain-handoff.json").write_text(
            json.dumps(
                {
                    "interface": interface,
                    "verifier_volumes": trial.config.environment.kwargs.get("volumes", {}),
                    "native_atif": trace,
                },
                indent=2,
            )
            + "\n"
        )

    trial.add_hook(TrialEvent.AGENT_START, remember_workspace)
    trial.add_hook(TrialEvent.VERIFICATION_START, before_verification)
