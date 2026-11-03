"""Short scripted Harbor trial, not an agent-performance experiment."""

import argparse
import asyncio
import importlib.metadata
import json
import uuid
from pathlib import Path

import modal
from harbor.models.trial.config import AgentConfig, EnvironmentConfig, TaskConfig, TrialConfig
from harbor.trial.trial import Trial

from gpu_harness import APP_NAME


async def run(proxy_secret, output, gpu, shared_workspace=False):
    operator_app = await modal.App.lookup.aio(APP_NAME, create_if_missing=True)
    image = modal.Image.debian_slim(python_version="3.12").workdir("/workspace")
    if shared_workspace:
        image = image.run_commands("mkdir -p /workspace && echo image-content > /workspace/image-file")
    image = await image.build.aio(operator_app)
    trial = await Trial.create(
        TrialConfig(
            task=TaskConfig(path=Path(__file__).parent / "harbor-smoke"),
            trial_name=f"gpu-smoke-{uuid.uuid4().hex[:8]}",
            trials_dir=Path(output),
            agent=AgentConfig(name="oracle"),
            environment=EnvironmentConfig(
                import_path="gpu_harness.harbor:GPUEnvironment",
                kwargs={
                    "gpu_proxy_secret": proxy_secret,
                    "gpu_limits": {
                        "image_id": image.object_id,
                        "gpu": gpu,
                        "max_boxes": 1,
                        "box_seconds": 180,
                        "cpu": 0.25,
                        "memory": 1024,
                        "shared_workspace": shared_workspace,
                    },
                },
            ),
        )
    )
    result = await trial.run()
    if result.exception_info:
        raise RuntimeError(
            f"{result.exception_info.exception_type}: {result.exception_info.exception_message}"
        )
    assert result.verifier_result.rewards == {"reward": 1}
    snapshot = json.loads((trial.paths.trial_dir / "gpu-harness" / "session.json").read_text())
    assert snapshot["closed"] and not snapshot["errors"]
    assert snapshot["session"]["run_id"] == str(result.id)
    assert len(snapshot["boxes"]) == (2 if shared_workspace else 1)
    assert all(box["observed_exit_code"] is not None for box in snapshot["boxes"])
    assert (trial.paths.trial_dir / "gpu-harness" / "traces" / "events.jsonl").is_file()
    summary = {
        "harbor_version": importlib.metadata.version("harbor"),
        "trial_id": str(result.id),
        "trial_dir": str(trial.paths.trial_dir),
        "gpu_session": snapshot["session"]["trial"],
        "gpu_boxes_stopped": [box["id"] for box in snapshot["boxes"]],
        "verifier_mode": result.verifier_environment_mode,
        "rewards": result.verifier_result.rewards,
        "gpu_report": json.loads((trial.paths.artifacts_dir / "workspace" / "report.json").read_text()),
        "scope": "Scripted lifecycle/file/GPU-discovery check; not optimization or model behavior evidence",
    }
    (trial.paths.trial_dir / "smoke-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proxy-secret", required=True)
    parser.add_argument("--output", default="artifacts/harbor-smoke")
    parser.add_argument("--gpu", default="T4")
    parser.add_argument("--shared-workspace", action="store_true")
    args = parser.parse_args()
    asyncio.run(run(**vars(args)))
