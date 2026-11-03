"""Short scripted Harbor trial, not an agent-performance experiment."""

import argparse
import asyncio
import importlib.metadata
import json
import uuid
from pathlib import Path

from harbor.models.trial.config import AgentConfig, EnvironmentConfig, TaskConfig, TrialConfig
from harbor.trial.trial import Trial


async def run(proxy_secret, output, mode="run-only"):
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
                    "mode": mode,
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
    assert len(snapshot["boxes"]) == (1 if mode == "manual" else 2)
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
    parser.add_argument("--mode", choices=["run-only", "shared", "manual"], default="run-only")
    args = parser.parse_args()
    asyncio.run(run(**vars(args)))
