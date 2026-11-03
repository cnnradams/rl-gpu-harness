"""Execute recorded study rows through native Modal queuing and Harbor trials."""

import hashlib
import json
import uuid
from pathlib import Path

import modal

from benchmarks.manifest import runtime_sources, validate_launch, verify
from benchmarks.run import image, operator_secrets, results

app = modal.App("gpu-harness-study")
image = image.add_local_file("pyproject.toml", "/study-meta/pyproject.toml", copy=True).add_local_file(
    "uv.lock",
    "/study-meta/uv.lock",
    copy=True,
)


@app.function(
    image=image,
    volumes={"/results": results},
    cpu=1,
    memory=4096,
    timeout=86400,
    nonpreemptible=True,
    max_containers=64,
    retries=0,
)
async def run_row(index: int, manifest: dict, cohort: str):
    from benchmarks.admission import claim_once
    from benchmarks.run import execute

    validate_launch(manifest)
    changed = verify(manifest, Path("/tasks"), Path(), actual_source=runtime_sources())
    if changed:
        raise ValueError("Study inputs changed: " + ", ".join(changed))
    row = manifest["trials"][index]
    if manifest["tasks"][row["task"]].get("requires_verifier_secret"):
        import os

        if not os.environ.get("OPENAI_API_KEY"):
            raise ValueError("Native verifier OPENAI_API_KEY is missing; no trial started")
    study = {
        "cohort": cohort,
        "index": index,
        "row": row,
        "manifest_sha256": hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest(),
        "task": manifest["tasks"][row["task"]],
        "agent": manifest["agent"],
        "service": manifest["service"],
    }
    await claim_once(f"{cohort}/{index}")
    return await execute(
        row["task"],
        study["task"]["gpu_image"],
        row["interface"],
        manifest["agent"]["name"],
        row["agent_seconds"],
        study=study,
        agent_runtime=row.get("agent_runtime"),
    )


@app.local_entrypoint()
def main(
    manifest_path: str, verifier_secret: str = "", agent_secret: str = "",
    proxy_secret: str = "rl-gpu-harness-proxy",
):
    """Submit persisted calls; invoke with modal run --detach."""
    manifest = json.loads(Path(manifest_path).read_text())
    validate_launch(manifest)
    changed = verify(manifest, Path("artifacts/tasks"), Path(__file__).resolve().parent.parent)
    if changed:
        raise ValueError("Study inputs changed: " + ", ".join(changed))
    if any(t.get("requires_verifier_secret") for t in manifest["tasks"].values()) and not verifier_secret:
        raise ValueError("Native judges require --verifier-secret supplying OPENAI_API_KEY")
    cohort = Path(manifest_path).stem + "-" + uuid.uuid4().hex[:8]
    output = Path("artifacts/scientific-results") / cohort
    output.mkdir(parents=True)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    secrets = operator_secrets(agent_secret, proxy_secret, verifier_secret)
    runner = run_row.with_options(max_containers=manifest["concurrency"], secrets=secrets)
    runner.hydrate()
    controller = {"function_id": runner.object_id, "max_containers": manifest["concurrency"]}
    (output / "controller.json").write_text(json.dumps(controller, indent=2) + "\n")
    print(json.dumps({"cohort": cohort, "controller": controller}), flush=True)
    for index in range(len(manifest["trials"])):
        call = runner.spawn(index, manifest=manifest, cohort=cohort)
        record = {"index": index, "function_call_id": call.object_id}
        (output / f"row-{index}.json").write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps(record), flush=True)
