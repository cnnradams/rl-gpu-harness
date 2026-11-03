import secrets
import subprocess
import sys
import uuid

import modal

from . import APP_NAME, POLICY_INDEX_NAME, RECORDS_NAME, SERVER_NAME, TRACE_VOLUME_NAME
from .service import Limits, Service, create_api, token_hash

app = modal.App(APP_NAME)
records = modal.Dict.from_name(RECORDS_NAME, create_if_missing=True)
policy_index = modal.Dict.from_name(POLICY_INDEX_NAME, create_if_missing=True)
traces = modal.Volume.from_name(TRACE_VOLUME_NAME, create_if_missing=True, version=2)
image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "modal==1.6.3", "fastapi==0.162.0", "uvicorn==0.42.0"
)
image = image.add_local_python_source("gpu_harness")


def http_app():
    return create_api(Service(records, modal.App.lookup(APP_NAME), traces, policy_index=policy_index))


@app.server(
    image=image,
    cpu=0.25,
    memory=512,
    min_containers=1,
    max_containers=1,
    nonpreemptible=True,
    volumes={"/traces": traces},
    name=SERVER_NAME,
)
class gpu_api:
    @modal.enter()
    def startup(self):
        subprocess.Popen([sys.executable, "-m", "gpu_harness.app"])


@app.function(image=image, cpu=0.125, memory=256, min_containers=0, timeout=60)
def mint_session(limits: dict, run_id: str | None = None):
    limits = Limits(**limits)
    url = modal.Server.from_name(app.name, SERVER_NAME).get_url()
    if not url:
        raise ValueError("Deploy the GPU harness server before creating a session")
    token, session_id = secrets.token_urlsafe(32), uuid.uuid4().hex
    key = f"trial:{token_hash(token)}"
    workspace = f"gpu-harness-work-{session_id}"
    if limits.shared_workspace:
        modal.Volume.from_name(workspace, create_if_missing=True, version=2).hydrate()
    records[key] = {
        "id": session_id,
        "key": key,
        "run_id": run_id,
        "limits": limits.model_dump(),
        "boxes": [],
        "workspace_volume": workspace,
    }
    if limits.gpu_seconds is not None or limits.idle_seconds is not None:
        policy_index[key] = True
    return {
        "url": url,
        "token": token,
        "trial": session_id,
        "run_id": run_id,
        "limits": limits.model_dump(),
        "workspace_volume": workspace if limits.shared_workspace else None,
    }


@app.function(image=image, cpu=0.125, memory=256, min_containers=0, timeout=60)
def bind_agent(token: str, sandbox_id: str):
    """Trusted launcher RPC: identify the agent box for Volume visibility updates."""
    key = f"trial:{token_hash(token)}"
    trial = records[key]
    if trial.get("agent_box") or trial["boxes"]:
        raise ValueError("Bind the agent once, before GPU work starts")
    trial["agent_box"] = sandbox_id
    records[key] = trial


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(http_app(), host="0.0.0.0", port=8000)
