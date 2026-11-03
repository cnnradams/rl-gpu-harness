"""Trusted launcher helpers and CLI for creating GPU-access sessions."""

import argparse
import json
from pathlib import Path
from urllib.parse import urlsplit

import modal

from . import APP_NAME
from .service import Limits


def create_session(limits: Limits, run_id: str | None = None):
    return modal.Function.from_name(APP_NAME, "mint_session").remote(limits.model_dump(), run_id)


def credential_policy(session, proxy_secret_name: str):
    """Launcher-side: keep auth outside the agent using Modal's outbound policy."""
    from modal.experimental import OutboundPolicy

    return OutboundPolicy().with_header_replacement(
        domain=urlsplit(session["url"]).hostname,
        secret=modal.Secret.from_name(proxy_secret_name),
        headers={
            "Authorization": f"Bearer {session['token']}",
            "Modal-Key": "$MODAL_PROXY_TOKEN_ID",
            "Modal-Secret": "$MODAL_PROXY_TOKEN_SECRET",
        },
    )


def save_session(
    image_id: str = "",
    gpu: str = "H100",
    gpus_per_box: int = 1,
    max_boxes: int = 1,
    gpu_hours: float = 0,
    shared_workspace: bool = False,
    box_seconds: int = 7200,
    cpu: float = 4,
    memory_mib: int = 16384,
    block_network: bool = False,
    idle_seconds: int | None = None,
    idle_probe: str = "commands",
    output: str = "artifacts/session.json",
    run_only: bool = False,
):
    if run_only and not shared_workspace:
        raise ValueError("run_only requires shared_workspace")
    if not image_id:
        operator_app = modal.App.lookup(APP_NAME, create_if_missing=True)
        built = modal.Image.debian_slim(python_version="3.12").workdir("/workspace").build(operator_app)
        image_id = built.object_id
    limits = Limits(
        image_id=image_id,
        gpu=gpu,
        gpus_per_box=gpus_per_box,
        max_boxes=max_boxes,
        gpu_seconds=gpu_hours * 3600 if gpu_hours else None,
        shared_workspace=shared_workspace,
        run_only=run_only,
        box_seconds=box_seconds,
        cpu=cpu,
        memory=memory_mib,
        block_network=block_network,
        idle_seconds=idle_seconds,
        idle_probe=idle_probe,
    )
    session = create_session(limits)
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(mode=0o600, exist_ok=False)
    path.write_text(json.dumps(session, indent=2))
    print(f"Session {session['trial']}; URL and credential saved to {path}")


def main():
    parser = argparse.ArgumentParser(description="Mint one GPU session through the deployed Modal app")
    parser.add_argument("--gpu", default="H100")
    parser.add_argument("--gpus-per-box", type=int, default=1)
    parser.add_argument("--image-id", default="")
    parser.add_argument("--max-boxes", type=int, default=1)
    parser.add_argument("--box-seconds", type=int, default=7200)
    parser.add_argument("--gpu-hours", type=float, default=0)
    parser.add_argument("--cpu", type=float, default=4)
    parser.add_argument("--memory-mib", type=int, default=16384)
    parser.add_argument("--block-network", action="store_true")
    parser.add_argument("--shared-workspace", action="store_true")
    parser.add_argument("--run-only", action="store_true")
    parser.add_argument("--idle-seconds", type=int)
    parser.add_argument("--idle-probe", choices=["commands", "cuda"], default="commands")
    parser.add_argument("--output", default="artifacts/session.json")
    save_session(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
