"""Opt-in CPU-only check: PYTHONPATH=src python tests/native_pull_fixture.py.

Bridges an in-process FastAPI app to a native Modal CPU Sandbox. No GPU session
or deployed harness is used. MODAL_SANDBOX_V2=0 selects the GPU-compatible V1 path.
"""

import asyncio
import json
from types import SimpleNamespace

import httpx
import modal

from gpu_harness.service import Service, create_api, token_hash


async def main():
    app = modal.App("gpu-harness-pull-fixture")
    image = modal.Image.debian_slim(python_version="3.12")
    async with app.run.aio():
        box = await modal.Sandbox.create.aio(
            "sleep", "infinity", app=app, image=image, cpu=0.25, memory=256, timeout=120
        )
        try:
            setup = await box.exec.aio(
                "python",
                "-c",
                "from pathlib import Path; "
                "p=Path('/tmp/pull-fixture'); p.mkdir(); "
                "(p/'file').write_bytes(bytes([0,255,10])); "
                "(p/'directory').mkdir(); "
                "(p/'file-link').symlink_to(p/'file'); "
                "(p/'directory-link').symlink_to(p/'directory')",
            )
            assert await setup.wait.aio() == 0
            token = "cpu-fixture-token"
            trial = {"id": "fixture"}
            records = {
                "trial:" + token_hash(token): trial,
                f"box:{box.object_id}": {"id": box.object_id, "trial": trial["id"]},
            }

            async def get(key):
                return records.get(key)

            async def no_policies():
                for key in ():
                    yield key

            service = Service(
                SimpleNamespace(get=SimpleNamespace(aio=get)), app, None,
                policy_index=SimpleNamespace(keys=SimpleNamespace(aio=no_policies)),
            )
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=create_api(service)),
                base_url="http://fixture",
                headers={"Authorization": f"Bearer {token}"},
            ) as client:
                for name in ("file", "file-link", "directory", "directory-link"):
                    path = "/tmp/pull-fixture/" + name
                    response = await client.get(f"/boxes/{box.object_id}/file", params={"path": path})
                    if name in {"file", "file-link"}:
                        assert response.status_code == 200 and response.content == bytes([0, 255, 10])
                        assert response.headers["Content-Length"] == "3"
                        detail = {"output_hex": response.content.hex()}
                    else:
                        assert response.status_code == 400
                        assert "requires a regular file" in response.json()["detail"]
                        detail = response.json()
                    print(json.dumps({"path": path, "status": response.status_code, **detail}), flush=True)
        finally:
            await box.terminate.aio()
            await box.wait.aio(raise_on_termination=False)


if __name__ == "__main__":
    asyncio.run(main())
