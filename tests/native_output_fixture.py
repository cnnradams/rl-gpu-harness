"""Opt-in CPU-only Modal check: PYTHONPATH=src python tests/native_output_fixture.py.

Uses an ephemeral app and two CPU sandboxes; no deployed harness or GPU session.
Run with MODAL_SANDBOX_V2=0 to exercise the V1 backend used by GPU sandboxes.
V1 entrypoint logs replace invalid UTF-8; binary preservation is available for exec.
"""

import asyncio
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import modal

from gpu_harness.service import Service


class Records:
    def __init__(self):
        self.data = {}
        self.get = SimpleNamespace(aio=self.read)
        self.put = SimpleNamespace(aio=self.write)

    async def read(self, key):
        return self.data.get(key)

    async def write(self, key, value):
        self.data[key] = dict(value)


async def main():
    app = modal.App("gpu-harness-output-fixture")
    image = modal.Image.debian_slim(python_version="3.12")
    with TemporaryDirectory() as folder:
        async with app.run.aio():
            service = Service(Records(), app, None, folder, policy_index=Records())
            trial = {"id": "fixture", "limits": {}}
            (Path(folder) / trial["id"]).mkdir()

            async def event(*args, **kwargs):
                pass

            service.event = event

            async def collect(process, box, name, sandbox=False):
                command = {"id": name, "trial": trial["id"], "box": box.object_id, "state": "running"}
                service.processes[name] = process
                await service.records.put.aio(
                    f"box:{box.object_id}", {"id": box.object_id, "trial": trial["id"]}
                )
                await service.collect(trial, command, process, sandbox=sandbox)
                assert command["state"] == "succeeded" and command["exit_code"] == 0, command
                print(json.dumps(command), flush=True)
                return command

            box = await modal.Sandbox.create.aio(
                "sleep", "infinity", app=app, image=image, cpu=0.25, memory=256, timeout=120
            )
            try:
                text_process = await box.exec.aio(
                    "python", "-c", "import sys; sys.stdout.buffer.write(bytes([255]))"
                )
                try:
                    await text_process.stdout.read.aio()
                except UnicodeDecodeError as error:
                    print(json.dumps({"native_text_exec_error": str(error)}), flush=True)
                else:
                    raise AssertionError("Expected the native text exec reader to reject invalid UTF-8")
                assert await text_process.wait.aio() == 0
                process = await box.exec.aio(
                    "python", "-c", "import sys; sys.stdout.buffer.write(bytes([0,255,254,10]))", text=False
                )
                result = await collect(process, box, "exec")
                assert "error" not in result
                assert (Path(folder) / "fixture/exec.stdout").read_bytes() == bytes([0, 255, 254, 10])
            finally:
                await box.terminate.aio()
                await box.wait.aio(raise_on_termination=False)

            box = await modal.Sandbox.create.aio(
                "python",
                "-c",
                "import sys,time; sys.stdout.buffer.write(bytes([255])); sys.stdout.flush(); time.sleep(2)",
                app=app,
                image=image,
                cpu=0.25,
                memory=256,
                timeout=120,
            )
            try:
                result = await collect(box, box, "run", sandbox=True)
                output = Path(folder) / "fixture/run.stdout"
                print(
                    json.dumps({"run_output_hex": output.read_bytes().hex() if output.exists() else None}),
                    flush=True,
                )
                assert await box.poll.aio() == 0
            finally:
                await box.terminate.aio()
                await box.wait.aio(raise_on_termination=False)


if __name__ == "__main__":
    asyncio.run(main())
