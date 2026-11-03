import asyncio
import contextlib
import hashlib
import json
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import modal
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .transfer import download, upload


class Limits(BaseModel):
    model_config = ConfigDict(extra="forbid")
    gpu: str = "H100"
    gpus_per_box: int = Field(default=1, ge=1)
    max_boxes: int = Field(default=1, ge=1)
    box_seconds: int = Field(default=7200, ge=1)
    gpu_seconds: float | None = Field(default=None, gt=0)
    cpu: float = Field(default=4, gt=0)
    memory: int = Field(default=16384, gt=0)
    image_id: str
    shared_workspace: bool = False
    run_only: bool = False
    block_network: bool = False
    outbound_domain_allowlist: list[str] | None = None
    outbound_cidr_allowlist: list[str] | None = None
    environment_secret_name: str | None = None
    idle_seconds: int | None = Field(default=None, gt=0)
    idle_probe: Literal["commands", "cuda"] = "commands"


    @model_validator(mode="after")
    def require_shared_run_only(self):
        if self.run_only and not self.shared_workspace:
            raise ValueError("run_only requires shared_workspace")
        return self


class Exec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    argv: list[str] = Field(min_length=1)
    cwd: str = "/workspace"
    timeout: int | None = Field(default=None, gt=0)


def token_hash(token):
    return hashlib.sha256(token.encode()).hexdigest()


def require_worker_access(trial):
    if trial["limits"].get("run_only"):
        raise HTTPException(403, "This session only supports GPU access through run")


class Service:
    def __init__(self, records, app, trace_volume, trace_root="/traces", *, policy_index):
        self.records, self.app, self.volume = records, app, trace_volume
        self.policy_index = policy_index
        self.root = Path(trace_root)
        self.lock = asyncio.Lock()
        self.processes, self.readers = {}, set()
        self.budget_task = None

    def check_budget_monitor(self, limits):
        if (
            (limits.gpu_seconds is not None or limits.idle_seconds is not None)
            and self.budget_task
            and self.budget_task.done()
        ):
            raise HTTPException(503, "Resource-policy monitor stopped; operator attention required")

    async def closed(self, trial):
        # A separate key lets the trusted launcher close access without racing
        # updates to the session's box list during provisioning.
        return trial.get("closed") or await self.records.get.aio(f"closed:{trial['id']}")

    async def event(self, owner, kind, **data):
        folder = self.root / owner["id"]
        folder.mkdir(parents=True, exist_ok=True)
        # One service container writes traces. Each write closes the file before commit.
        with (folder / "events.jsonl").open("a") as stream:
            stream.write(json.dumps({"time": time.time(), "kind": kind, **data}) + "\n")
        await self.volume.commit.aio()

    async def owned(self, trial, kind, resource_id):
        value = await self.records.get.aio(f"{kind}:{resource_id}")
        if not value or value["trial"] != trial["id"]:
            raise HTTPException(404, f"Unknown {kind} ID for this session")
        return value

    async def claim(self, trial, key, operation, arguments):
        if not key or len(key) > 200:
            raise HTTPException(422, "Supply an Idempotency-Key of 1–200 characters")
        record_key = f"request:{trial['id']}:{key}"
        claim = {"operation": operation, "arguments": arguments, "state": "pending"}
        if await self.records.put.aio(record_key, claim, skip_if_exists=True):
            return record_key, None
        previous = await self.records.get.aio(record_key)
        if previous["operation"] != operation or previous["arguments"] != arguments:
            raise HTTPException(409, "Request ID already used with different arguments")
        if "result" not in previous:
            raise HTTPException(
                409, f"Request outcome is {previous['state']}; do not submit a new ID blindly"
            )
        return record_key, previous["result"]

    async def finish(self, key, result):
        record = await self.records.get.aio(key)
        await self.records.put.aio(key, {**record, "state": "done", "result": result})
        return result

    async def provisioning_failed(self, trial, key, request_key, error):
        record = await self.records.get.aio(request_key)
        detail = f"{type(error).__name__}: {error}"
        await self.records.put.aio(request_key, {**record, "state": "failed", "error": detail})
        await self.event(trial, "provisioning_failed", request_id=key, error=detail)

    async def inventory(self, trial):
        boxes = [await self.records.get.aio(f"box:{box_id}") for box_id in trial["boxes"]]
        for box in boxes:
            if box.get("ready_at") and not box.get("ended_at"):
                try:
                    sb = await modal.Sandbox.from_id.aio(box["id"])
                    code = await sb.poll.aio()
                except modal.exception.Error:
                    box.setdefault("unavailable_since", box.get("last_available_at", time.time()))
                else:
                    now = time.time()
                    if code is not None:
                        # Conservatively stop charging at the last confirmed
                        # access, rather than charging until death is discovered.
                        box["ended_at"] = box.get("unavailable_since", box.get("last_available_at", now))
                    else:
                        since = box.pop("unavailable_since", None)
                        if since is not None:
                            box["unavailable_seconds"] = box.get("unavailable_seconds", 0) + now - since
                        box["last_available_at"] = now
                await self.records.put.aio(f"box:{box['id']}", box)
        used = sum(
            max(
                0,
                b.get("ended_at", b.get("unavailable_since", time.time()))
                - b["ready_at"]
                - b.get("unavailable_seconds", 0),
            )
            * b["gpus"]
            for b in boxes
            if b.get("ready_at")
        )
        return boxes, used

    async def create(self, trial, key, command=None):
        if command is None:
            require_worker_access(trial)
        # Only admission is serialized; provisioning must not block other GPU work.
        async with self.lock:
            trial = await self.records.get.aio(trial["key"])
            request_key, previous = await self.claim(
                trial, key, "run" if command else "create", command.model_dump() if command else {}
            )
            if previous is not None:
                return previous
            boxes, used = await self.inventory(trial)
            limits = Limits(**trial["limits"])
            self.check_budget_monitor(limits)
            if await self.closed(trial) or limits.gpu_seconds is not None and used >= limits.gpu_seconds:
                raise HTTPException(409, "GPU access has ended for this trial")
            if sum(not b.get("ended_at") for b in boxes) + trial.get("creating", 0) >= limits.max_boxes:
                raise HTTPException(409, "Box limit reached")
            trial["creating"] = trial.get("creating", 0) + 1
            await self.records.put.aio(trial["key"], trial)
        # No GPU seconds are reserved. Provisioning failures are infrastructure
        # failures: release admission capacity and retain the error for inspection.
        requested_at = time.time()
        box_timeout = (
            min(limits.box_seconds, command.timeout) if command and command.timeout else limits.box_seconds
        )
        try:
            if limits.run_only:
                await self.sync_workspace(trial, commit_only=True)
            volumes = (
                {"/workspace": modal.Volume.from_name(trial["workspace_volume"])}
                if limits.shared_workspace
                else {}
            )
            image = modal.Image.from_id(limits.image_id)
            if command:
                image = image.entrypoint([])
            sb = await modal.Sandbox.create.aio(
                *(command.argv if command else ["sleep", "infinity"]),
                app=self.app,
                image=image,
                gpu=f"{limits.gpu}:{limits.gpus_per_box}",
                cpu=limits.cpu,
                memory=limits.memory,
                timeout=box_timeout,
                readiness_probe=modal.Probe.with_exec("true"),
                volumes=volumes,
                block_network=limits.block_network,
                outbound_domain_allowlist=limits.outbound_domain_allowlist,
                outbound_cidr_allowlist=limits.outbound_cidr_allowlist,
                secrets=([modal.Secret.from_name(limits.environment_secret_name)]
                         if limits.environment_secret_name else []),
                # The idle process must not hold the shared mount busy.
                workdir=command.cwd if command else ("/" if limits.shared_workspace else None),
            )
        except BaseException as error:
            async with self.lock:
                current = await self.records.get.aio(trial["key"])
                current["creating"] -= 1
                await self.records.put.aio(trial["key"], current)
            await self.provisioning_failed(trial, key, request_key, error)
            raise
        box = {
            "id": sb.object_id,
            "trial": trial["id"],
            "gpus": limits.gpus_per_box,
            "requested_at": requested_at,
            "request_id": key,
            "readiness_observed": False,
        }
        async with self.lock:
            trial = await self.records.get.aio(trial["key"])
            trial["creating"] -= 1
            trial["boxes"].append(sb.object_id)
            await self.records.put.aio(f"box:{sb.object_id}", box)
            await self.records.put.aio(trial["key"], trial)
        try:
            if command is None:
                process = await sb.exec.aio("mkdir", "-p", "/workspace")
                if await process.wait.aio() != 0:
                    raise RuntimeError("Could not prepare workspace")
            try:
                await sb.wait_until_ready.aio(timeout=box_timeout)
                box["ready_at"] = time.time()
                box["readiness_observed"] = True
            except Exception as error:
                # A terminated router may precede the final SandboxWait result.
                if command is not None and isinstance(error, modal.exception.ConflictError):
                    with contextlib.suppress(modal.exception.SandboxTimeoutError):
                        await sb.wait.aio(raise_on_termination=False)
                # A native main may finish before readiness is observed. Its
                # result is still valid; its allocation interval is unknown.
                elif command is None or await sb.poll.aio() is None:
                    raise
                box["ended_at"] = time.time()
            async with self.lock:
                trial = await self.records.get.aio(trial["key"])
                _, used = await self.inventory(trial)
                if not box.get("ended_at") and (
                    await self.closed(trial) or limits.gpu_seconds is not None and used >= limits.gpu_seconds
                ):
                    raise HTTPException(409, "GPU access ended while this box provisioned")
                if box["readiness_observed"]:
                    box["last_available_at"] = box["ready_at"]
                await self.records.put.aio(f"box:{sb.object_id}", box)
        except BaseException as error:
            await sb.terminate.aio()
            box["ended_at"] = time.time()
            await self.records.put.aio(f"box:{sb.object_id}", box)
            await self.provisioning_failed(trial, key, request_key, error)
            raise
        await self.event(
            trial,
            "box_ready" if box["readiness_observed"] else "box_readiness_unobserved",
            **box,
        )
        if command is None:
            return await self.finish(request_key, box)
        job = {"id": uuid.uuid4().hex, "trial": trial["id"], "box": sb.object_id, "state": "running"}
        self.processes[job["id"]] = sb
        await self.records.put.aio(f"command:{job['id']}", job)
        await self.event(
            trial, "run", command=job["id"], box=sb.object_id, request_id=key, **command.model_dump()
        )
        task = asyncio.create_task(self.collect(trial, job, sb, sandbox=True))
        self.readers.add(task)
        task.add_done_callback(self.readers.discard)
        return await self.finish(request_key, job)

    async def execute(self, trial, box_id, request, key):
        require_worker_access(trial)
        box = await self.owned(trial, "box", box_id)
        request_key, previous = await self.claim(trial, key, "exec", {"box": box_id, **request.model_dump()})
        if previous is not None:
            return previous
        async with self.lock:
            current = await self.records.get.aio(trial["key"])
            self.check_budget_monitor(Limits(**current["limits"]))
            _, used = await self.inventory(current)
            budget = current["limits"]["gpu_seconds"]
            if await self.closed(current) or budget is not None and used >= budget:
                raise HTTPException(409, "GPU access has ended for this trial")
        sb = await modal.Sandbox.from_id.aio(box_id)
        process = await sb.exec.aio(*request.argv, workdir=request.cwd, timeout=request.timeout, text=False)
        command_id = uuid.uuid4().hex
        command = {"id": command_id, "trial": trial["id"], "box": box["id"], "state": "running"}
        self.processes[command_id] = process
        await self.records.put.aio(f"command:{command_id}", command)
        await self.event(
            trial, "exec", command=command_id, box=box_id, request_id=key, **request.model_dump()
        )
        task = asyncio.create_task(self.collect(trial, command, process))
        self.readers.add(task)
        task.add_done_callback(self.readers.discard)
        return await self.finish(request_key, command)

    async def sync_workspace(self, trial, box_id=None, *, commit_only=False):
        if box_id is not None:
            require_worker_access(trial)
            await self.owned(trial, "box", box_id)
        if not trial["limits"].get("shared_workspace"):
            raise HTTPException(409, "This session does not have a shared workspace")
        if not trial.get("agent_box"):
            raise HTTPException(409, "Shared workspace requires a bound agent sandbox")
        progress = {"box": box_id, "committed": [], "reloaded": []}
        participants = [("agent", trial["agent_box"])]
        if box_id is not None:
            participants.append(("gpu", box_id))
        # Legacy box sync commits both views; CPU-only sync does not touch GPU jobs.
        for phase in (("commit",) if commit_only else ("commit", "reload")):
            for side, sandbox_id in participants:
                try:
                    sandbox = await modal.Sandbox.from_id.aio(sandbox_id)
                    if phase == "commit":
                        process = await sandbox.exec.aio("sync", "/workspace")
                        if await process.wait.aio() != 0:
                            raise HTTPException(502, await process.stderr.read.aio())
                    else:
                        await sandbox.reload_volumes.aio()
                except (modal.exception.Error, HTTPException) as error:
                    detail = {
                        **progress,
                        "failed_side": side,
                        "failed_phase": phase,
                        "error": error.detail if isinstance(error, HTTPException) else str(error),
                    }
                    await self.event(trial, "workspace_sync_failed", **detail)
                    status = 409 if isinstance(error, modal.exception.ConflictError) else 502
                    raise HTTPException(status, detail) from error
                progress["committed" if phase == "commit" else "reloaded"].append(side)
        await self.event(trial, "workspace_synced", **progress)
        return progress

    async def collect(self, trial, command, process, *, sandbox=False):
        async def copy(reader, name):
            path = self.root / trial["id"] / f"{command['id']}.{name}"
            async for chunk in reader:
                with path.open("ab") as stream:
                    stream.write(chunk.encode() if isinstance(chunk, str) else chunk)

        try:
            # Commands have no stdin forwarding; do not leave readers waiting for input.
            with contextlib.suppress(
                modal.exception.ConflictError, modal.exception.NotFoundError,
                BrokenPipeError, ConnectionResetError,
            ):
                process.stdin.write_eof()
                await process.stdin.drain.aio()
            captured = await asyncio.gather(
                copy(process.stdout, "stdout"), copy(process.stderr, "stderr"), return_exceptions=True
            )
            # A log-reader error does not establish whether the native process exited.
            errors = [
                f"{name}: {type(error).__name__}: {error}"
                for name, error in zip(("stdout", "stderr"), captured, strict=True)
                if isinstance(error, Exception)
            ]
            if errors:
                command["error"] = "Output capture failed (" + "; ".join(errors) + ")"
            if sandbox:
                with contextlib.suppress(modal.exception.SandboxTimeoutError):
                    await process.wait.aio(raise_on_termination=False)
                code = await process.poll.aio()
                if code is None:
                    raise RuntimeError("Sandbox exit is not confirmed")
                box = await self.owned(trial, "box", command["box"])
                box.setdefault("ended_at", box.get("unavailable_since", time.time()))
                await self.records.put.aio(f"box:{box['id']}", box)
            else:
                code = await process.wait.aio()
            command.update(state="succeeded" if code == 0 else "failed", exit_code=code)
        except Exception as error:
            command.update(state="failed", error=f"{type(error).__name__}: {error}")
        await self.records.put.aio(f"command:{command['id']}", command)
        self.processes.pop(command["id"], None)
        await self.event(trial, "command_finished", **command)

    async def stop(self, trial, box_id):
        async with self.lock:
            box = await self.owned(trial, "box", box_id)
            await self._stop_box(box)
        await self.event(trial, "box_stop", box=box_id)

    async def _stop_box(self, box):
        # Caller holds the admission lock. terminate() requests a stop without
        # waiting for worker exit or the background output collector.
        box_id = box["id"]
        sb = await modal.Sandbox.from_id.aio(box_id)
        await sb.terminate.aio()
        box.setdefault("ended_at", box.get("unavailable_since", time.time()))
        await self.records.put.aio(f"box:{box_id}", box)

    async def enforce_budgets(self):
        # Enforce requested policies. No automatic replacement or process restart.
        # GPU-process probes must not delay the budget clock's next pass.
        async with asyncio.TaskGroup() as monitors:
            monitors.create_task(self.policy_loop(idle_checks=False))
            monitors.create_task(self.policy_loop(idle_checks=True))

    async def policy_trials(self):
        async def read(key):
            trial = await self.records.get.aio(key)
            if trial is None or await self.closed(trial):
                await self.policy_index.pop.aio(key, None)
                return None
            return trial

        keys = [key async for key in self.policy_index.keys.aio()]
        return [trial for trial in await asyncio.gather(*(read(key) for key in keys)) if trial]

    async def policy_loop(self, *, idle_checks):
        while True:
            trials = await self.policy_trials()
            running = set()
            # Only the legacy command-idle policy needs command history.
            # Budget and CUDA-idle passes enumerate the small active index only.
            if idle_checks and any(
                t["limits"]["idle_seconds"] is not None and t["limits"]["idle_probe"] == "commands"
                for t in trials
            ):
                running = {
                    value["box"]
                    async for key, value in self.records.items.aio()
                    if key.startswith("command:") and value["state"] == "running"
                }
            # A slow GPU probe must not delay another trial's budget check.
            # TaskGroup keeps each pass's checks inside the monitor lifetime.
            async with asyncio.TaskGroup() as checks:
                for trial in trials:
                    checks.create_task(self.check_policy(trial, running, idle_checks=idle_checks))
            await asyncio.sleep(1)

    async def check_policy(self, trial, running, *, idle_checks):
        limits = Limits(**trial["limits"])
        if (limits.idle_seconds if idle_checks else limits.gpu_seconds) is None:
            return
        if await self.closed(trial):
            return
        async with self.lock:
            boxes, used = await self.inventory(trial)
        for box in boxes:
            if not box.get("ready_at") or box.get("ended_at"):
                continue
            if not idle_checks and used >= limits.gpu_seconds:
                await self.stop(trial, box["id"])
            elif idle_checks:
                await self.check_idle(trial, box, limits, box["id"] in running)

    async def check_idle(self, trial, box, limits, command_running):
        now = time.time()
        if limits.idle_probe == "cuda":
            if now - box.get("last_idle_probe", 0) < 10:
                return
            box["last_idle_probe"] = now
            try:
                sb = await modal.Sandbox.from_id.aio(box["id"])
                probe = await sb.exec.aio(
                    "nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits", timeout=10
                )
                output = await probe.stdout.read.aio()
                if await probe.wait.aio() != 0:
                    raise RuntimeError(await probe.stderr.read.aio())
                busy = bool(output.strip())
            except Exception as error:
                # Unknown GPU state is not evidence of idleness.
                async with self.lock:
                    box = await self.owned(trial, "box", box["id"])
                    box["last_idle_probe"] = now
                    box.pop("idle_since", None)
                    await self.records.put.aio(f"box:{box['id']}", box)
                await self.event(trial, "idle_probe_error", box=box["id"], error=str(error))
                return
        else:
            busy = command_running
        released = False
        async with self.lock:
            # Re-read lifecycle state after the probe, before applying its result.
            box = await self.owned(trial, "box", box["id"])
            if box.get("ended_at"):
                return
            if limits.idle_probe == "cuda":
                box["last_idle_probe"] = now
            now = time.time()
            if busy:
                box.pop("idle_since", None)
            else:
                box.setdefault("idle_since", now)
            await self.records.put.aio(f"box:{box['id']}", box)
            if not busy and now - box["idle_since"] >= limits.idle_seconds:
                await self._stop_box(box)
                released = True
        if released:
            await self.event(
                trial,
                "idle_release",
                box=box["id"],
                probe=limits.idle_probe,
                idle_seconds=now - box["idle_since"],
            )
            await self.event(trial, "box_stop", box=box["id"])


def create_api(service):
    @asynccontextmanager
    async def lifespan(api):
        task = asyncio.create_task(service.enforce_budgets())
        service.budget_task = task
        try:
            yield
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    api = FastAPI(title="GPU harness", lifespan=lifespan)

    @api.exception_handler(modal.exception.Error)
    async def modal_error(request, error):
        return JSONResponse({"error": type(error).__name__, "detail": str(error)}, status_code=502)

    async def trial(authorization: str = Header(default="")):
        token = authorization.removeprefix("Bearer ")
        result = await service.records.get.aio(f"trial:{token_hash(token)}")
        if not result or not authorization.startswith("Bearer "):
            raise HTTPException(401, "Invalid trial credential")
        return result

    @api.get("/policy")
    async def policy(t=Depends(trial)):
        boxes, used = await service.inventory(t)
        return {
            "trial": t["id"], "limits": t["limits"], "boxes": boxes, "gpu_seconds_used": used,
            "gpu_time_accounting_complete": all(box.get("readiness_observed", False) for box in boxes),
        }

    @api.post("/boxes")
    async def create(idempotency_key: str = Header(), t=Depends(trial)):
        return await service.create(t, idempotency_key)

    @api.get("/requests/{request_id}")
    async def request_status(request_id: str, t=Depends(trial)):
        record = await service.records.get.aio(f"request:{t['id']}:{request_id}")
        if record is None:
            raise HTTPException(404, "Unknown request ID")
        return record

    @api.delete("/boxes/{box_id}")
    async def stop(box_id: str, t=Depends(trial)):
        await service.stop(t, box_id)
        return {"stopped": box_id}

    @api.post("/boxes/{box_id}/exec")
    async def execute(box_id: str, command: Exec, idempotency_key: str = Header(), t=Depends(trial)):
        return await service.execute(t, box_id, command, idempotency_key)

    @api.post("/run")
    async def run(command: Exec, idempotency_key: str = Header(), t=Depends(trial)):
        return await service.create(t, idempotency_key, command)

    @api.get("/commands/{command_id}")
    async def status(command_id: str, t=Depends(trial)):
        # Completion can remove the handle while the Dict read is in flight.
        local_handle_present = command_id in service.processes
        command = await service.owned(t, "command", command_id)
        if command["state"] == "running" and not local_handle_present:
            return {**command, "state": "unknown", "error": "Service restarted; inspect the box directly"}
        return command

    @api.get("/commands/{command_id}/logs")
    async def logs(command_id: str, stream: str = "stdout", offset: int = 0, t=Depends(trial)):
        await service.owned(t, "command", command_id)
        if stream not in {"stdout", "stderr"} or offset < 0:
            raise HTTPException(422, "Invalid stream or offset")
        path = service.root / t["id"] / f"{command_id}.{stream}"
        with path.open("rb") if path.exists() else contextlib.nullcontext() as file:
            if file:
                file.seek(offset)
            return Response(file.read(1024 * 1024) if file else b"", media_type="application/octet-stream")

    @api.put("/boxes/{box_id}/file")
    async def put_file(
        box_id: str, path: str, request: Request, content_length: int = Header(ge=0), t=Depends(trial)
    ):
        require_worker_access(t)
        await service.owned(t, "box", box_id)
        sb = await modal.Sandbox.from_id.aio(box_id)
        result = await upload(sb, path, content_length, request.stream())
        await service.event(t, "file_written", box=box_id, path=path, **result)
        return result

    @api.get("/boxes/{box_id}/file")
    async def get_file(box_id: str, path: str, t=Depends(trial)):
        require_worker_access(t)
        await service.owned(t, "box", box_id)
        sb = await modal.Sandbox.from_id.aio(box_id)
        size, chunks = await download(sb, path)
        return StreamingResponse(
            chunks, media_type="application/octet-stream", headers={"Content-Length": str(size)}
        )

    @api.post("/boxes/{box_id}/refresh")
    async def refresh(box_id: str, t=Depends(trial)):
        require_worker_access(t)
        await service.owned(t, "box", box_id)
        sb = await modal.Sandbox.from_id.aio(box_id)
        await sb.reload_volumes.aio()
        return {"refreshed": box_id}

    @api.post("/sync")
    async def sync_agent(t=Depends(trial)):
        return await service.sync_workspace(t)

    @api.post("/boxes/{box_id}/sync")
    async def sync(box_id: str, t=Depends(trial)):
        return await service.sync_workspace(t, box_id)

    return api
