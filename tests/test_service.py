import asyncio
import copy
import json
import secrets
from types import SimpleNamespace

import httpx
import pytest
from fastapi import HTTPException

import gpu_harness.service as code
from gpu_harness.service import Exec, Limits, Service, create_api, token_hash


def method(function):
    return SimpleNamespace(aio=function)


class Dict:
    def __init__(self):
        self.data = {}
        self.get = method(self._get)
        self.put = method(self._put)
        self.items = method(self._items)
        self.keys = method(self._keys)
        self.pop = method(self._pop)

    async def _get(self, key):
        return copy.deepcopy(self.data.get(key))

    async def _put(self, key, value, skip_if_exists=False):
        if skip_if_exists and key in self.data:
            return False
        self.data[key] = copy.deepcopy(value)
        return True

    async def _keys(self):
        for key in list(self.data):
            yield key

    async def _pop(self, key, default=None):
        return self.data.pop(key, default)

    async def _items(self):
        for item in list(self.data.items()):
            yield copy.deepcopy(item)


class Process:
    def __init__(self, output="hello\n", complete=True):
        self.done = asyncio.Event()
        if complete:
            self.done.set()

        async def chunks():
            yield output
            await self.done.wait()

        async def empty():
            await self.done.wait()
            if False:
                yield ""

        async def wait():
            await self.done.wait()
            return 0

        async def drain():
            pass

        self.stdin = SimpleNamespace(write_eof=lambda: None, drain=method(drain))
        self.stdout, self.stderr, self.wait = chunks(), empty(), method(wait)


class Box:
    def __init__(self, box_id):
        self.object_id, self.ended, self.files, self.commands = box_id, False, {}, []
        self.wait_until_ready = method(self.ready)
        self.exec = method(self.execute)
        self.poll = method(self.status)
        self.terminate = method(self.stop)
        self.reload_volumes = method(self.refresh)
        self.filesystem = SimpleNamespace(write_bytes=method(self.write), read_bytes=method(self.read))
        self.next_process = None
        self.main_process = Process(complete=False)
        self.stdout, self.stderr = self.main_process.stdout, self.main_process.stderr
        self.stdin = self.main_process.stdin
        self.wait = method(self.wait_for_exit)

    async def wait_for_exit(self, **kwargs):
        await self.main_process.done.wait()
        self.ended = True

    async def ready(self, *, timeout):
        self.readiness_timeout = timeout

    async def execute(self, *argv, **kwargs):
        if self.ended:
            raise RuntimeError("Sandbox terminated")
        self.commands.append((argv, kwargs))
        return self.next_process or Process()

    async def status(self):
        return 0 if self.ended else None

    async def stop(self):
        self.ended = True
        self.main_process.done.set()

    async def write(self, data, path):
        self.files[path] = data

    async def read(self, path):
        return self.files[path]

    async def refresh(self):
        self.refresh_count = getattr(self, "refresh_count", 0) + 1


@pytest.fixture
async def system(tmp_path, monkeypatch):
    boxes = {}

    async def create(*args, **kwargs):
        box = Box(f"sb-{len(boxes)}")
        box.entrypoint, box.options = args, kwargs
        boxes[box.object_id] = box
        return box

    async def lookup(box_id):
        return boxes[box_id]

    async def commit():
        pass

    monkeypatch.setattr(code.modal, "Sandbox", SimpleNamespace(create=method(create), from_id=method(lookup)))

    def from_id(image_id):
        image = SimpleNamespace(object_id=image_id, entrypoint_args=None)

        def entrypoint(argv):
            image.entrypoint_args = argv
            return image

        image.entrypoint = entrypoint
        return image

    monkeypatch.setattr(code.modal, "Image", SimpleNamespace(from_id=from_id))
    service = Service(
        Dict(), object(), SimpleNamespace(commit=method(commit)), str(tmp_path), policy_index=Dict()
    )
    yield service, boxes
    for task in service.readers:
        task.cancel()
    await asyncio.gather(*service.readers, return_exceptions=True)


async def trial(service, **kwargs):
    token = secrets.token_urlsafe()
    key = f"trial:{token_hash(token)}"
    record = {
        "id": token_hash(token)[:10],
        "key": key,
        "limits": Limits(image_id="im-test", **kwargs).model_dump(),
        "boxes": [],
        "workspace_volume": "test",
    }
    await service.records.put.aio(key, record)
    if record["limits"]["gpu_seconds"] is not None or record["limits"]["idle_seconds"] is not None:
        await service.policy_index.put.aio(key, True)
    return record, token


async def test_run_uses_native_entrypoint_and_exec_works_only_during_its_lifetime(system):
    service, boxes = system
    t, _ = await trial(service)
    request = Exec(argv=["python", "train.py"], cwd="/tmp")
    job = await service.create(t, "same-request", request)
    native = boxes[job["box"]]
    assert native.entrypoint == ("python", "train.py")
    assert native.options["image"].entrypoint_args == []
    assert native.options["workdir"] == "/tmp"
    assert native.commands == []  # The provider runs its readiness probe; no service-side supervisor.
    assert native.options["readiness_probe"].exec_argv == ("true",)
    assert native.readiness_timeout == native.options["timeout"]
    assert await service.create(t, "same-request", request) == job
    assert len(boxes) == 1
    extra = await service.execute(t, job["box"], Exec(argv=["nvidia-smi"]), "extra")
    while (await service.records.get.aio(f"command:{extra['id']}"))["state"] == "running":
        await asyncio.sleep(0)
    assert not native.ended
    with pytest.raises(HTTPException) as full:
        await service.create(t, "overlap", request)
    assert full.value.status_code == 409
    native.main_process.done.set()
    await asyncio.gather(*service.readers)
    assert native.ended
    assert (await service.records.get.aio(f"command:{job['id']}"))["state"] == "succeeded"
    with pytest.raises(RuntimeError, match="terminated"):
        await service.execute(t, job["box"], Exec(argv=["nvidia-smi"]), "after-exit")
    next_job = await service.create(t, "next", request)
    assert next_job["box"] != job["box"]


@pytest.mark.parametrize("command_timeout,expected", [(None, 7200), (900, 900)])
async def test_native_run_budget_waits_for_probe_with_configured_lifetime(
    system, monkeypatch, command_timeout, expected
):
    service, boxes = system
    now = [1000]
    monkeypatch.setattr(code.time, "time", lambda: now[0])
    original = code.modal.Sandbox.create.aio
    original_inventory = service.inventory
    inventory_calls = 0

    async def delayed_inventory(t):
        nonlocal inventory_calls
        inventory_calls += 1
        if inventory_calls == 2:
            now[0] += 30  # Bookkeeping after readiness is allocated time.
        return await original_inventory(t)

    monkeypatch.setattr(service, "inventory", delayed_inventory)

    async def delayed(*args, **kwargs):
        box = await original(*args, **kwargs)

        async def ready(*, timeout):
            assert timeout == expected
            now[0] += 120  # Startup exceeds the budget, but is not allocated time.

        box.wait_until_ready = method(ready)
        return box

    monkeypatch.setattr(code.modal.Sandbox.create, "aio", delayed)
    t, _ = await trial(service, gpu_seconds=60)
    job = await service.create(t, "run", Exec(argv=["train"], timeout=command_timeout))
    box = await service.owned(t, "box", job["box"])
    assert box["ready_at"] == 1120 and box["readiness_observed"]
    assert box["last_available_at"] == 1120
    assert box["request_id"] == "run"
    events = [json.loads(line) for line in (service.root / t["id"] / "events.jsonl").read_text().splitlines()]
    assert next(event for event in events if event["kind"] == "box_ready")["request_id"] == "run"
    assert (await service.inventory(await service.records.get.aio(t["key"])))[1] == 30
    assert boxes[job["box"]].entrypoint == ("train",)


@pytest.mark.parametrize("exit_code", [0, 7])
async def test_completed_native_main_preserves_result_when_readiness_observer_arrives_late(
    system, monkeypatch, exit_code
):
    service, _ = system
    original = code.modal.Sandbox.create.aio

    async def finished(*args, **kwargs):
        box = await original(*args, **kwargs)
        box.main_process = Process("native output\n")
        box.stdout, box.stderr = box.main_process.stdout, box.main_process.stderr

        async def ready(*, timeout):
            raise code.modal.exception.ConflictError("Task has already finished")

        async def status():
            return exit_code

        box.wait_until_ready, box.poll = method(ready), method(status)
        return box

    monkeypatch.setattr(code.modal.Sandbox.create, "aio", finished)
    t, token = await trial(service, gpu_seconds=60)
    request = Exec(argv=["fast-command"])
    job = await service.create(t, "run", request)
    await asyncio.gather(*service.readers)
    result = await service.owned(t, "command", job["id"])
    assert result["exit_code"] == exit_code
    assert result["state"] == ("succeeded" if exit_code == 0 else "failed")
    assert (service.root / t["id"] / f"{job['id']}.stdout").read_text() == "native output\n"
    box = await service.owned(t, "box", job["box"])
    assert box["readiness_observed"] is False and "ready_at" not in box
    assert (await service.create(t, "run", request))["id"] == job["id"]
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=create_api(service)), base_url="http://test"
    ) as client:
        policy = (await client.get("/policy", headers={"Authorization": f"Bearer {token}"})).json()
    assert policy["gpu_seconds_used"] == 0  # Observed lower bound, explicitly incomplete.
    assert policy["gpu_time_accounting_complete"] is False
    events = (service.root / t["id"] / "events.jsonl").read_text()
    assert "box_readiness_unobserved" in events and "provisioning_failed" not in events
    readiness_event = next(
        event for event in map(json.loads, events.splitlines()) if event["kind"] == "box_readiness_unobserved"
    )
    assert readiness_event["request_id"] == box["request_id"] == "run"


@pytest.mark.parametrize("exit_code", [0, 127])
async def test_readiness_conflict_waits_for_late_native_exit_result(system, monkeypatch, exit_code):
    service, boxes = system
    original = code.modal.Sandbox.create.aio
    result_available = False
    polls = []

    async def finished(*args, **kwargs):
        box = await original(*args, **kwargs)
        output = Process("fast native output\n")
        box.stdout, box.stderr = output.stdout, output.stderr

        async def ready(*, timeout):
            raise code.modal.exception.ConflictError("Sandbox has already terminated")

        async def status():
            polls.append(result_available)
            return exit_code if result_available else None

        async def wait(**kwargs):
            nonlocal result_available
            result_available = True
            box.ended = True

        box.wait_until_ready, box.poll, box.wait = method(ready), method(status), method(wait)
        return box

    monkeypatch.setattr(code.modal.Sandbox.create, "aio", finished)
    t, _ = await trial(service)
    job = await service.create(t, "fast", Exec(argv=["fast-command"]))
    assert result_available and polls == []  # Classification trusts the definitive wait, not another poll.
    await asyncio.gather(*service.readers)
    result = await service.owned(t, "command", job["id"])
    assert result["exit_code"] == exit_code
    assert result["state"] == ("succeeded" if exit_code == 0 else "failed")
    assert (service.root / t["id"] / f"{job['id']}.stdout").read_text() == "fast native output\n"
    box = await service.owned(t, "box", job["box"])
    assert not box["readiness_observed"] and "ready_at" not in box
    assert box["request_id"] == "fast" and boxes[job["box"]].ended
    events = (service.root / t["id"] / "events.jsonl").read_text()
    assert "box_readiness_unobserved" in events and "provisioning_failed" not in events


async def test_native_probe_failure_on_running_main_remains_infrastructure_failure(system, monkeypatch):
    service, boxes = system

    async def failed(*, timeout):
        raise TimeoutError("readiness not reached")

    monkeypatch.setattr(Box, "ready", lambda self, **kwargs: failed(**kwargs))
    t, _ = await trial(service)
    with pytest.raises(TimeoutError, match="readiness not reached"):
        await service.create(t, "run", Exec(argv=["train"]))
    assert boxes["sb-0"].ended
    box = await service.owned(t, "box", "sb-0")
    assert box["request_id"] == "run"
    assert box["ended_at"] >= box["requested_at"] and "ready_at" not in box
    events = [json.loads(line) for line in (service.root / t["id"] / "events.jsonl").read_text().splitlines()]
    assert any(event["kind"] == "provisioning_failed" and event["request_id"] == "run" for event in events)
    assert not any(event["kind"] in {"run", "box_ready", "box_readiness_unobserved"} for event in events)
    assert not any(key.startswith("command:") for key in service.records.data)
    assert (await service.inventory(await service.records.get.aio(t["key"])))[1] == 0
    assert (await service.records.get.aio(f"request:{t['id']}:run"))["state"] == "failed"


@pytest.mark.parametrize("limit", [1, 4])
async def test_overlapping_runs_count_provisioning_against_existing_box_limit(system, monkeypatch, limit):
    service, boxes = system
    t, _ = await trial(service, max_boxes=limit)
    original = code.modal.Sandbox.create.aio
    release = asyncio.Event()
    started = []

    async def delayed(*args, **kwargs):
        started.append(True)
        await release.wait()
        return await original(*args, **kwargs)

    monkeypatch.setattr(code.modal.Sandbox, "create", method(delayed))
    calls = [asyncio.create_task(service.create(t, str(i), Exec(argv=["sleep", "10"]))) for i in range(9)]
    while len(started) < limit:
        await asyncio.sleep(0)
    assert len(started) == limit
    assert (await service.records.get.aio(t["key"]))["creating"] == limit
    release.set()
    outcomes = await asyncio.gather(*calls, return_exceptions=True)
    assert sum(isinstance(value, dict) for value in outcomes) == limit
    assert all(isinstance(value, dict) or isinstance(value, HTTPException) for value in outcomes)
    assert len(boxes) == limit


async def test_sync_cpu_without_any_gpu_and_without_reloading_live_gpu_jobs(system):
    service, boxes = system
    t, _ = await trial(service, shared_workspace=True)
    t["agent_box"] = "agent"
    boxes["agent"] = Box("agent")
    await service.records.put.aio(t["key"], t)
    first = await service.sync_workspace(t)
    assert first["committed"] == first["reloaded"] == ["agent"]
    assert len(boxes) == 1
    job = await service.create(t, "run", Exec(argv=["sleep", "10"]))
    second = await service.sync_workspace(t)
    assert second["committed"] == second["reloaded"] == ["agent"]
    assert not hasattr(boxes[job["box"]], "refresh_count")


async def test_duplicate_requests_do_not_repeat_modal_operations(system):
    service, boxes = system
    t, _ = await trial(service)
    first = await service.create(t, "request")
    assert await service.create(t, "request") == first
    assert len(boxes) == 1
    request = Exec(argv=["python", "train.py"])
    a = await service.execute(t, first["id"], request, "exec-request")
    b = await service.execute(t, first["id"], request, "exec-request")
    assert a == b
    assert len(boxes[first["id"]].commands) == 2  # mkdir plus one user command
    with pytest.raises(HTTPException, match="different arguments"):
        await service.execute(t, first["id"], Exec(argv=["different"]), "exec-request")


async def test_concurrent_claim_has_one_winner(system):
    service, _ = system
    t, _ = await trial(service)
    results = await asyncio.gather(
        *(service.claim(t, "same", "create", {}) for _ in range(10)), return_exceptions=True
    )
    assert sum(isinstance(r, tuple) for r in results) == 1
    assert all(r.status_code == 409 for r in results if isinstance(r, HTTPException))


async def test_parallel_admission_enforces_box_limit(system):
    service, boxes = system
    t, _ = await trial(service, max_boxes=1)
    results = await asyncio.gather(*(service.create(t, str(i)) for i in range(5)), return_exceptions=True)
    assert sum(isinstance(r, dict) for r in results) == 1
    assert len(boxes) == 1


@pytest.mark.parametrize("failure", [RuntimeError("provider unavailable"), asyncio.CancelledError()])
async def test_failed_provisioning_releases_capacity_without_charging_gpu_time(system, monkeypatch, failure):
    service, boxes = system
    t, _ = await trial(service, gpu_seconds=60)
    create = code.modal.Sandbox.create.aio

    async def fail(*args, **kwargs):
        raise failure

    monkeypatch.setattr(code.modal.Sandbox.create, "aio", fail)
    with pytest.raises(type(failure)):
        await service.create(t, "failed")
    current = await service.records.get.aio(t["key"])
    assert current["creating"] == 0
    assert await service.inventory(current) == ([], 0)
    request = await service.records.get.aio(f"request:{t['id']}:failed")
    assert request["state"] == "failed"
    assert type(failure).__name__ in request["error"]
    monkeypatch.setattr(code.modal.Sandbox.create, "aio", create)
    await service.create(t, "next")
    assert len(boxes) == 1


async def test_readiness_failure_does_not_consume_capacity_or_gpu_time(system, monkeypatch):
    service, boxes = system
    t, _ = await trial(service, gpu_seconds=60)
    create = code.modal.Sandbox.create.aio

    async def not_ready(*args, **kwargs):
        box = await create(*args, **kwargs)

        async def failed_wait():
            return 1

        process = Process()
        process.wait = method(failed_wait)
        box.next_process = process
        return box

    monkeypatch.setattr(code.modal.Sandbox.create, "aio", not_ready)
    with pytest.raises(RuntimeError, match="prepare workspace"):
        await service.create(t, "failed")
    current = await service.records.get.aio(t["key"])
    assert (await service.inventory(current))[1] == 0
    assert boxes["sb-0"].ended
    request = await service.records.get.aio(f"request:{t['id']}:failed")
    assert request["state"] == "failed"
    monkeypatch.setattr(code.modal.Sandbox.create, "aio", create)
    await service.create(t, "next")


async def test_commands_have_no_implicit_deadline(system):
    service, boxes = system
    t, _ = await trial(service)
    box = await service.create(t, "box")
    await service.execute(t, box["id"], Exec(argv=["train"]), "train")
    assert boxes[box["id"]].commands[-1][1]["timeout"] is None
    await service.execute(t, box["id"], Exec(argv=["eval"], timeout=900), "eval")
    assert boxes[box["id"]].commands[-1][1]["timeout"] == 900


async def test_exec_preserves_binary_output(system):
    service, boxes = system
    t, _ = await trial(service)
    box = await service.create(t, "box")
    boxes[box["id"]].next_process = Process(b"\x00\xff\xfe\n")
    command = await service.execute(t, box["id"], Exec(argv=["binary-output"]), "binary")
    assert boxes[box["id"]].commands[-1][1]["text"] is False
    await asyncio.gather(*service.readers)
    status = await service.records.get.aio(f"command:{command['id']}")
    assert status["state"] == "succeeded" and status["exit_code"] == 0
    assert "error" not in status
    assert (service.root / t["id"] / f"{command['id']}.stdout").read_bytes() == b"\x00\xff\xfe\n"


@pytest.mark.parametrize("sandbox", [False, True])
async def test_commands_reading_stdin_receive_eof(system, sandbox):
    service, boxes = system
    t, _ = await trial(service)
    box = await service.create(t, "box")
    process = boxes[box["id"]] if sandbox else Process(complete=False)
    reader = process.main_process if sandbox else process
    process.stdin.write_eof = reader.done.set
    command = {"id": "stdin-reader", "box": box["id"], "trial": t["id"], "state": "running"}
    service.processes[command["id"]] = process
    await asyncio.wait_for(service.collect(t, command, process, sandbox=sandbox), timeout=1)
    assert command["state"] == "succeeded" and command["exit_code"] == 0


@pytest.mark.parametrize("error", [code.modal.exception.NotFoundError, code.modal.exception.ConflictError])
async def test_closing_stdin_after_exit_preserves_output_and_exit_code(system, error):
    service, _ = system
    t, _ = await trial(service)
    process = Process()

    async def closed():
        raise error("Process already finished")

    async def wait():
        return 17

    process.stdin.drain = method(closed)
    process.wait = method(wait)
    command = {"id": "fast-exit", "trial": t["id"], "state": "running"}
    service.processes[command["id"]] = process
    (service.root / t["id"]).mkdir(parents=True)
    await service.collect(t, command, process)
    assert command["state"] == "failed" and command["exit_code"] == 17
    assert "error" not in command
    assert (service.root / t["id"] / "fast-exit.stdout").read_text() == "hello\n"


@pytest.mark.parametrize("sandbox", [False, True])
@pytest.mark.parametrize("exit_code", [0, 7])
async def test_capture_failure_waits_for_native_exit_and_preserves_exit_code(system, sandbox, exit_code):
    service, boxes = system
    t, token = await trial(service)
    box = await service.create(t, "box")
    failed, exiting = asyncio.Event(), asyncio.Event()

    async def bad_output():
        failed.set()
        raise UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte")
        yield  # Make this an async iterator, as native readers are.

    async def empty():
        if False:
            yield ""

    async def wait(**kwargs):
        await exiting.wait()
        return exit_code

    async def poll():
        return exit_code if exiting.is_set() else None

    process = SimpleNamespace(
        stdout=bad_output(), stderr=empty(), wait=method(wait), poll=method(poll), stdin=Process().stdin
    )
    command = {"id": "capture-error", "box": box["id"], "trial": t["id"], "state": "running"}
    service.processes[command["id"]] = process
    await service.records.put.aio(f"command:{command['id']}", command)
    collecting = asyncio.create_task(service.collect(t, command, process, sandbox=sandbox))
    try:
        await failed.wait()
        await asyncio.sleep(0)
        assert not collecting.done()
        assert command["id"] in service.processes
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=create_api(service)),
            base_url="http://test",
            headers={"Authorization": f"Bearer {token}"},
        ) as client:
            assert (await client.get(f"/commands/{command['id']}")).json()["state"] == "running"
        exiting.set()
        await collecting
        status = await service.records.get.aio(f"command:{command['id']}")
        assert status["state"] == ("succeeded" if exit_code == 0 else "failed")
        assert status["exit_code"] == exit_code
        assert status["error"].startswith("Output capture failed (stdout: UnicodeDecodeError:")
        assert command["id"] not in service.processes
    finally:
        exiting.set()
        await collecting


async def test_native_wait_failure_remains_a_command_failure(system):
    service, _ = system
    t, _ = await trial(service)
    process = Process()

    async def unavailable():
        raise code.modal.exception.Error("Worker unavailable during wait")

    process.wait = method(unavailable)
    command = {"id": "wait-error", "trial": t["id"], "state": "running"}
    service.processes[command["id"]] = process
    (service.root / t["id"]).mkdir(parents=True)
    await service.collect(t, command, process)
    assert command["state"] == "failed" and "exit_code" not in command
    assert command["error"] == "Error: Worker unavailable during wait"


def test_resource_validation_leaves_provider_bounds_to_modal():
    limits = Limits(image_id="im-test", gpus_per_box=16, memory=128, box_seconds=172800)
    assert limits.gpus_per_box == 16
    assert limits.memory == 128
    assert limits.box_seconds == 172800


async def test_budget_starts_after_readiness_and_does_not_reserve_lifetime(system, monkeypatch):
    service, boxes = system
    now = [1000]
    monkeypatch.setattr(code.time, "time", lambda: now[0])
    original = code.modal.Sandbox.create.aio

    async def create(*args, **kwargs):
        box = await original(*args, **kwargs)

        execute = box.exec.aio

        async def first_command(*argv, **kwargs):
            now[0] += 120  # Waiting for startup exceeds the experimental GPU budget.
            return await execute(*argv, **kwargs)

        box.exec = method(first_command)
        return box

    monkeypatch.setattr(code.modal.Sandbox.create, "aio", create)
    t, _ = await trial(service, gpu_seconds=60, max_boxes=2, box_seconds=7200)
    first = await service.create(t, "one")
    assert first["ready_at"] == 1120
    current = await service.records.get.aio(t["key"])
    assert (await service.inventory(current))[1] == 0
    # A long requested lifetime did not consume the budget up front.
    monkeypatch.setattr(code.modal.Sandbox.create, "aio", original)
    await service.create(t, "two")
    now[0] += 10
    current = await service.records.get.aio(t["key"])
    assert (await service.inventory(current))[1] == 20
    assert len(boxes) == 2


async def test_hard_budget_stops_all_boxes(system, monkeypatch):
    service, boxes = system
    now = [1000]
    monkeypatch.setattr(code.time, "time", lambda: now[0])
    t, _ = await trial(service, gpu_seconds=10, max_boxes=2)
    await service.create(t, "one")
    await service.create(t, "two")
    now[0] += 6
    guard = asyncio.create_task(service.enforce_budgets())
    for _ in range(50):
        if all(b.ended for b in boxes.values()):
            break
        await asyncio.sleep(0.001)
    guard.cancel()
    await asyncio.gather(guard, return_exceptions=True)
    assert all(b.ended for b in boxes.values())
    with pytest.raises(HTTPException, match="ended"):
        await service.create(t, "after-budget")


@pytest.mark.parametrize("idle_trials", [1, 63])
async def test_slow_other_trial_probe_does_not_delay_expired_budget(system, monkeypatch, idle_trials):
    service, boxes = system
    now = [1000]
    monkeypatch.setattr(code.time, "time", lambda: now[0])
    idle_boxes = []
    for _ in range(idle_trials):
        idle_trial, _ = await trial(service, idle_seconds=300, idle_probe="cuda")
        idle_boxes.append(await service.create(idle_trial, "idle"))
    budget_trial, _ = await trial(service, gpu_seconds=10)
    budget_box = await service.create(budget_trial, "budget")
    started, release = asyncio.Event(), asyncio.Event()

    async def slow_read():
        started.set()
        await release.wait()
        return "123\n"

    process = probe()
    process.stdout.read = method(slow_read)
    for idle_box in idle_boxes:
        boxes[idle_box["id"]].next_process = process
    now[0] += 11
    guard = asyncio.create_task(service.enforce_budgets())
    try:
        await asyncio.wait_for(started.wait(), 1)
        await asyncio.wait_for(boxes[budget_box["id"]].main_process.done.wait(), 0.2)
        assert all(not boxes[box["id"]].ended for box in idle_boxes)
    finally:
        release.set()
        guard.cancel()
        await asyncio.gather(guard, return_exceptions=True)


async def test_policy_monitor_cancellation_leaves_no_probe_task(system):
    service, boxes = system
    idle_trial, _ = await trial(service, idle_seconds=300, idle_probe="cuda")
    idle_box = await service.create(idle_trial, "idle")
    started, finished = asyncio.Event(), asyncio.Event()

    async def blocked_read():
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            finished.set()

    process = probe()
    process.stdout.read = method(blocked_read)
    boxes[idle_box["id"]].next_process = process
    guard = asyncio.create_task(service.enforce_budgets())
    try:
        await asyncio.wait_for(started.wait(), 1)
    finally:
        guard.cancel()
        await asyncio.gather(guard, return_exceptions=True)
    assert finished.is_set()


async def test_budget_crossing_after_first_check_is_enforced_while_other_probe_stays_blocked(
    system, monkeypatch
):
    service, boxes = system
    now = [1000]
    monkeypatch.setattr(code.time, "time", lambda: now[0])
    idle_trial, _ = await trial(service, idle_seconds=300, idle_probe="cuda")
    idle_box = await service.create(idle_trial, "idle")
    budget_trial, _ = await trial(service, gpu_seconds=10)
    budget_box = await service.create(budget_trial, "budget")
    started, release, first_budget_check = asyncio.Event(), asyncio.Event(), asyncio.Event()
    inventory = service.inventory

    async def observed_inventory(t):
        result = await inventory(t)
        if t["id"] == budget_trial["id"]:
            first_budget_check.set()
        return result

    async def blocked_read():
        started.set()
        await release.wait()
        return "123\n"

    monkeypatch.setattr(service, "inventory", observed_inventory)
    process = probe()
    process.stdout.read = method(blocked_read)
    boxes[idle_box["id"]].next_process = process
    guard = asyncio.create_task(service.enforce_budgets())
    try:
        await asyncio.wait_for(started.wait(), 1)
        await asyncio.wait_for(first_budget_check.wait(), 1)
        assert not boxes[budget_box["id"]].ended
        now[0] += 11  # The budget expires after its first check, while the other probe still awaits output.
        await asyncio.wait_for(boxes[budget_box["id"]].main_process.done.wait(), 1.5)
        assert not release.is_set()
        assert not boxes[idle_box["id"]].ended
    finally:
        release.set()
        guard.cancel()
        await asyncio.gather(guard, return_exceptions=True)


async def test_trial_isolation_and_auth(system):
    service, _ = system
    t, owner_token = await trial(service)
    _, other_token = await trial(service)
    box = await service.create(t, "box")
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=create_api(service)), base_url="http://test"
    ) as c:
        assert (await c.get("/policy")).status_code == 401
        response = await c.delete(f"/boxes/{box['id']}", headers={"Authorization": f"Bearer {other_token}"})
        assert response.status_code == 404
        assert response.json()["detail"] == "Unknown box ID for this session"
        response = await c.get(f"/commands/{box['id']}", headers={"Authorization": f"Bearer {owner_token}"})
        assert response.status_code == 404
        assert response.json()["detail"] == "Unknown command ID for this session"


async def test_concurrent_commands_use_native_exec_and_capture_logs(system):
    service, boxes = system
    t, _ = await trial(service)
    box = await service.create(t, "box")
    process = Process("training progress\n", complete=False)
    boxes[box["id"]].next_process = process
    training = await service.execute(t, box["id"], Exec(argv=["python", "train.py"]), "train")
    boxes[box["id"]].next_process = Process("inspection\n")
    inspection = await service.execute(t, box["id"], Exec(argv=["nvidia-smi"]), "inspect")
    for _ in range(50):
        status = await service.records.get.aio(f"command:{inspection['id']}")
        if status["state"] == "succeeded":
            break
        await asyncio.sleep(0.001)
    assert status["state"] == "succeeded"
    assert (await service.records.get.aio(f"command:{training['id']}"))["state"] == "running"
    process.done.set()
    await asyncio.gather(*service.readers)
    assert (service.root / t["id"] / f"{training['id']}.stdout").read_text() == "training progress\n"


async def test_preempted_box_is_not_replaced(system):
    service, boxes = system
    t, _ = await trial(service)
    box = await service.create(t, "box")
    boxes[box["id"]].ended = True
    with pytest.raises(RuntimeError, match="terminated"):
        await service.execute(t, box["id"], Exec(argv=["echo", "hello"]), "after-preemption")
    assert len(boxes) == 1


async def test_native_errors_are_visible_through_http(system):
    service, boxes = system
    t, token = await trial(service)
    box = await service.create(t, "box")

    async def gone(*args, **kwargs):
        raise code.modal.exception.Error("GPU box went away")

    boxes[box["id"]].exec = method(gone)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=create_api(service)),
        base_url="http://test",
        headers={"Authorization": f"Bearer {token}"},
    ) as client:
        response = await client.post(
            f"/boxes/{box['id']}/exec",
            json={"argv": ["python", "train.py"]},
            headers={"Idempotency-Key": "gone"},
        )
        assert response.status_code == 502
        assert response.json()["detail"] == "GPU box went away"
        request = await client.get("/requests/gone")
        assert request.json()["state"] == "pending"


async def test_service_restart_reports_unfinished_command_as_unknown(system):
    service, boxes = system
    t, token = await trial(service)
    box = await service.create(t, "box")
    process = Process("progress", complete=False)
    boxes[box["id"]].next_process = process
    command = await service.execute(t, box["id"], Exec(argv=["train"]), "command")
    restarted = Service(
        service.records, service.app, service.volume, str(service.root), policy_index=service.policy_index
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=create_api(restarted)),
        base_url="http://test",
        headers={"Authorization": f"Bearer {token}"},
    ) as client:
        result = (await client.get(f"/commands/{command['id']}")).json()
        assert result["state"] == "unknown"
        assert "Service restarted" in result["error"]
    assert len(boxes) == 1
    process.done.set()
    await asyncio.gather(*service.readers)


async def test_completion_during_status_record_read_is_not_a_service_restart(system, monkeypatch):
    service, boxes = system
    t, token = await trial(service)
    box = await service.create(t, "box")
    process = Process("bg\n", complete=False)
    boxes[box["id"]].next_process = process
    command = await service.execute(t, box["id"], Exec(argv=["background-launch"]), "command")
    key = f"command:{command['id']}"
    get = service.records.get.aio
    read, reply = asyncio.Event(), asyncio.Event()
    first = True

    async def delayed_get(record_key):
        nonlocal first
        value = await get(record_key)
        if record_key == key and first:
            first = False
            read.set()
            await reply.wait()
        return value

    monkeypatch.setattr(service.records.get, "aio", delayed_get)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=create_api(service)),
        base_url="http://test",
        headers={"Authorization": f"Bearer {token}"},
    ) as client:
        pending = asyncio.create_task(client.get(f"/commands/{command['id']}"))
        await read.wait()  # The RPC has read "running", but its reply is still in flight.
        process.done.set()
        await asyncio.gather(*service.readers)
        assert (await get(key))["state"] == "succeeded"
        assert command["id"] not in service.processes
        reply.set()
        observed = (await pending).json()
        assert observed["state"] in {"running", "succeeded"}
        assert "error" not in observed
        terminal = (await client.get(f"/commands/{command['id']}")).json()
        assert terminal["state"] == "succeeded" and terminal["exit_code"] == 0


async def test_launcher_close_blocks_new_work_and_a_box_still_provisioning(system, monkeypatch):
    service, boxes = system
    t, _ = await trial(service, max_boxes=2)
    box = await service.create(t, "existing")
    original = code.modal.Sandbox.create.aio
    creating, release = asyncio.Event(), asyncio.Event()

    async def slow_create(*args, **kwargs):
        creating.set()
        await release.wait()
        return await original(*args, **kwargs)

    monkeypatch.setattr(code.modal.Sandbox.create, "aio", slow_create)
    pending = asyncio.create_task(service.create(t, "pending"))
    await creating.wait()
    await service.records.put.aio(f"closed:{t['id']}", True)
    release.set()
    with pytest.raises(HTTPException, match="ended while"):
        await pending
    assert boxes["sb-1"].ended
    with pytest.raises(HTTPException, match="ended"):
        await service.create(t, "new")
    with pytest.raises(HTTPException, match="ended"):
        await service.execute(t, box["id"], Exec(argv=["echo", "no"]), "new-command")


async def test_unavailable_time_is_excluded_from_the_experimental_budget(system, monkeypatch):
    service, boxes = system
    now = [1000]
    monkeypatch.setattr(code.time, "time", lambda: now[0])
    t, _ = await trial(service, gpu_seconds=60)
    box = await service.create(t, "box")
    current = await service.records.get.aio(t["key"])
    now[0] = 1010
    assert (await service.inventory(current))[1] == 10
    status = boxes[box["id"]].poll.aio

    async def unavailable():
        raise code.modal.exception.Error("Worker unreachable")

    boxes[box["id"]].poll = method(unavailable)
    now[0] = 1030
    assert (await service.inventory(current))[1] == 10
    boxes[box["id"]].poll = method(status)
    now[0] = 1100
    assert (await service.inventory(current))[1] == 10
    now[0] = 1105
    assert (await service.inventory(current))[1] == 15


async def test_explicit_stop_counts_accessible_time_without_waiting_for_another_poll(system, monkeypatch):
    service, _ = system
    now = [1000]
    monkeypatch.setattr(code.time, "time", lambda: now[0])
    t, _ = await trial(service)
    box = await service.create(t, "box")
    now[0] = 1123
    await service.stop(t, box["id"])
    current = await service.records.get.aio(t["key"])
    assert (await service.inventory(current))[1] == 123


async def test_command_idle_release_keeps_running_work_then_counts_a_full_idle_period(system, monkeypatch):
    service, boxes = system
    now = [1000]
    monkeypatch.setattr(code.time, "time", lambda: now[0])
    t, _ = await trial(service, idle_seconds=300)
    box = await service.create(t, "box")
    limits = Limits(**t["limits"])
    await service.check_idle(t, box, limits, False)
    now[0] = 1400
    await service.check_idle(t, box, limits, True)
    assert not boxes[box["id"]].ended
    now[0] = 1401
    await service.check_idle(t, box, limits, False)
    now[0] = 1700
    await service.check_idle(t, box, limits, False)
    assert not boxes[box["id"]].ended
    now[0] = 1701
    await service.check_idle(t, box, limits, False)
    assert boxes[box["id"]].ended


def probe(output="", code=0, error=""):
    async def read():
        return output

    async def stderr():
        return error

    async def wait():
        return code

    return SimpleNamespace(
        stdout=SimpleNamespace(read=method(read)),
        stderr=SimpleNamespace(read=method(stderr)),
        wait=method(wait),
    )


async def test_cuda_presence_resets_an_existing_absence_timer(system, monkeypatch):
    service, boxes = system
    now = [1000]
    monkeypatch.setattr(code.time, "time", lambda: now[0])
    t, _ = await trial(service, idle_seconds=300, idle_probe="cuda")
    box = await service.create(t, "box")
    native = boxes[box["id"]]
    limits = Limits(**t["limits"])
    native.next_process = probe()
    await service.check_idle(t, box, limits, True)
    assert (await service.records.get.aio(f"box:{box['id']}"))["idle_since"] == 1000
    now[0] = 1300
    native.next_process = probe("123\n")
    await service.check_idle(t, box, limits, True)
    assert not native.ended
    assert "idle_since" not in await service.records.get.aio(f"box:{box['id']}")
    native.next_process = probe()
    now[0] = 1400
    await service.check_idle(t, box, limits, True)
    now[0] = 1690
    await service.check_idle(t, box, limits, True)
    assert not native.ended
    now[0] = 1700
    await service.check_idle(t, box, limits, True)
    assert native.ended


async def test_probe_failure_is_not_treated_as_an_idle_gpu(system, monkeypatch):
    service, boxes = system
    now = [1000]
    monkeypatch.setattr(code.time, "time", lambda: now[0])
    t, _ = await trial(service, idle_seconds=300, idle_probe="cuda")
    box = await service.create(t, "box")
    native = boxes[box["id"]]
    limits = Limits(**t["limits"])
    native.next_process = probe()
    await service.check_idle(t, box, limits, False)
    now[0] = 1400
    native.next_process = probe(code=1, error="Driver unavailable")
    await service.check_idle(t, box, limits, False)
    assert not native.ended
    native.next_process = probe()
    now[0] = 1410
    await service.check_idle(t, box, limits, False)
    assert not native.ended
    now[0] = 1710
    await service.check_idle(t, box, limits, False)
    assert native.ended


async def test_shared_workspace_exec_does_not_implicitly_commit_or_reload(system):
    service, boxes = system
    t, _ = await trial(service, shared_workspace=True)
    t["agent_box"] = "agent"
    await service.records.put.aio(t["key"], t)
    boxes["agent"] = Box("agent")
    box = await service.create(t, "box")
    command = await service.execute(t, box["id"], Exec(argv=["python", "train.py"]), "train")
    await asyncio.gather(*service.readers)
    assert (await service.records.get.aio(f"command:{command['id']}"))["state"] == "succeeded"
    assert boxes["agent"].commands == []
    assert not hasattr(boxes[box["id"]], "refresh_count")
    assert not hasattr(boxes["agent"], "refresh_count")


async def test_explicit_sync_commits_both_views_before_reloading_either(system, monkeypatch):
    service, boxes = system
    t, _ = await trial(service, shared_workspace=True)
    t["agent_box"] = "agent"
    boxes["agent"] = Box("agent")
    box = await service.create(t, "box")
    steps = []

    for side, native in (("agent", boxes["agent"]), ("gpu", boxes[box["id"]])):

        async def commit(*argv, side=side):
            assert argv == ("sync", "/workspace")
            steps.append(("commit", side))
            return Process()

        async def reload(side=side):
            steps.append(("reload", side))

        monkeypatch.setattr(native, "exec", method(commit))
        monkeypatch.setattr(native, "reload_volumes", method(reload))

    result = await service.sync_workspace(t, box["id"])
    assert steps == [("commit", "agent"), ("commit", "gpu"), ("reload", "agent"), ("reload", "gpu")]
    assert result == {"box": box["id"], "committed": ["agent", "gpu"], "reloaded": ["agent", "gpu"]}


async def test_sync_busy_error_reports_partial_progress_without_stopping_work(system, monkeypatch):
    service, boxes = system
    t, token = await trial(service, shared_workspace=True)
    t["agent_box"] = "agent"
    await service.records.put.aio(t["key"], t)
    boxes["agent"] = Box("agent")
    box = await service.create(t, "box")

    async def busy():
        raise code.modal.exception.ConflictError("there are open files preventing the operation")

    monkeypatch.setattr(boxes[box["id"]], "reload_volumes", method(busy))
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=create_api(service)), base_url="http://test"
    ) as client:
        response = await client.post(f"/boxes/{box['id']}/sync", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 409
    assert response.json()["detail"] == {
        "box": box["id"],
        "committed": ["agent", "gpu"],
        "reloaded": ["agent"],
        "failed_side": "gpu",
        "failed_phase": "reload",
        "error": "there are open files preventing the operation",
    }
    assert not boxes["agent"].ended and not boxes[box["id"]].ended


async def test_sync_rejects_other_trials_boxes_and_nonshared_sessions(system):
    service, _ = system
    t, _ = await trial(service)
    box = await service.create(t, "box")
    other, _ = await trial(service, shared_workspace=True)
    with pytest.raises(HTTPException) as unknown:
        await service.sync_workspace(other, box["id"])
    assert unknown.value.status_code == 404
    with pytest.raises(HTTPException) as not_shared:
        await service.sync_workspace(t, box["id"])
    assert not_shared.value.status_code == 409


async def test_cpu_only_execs_do_not_extend_cuda_absence_timer(system, monkeypatch):
    service, boxes = system
    now = [1000]
    monkeypatch.setattr(code.time, "time", lambda: now[0])
    t, _ = await trial(service, idle_seconds=300, idle_probe="cuda")
    box = await service.create(t, "box")
    native = boxes[box["id"]]
    limits = Limits(**t["limits"])
    original = native.exec.aio

    async def execute(*argv, **kwargs):
        return probe() if argv[0] == "nvidia-smi" else await original(*argv, **kwargs)

    native.exec = method(execute)
    await service.check_idle(t, box, limits, True)
    for index, instant in enumerate([1100, 1290]):
        now[0] = instant
        await service.execute(t, box["id"], Exec(argv=["python", "cpu_only.py"]), f"cpu-{index}")
        await asyncio.gather(*service.readers)
        await service.check_idle(t, box, limits, True)
        assert not native.ended
        assert (await service.records.get.aio(f"box:{box['id']}"))["idle_since"] == 1000
    now[0] = 1300
    await service.check_idle(t, box, limits, True)
    assert native.ended


@pytest.mark.parametrize("activity", ["start", "completion"])
async def test_cpu_exec_during_cuda_probe_does_not_reset_absence(system, monkeypatch, activity):
    service, boxes = system
    now = [1000]
    monkeypatch.setattr(code.time, "time", lambda: now[0])
    t, _ = await trial(service, idle_seconds=300, idle_probe="cuda")
    box = await service.create(t, "box")
    native = boxes[box["id"]]
    limits = Limits(**t["limits"])
    work = Process(complete=False)
    entered, release = asyncio.Event(), asyncio.Event()
    blocking = False

    async def read_probe():
        if blocking:
            entered.set()
            await release.wait()
        return ""

    async def execute(*argv, **kwargs):
        if argv[0] != "nvidia-smi":
            return work
        result = probe()
        result.stdout.read = method(read_probe)
        return result

    native.exec = method(execute)
    command = None
    if activity == "completion":
        command = await service.execute(t, box["id"], Exec(argv=["work"]), "work")
    await service.check_idle(t, box, limits, True)
    now[0] = 1301
    blocking = True
    checking = asyncio.create_task(service.check_idle(t, box, limits, True))
    try:
        await asyncio.wait_for(entered.wait(), 1)
        if activity == "start":
            await asyncio.wait_for(service.execute(t, box["id"], Exec(argv=["work"]), "work"), 1)
        else:
            work.done.set()

            async def collected():
                while (await service.records.get.aio(f"command:{command['id']}"))["state"] == "running":
                    await asyncio.sleep(0)

            await asyncio.wait_for(collected(), 1)
        release.set()
        await asyncio.wait_for(checking, 1)
        assert native.ended
        assert (await service.records.get.aio(f"box:{box['id']}"))["idle_since"] == 1000
    finally:
        release.set()
        work.done.set()
        await asyncio.gather(checking, return_exceptions=True)
        await asyncio.gather(*service.readers)


async def test_cuda_idle_can_stop_long_cpu_work_without_waiting_for_output_drain(system, monkeypatch):
    service, boxes = system
    now = [1000]
    monkeypatch.setattr(code.time, "time", lambda: now[0])
    t, _ = await trial(service, idle_seconds=300, idle_probe="cuda")
    box = await service.create(t, "box")
    native = boxes[box["id"]]
    work = Process(complete=False)

    async def execute(*argv, **kwargs):
        return probe() if argv[0] == "nvidia-smi" else work

    native.exec = method(execute)
    await service.execute(t, box["id"], Exec(argv=["long-cpu-work"]), "cpu")
    await service.check_idle(t, box, Limits(**t["limits"]), True)
    now[0] = 1300
    await asyncio.wait_for(service.check_idle(t, box, Limits(**t["limits"]), True), 1)
    assert native.ended
    assert not work.done.is_set()
    assert not service.lock.locked()
    work.done.set()
    await asyncio.gather(*service.readers)


async def test_budget_and_cuda_idle_do_not_enumerate_command_or_request_history(system, monkeypatch):
    service, boxes = system
    now = [1000]
    monkeypatch.setattr(code.time, "time", lambda: now[0])
    for index in range(3300):
        service.records.data[f"request:history:{index}"] = {"state": "done"}
    budget_trial, _ = await trial(service, gpu_seconds=10)
    budget_box = await service.create(budget_trial, "budget")
    idle_trial, _ = await trial(service, idle_seconds=300, idle_probe="cuda")
    idle_box = await service.create(idle_trial, "idle")
    probed = asyncio.Event()

    async def read():
        probed.set()
        return "123\n"

    process = probe()
    process.stdout.read = method(read)
    boxes[idle_box["id"]].next_process = process

    async def forbidden_history_scan():
        raise AssertionError("Policy pass enumerated authoritative history")
        yield

    monkeypatch.setattr(service.records.items, "aio", forbidden_history_scan)
    now[0] += 11
    guard = asyncio.create_task(service.enforce_budgets())
    try:
        await asyncio.wait_for(boxes[budget_box["id"]].main_process.done.wait(), 1)
        await asyncio.wait_for(probed.wait(), 1)
        assert not boxes[idle_box["id"]].ended
    finally:
        guard.cancel()
        outcome = (await asyncio.gather(guard, return_exceptions=True))[0]
        if isinstance(outcome, BaseException) and not isinstance(outcome, asyncio.CancelledError):
            raise outcome


async def test_active_trial_reads_are_parallel_and_survive_monitor_recreation(system, monkeypatch):
    service, _ = system
    expected = {(await trial(service, gpu_seconds=10))[0]["key"] for _ in range(66)}
    started, all_started = set(), asyncio.Event()
    read = service.records.get.aio

    async def concurrent_read(key):
        if key in expected:
            started.add(key)
            if started == expected:
                all_started.set()
            await all_started.wait()
        return await read(key)

    monkeypatch.setattr(service.records.get, "aio", concurrent_read)
    restarted = Service(
        service.records, service.app, service.volume, str(service.root), policy_index=service.policy_index
    )
    trials = await asyncio.wait_for(restarted.policy_trials(), 1)
    assert {t["key"] for t in trials} == expected


async def test_closed_and_missing_index_entries_pruned_before_policy_kind_filter(system):
    service, _ = system
    closed, _ = await trial(service, idle_seconds=300, idle_probe="cuda")
    await service.records.put.aio(f"closed:{closed['id']}", True)
    # Closed entries are removed even if their limits cannot be inspected, or
    # the budget loop encounters an idle-only record before the idle loop does.
    closed["limits"] = {}
    await service.records.put.aio(closed["key"], closed)
    await service.policy_index.put.aio("trial:missing", True)
    assert await asyncio.gather(service.policy_trials(), service.policy_trials()) == [[], []]
    assert service.policy_index.data == {}
    assert await service.records.get.aio(closed["key"]) == closed


async def test_legacy_command_idle_alone_scans_running_commands(system, monkeypatch):
    service, _ = system
    await trial(service, idle_seconds=300, idle_probe="commands")
    await service.records.put.aio("command:legacy", {"box": "box-legacy", "state": "running"})
    checked = asyncio.Event()
    observed = []

    async def check(t, running, *, idle_checks):
        observed.append((idle_checks, running))
        checked.set()

    monkeypatch.setattr(service, "check_policy", check)
    guard = asyncio.create_task(service.policy_loop(idle_checks=True))
    try:
        await asyncio.wait_for(checked.wait(), 1)
        assert observed == [(True, {"box-legacy"})]
    finally:
        guard.cancel()
        await asyncio.gather(guard, return_exceptions=True)


def test_run_only_requires_shared_workspace_and_defaults_off():
    assert not Limits(image_id="im-test").run_only
    with pytest.raises(ValueError, match="run_only requires shared_workspace"):
        Limits(image_id="im-test", run_only=True)
    assert Limits(image_id="im-test", run_only=True, shared_workspace=True).run_only


@pytest.mark.parametrize("verb,path,body", [
    ("POST", "/boxes", None),
    ("POST", "/boxes/sb-any/exec", {"argv": ["true"]}),
    ("GET", "/boxes/sb-any/file?path=/workspace/x", None),
    ("PUT", "/boxes/sb-any/file?path=/workspace/x", None),
    ("POST", "/boxes/sb-any/refresh", None),
    ("POST", "/boxes/sb-any/sync", None),
])
async def test_run_only_api_rejects_worker_access_before_provider_call(system, verb, path, body):
    service, boxes = system
    t, token = await trial(service, run_only=True, shared_workspace=True)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_api(service)),
                                 base_url="http://test") as client:
        response = await client.request(verb, path, json=body, headers={
            "Authorization": f"Bearer {token}", "Idempotency-Key": "denied", "Content-Length": "0"
        })
    assert response.status_code == 403
    assert not boxes
    assert not any(key.startswith("request:") for key in service.records.data)


async def test_run_only_commits_agent_before_native_create_without_reload(system, monkeypatch):
    service, boxes = system
    t, token = await trial(service, run_only=True, shared_workspace=True)
    cpu = boxes["agent"] = Box("agent")
    t["agent_box"] = "agent"
    await service.records.put.aio(t["key"], t)
    original_create = code.modal.Sandbox.create.aio

    async def checked_create(*args, **kwargs):
        assert cpu.commands == [(("sync", "/workspace"), {})]
        assert not getattr(cpu, "refresh_count", 0)
        return await original_create(*args, **kwargs)

    monkeypatch.setattr(code.modal.Sandbox, "create", method(checked_create))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_api(service)),
                                 base_url="http://test") as client:
        headers = {"Authorization": f"Bearer {token}", "Idempotency-Key": "run"}
        result = await client.post("/run", json={"argv": ["python", "train.py"]}, headers=headers)
        assert result.status_code == 200
        job = result.json()
        assert boxes[job["box"]].entrypoint == ("python", "train.py")
        assert boxes[job["box"]].options["volumes"].keys() == {"/workspace"}
        # Idempotent replay neither commits nor allocates again.
        assert (await client.post("/run", json={"argv": ["python", "train.py"]},
                                  headers=headers)).json() == job
        assert len(cpu.commands) == 1
        assert (await client.get("/policy", headers=headers)).status_code == 200
        assert (await client.get(f"/commands/{job['id']}", headers=headers)).status_code == 200
        assert (await client.get(f"/commands/{job['id']}/logs", headers=headers)).status_code == 200
        # CPU sync retains its original commit+reload behavior.
        assert (await client.post("/sync", headers=headers)).status_code == 200
        assert cpu.refresh_count == 1
        assert (await client.delete(f"/boxes/{job['box']}", headers=headers)).status_code == 200


@pytest.mark.parametrize("failure", ["unbound", "commit_error"])
async def test_run_only_failed_cpu_commit_allocates_no_gpu_and_releases_slot(system, failure):
    service, boxes = system
    t, _ = await trial(service, run_only=True, shared_workspace=True)
    if failure == "commit_error":
        cpu = boxes["agent"] = Box("agent")
        t["agent_box"] = "agent"
        await service.records.put.aio(t["key"], t)

        async def failed_wait():
            return 1

        async def read_error():
            return "CPU workspace commit failed"

        cpu.next_process = SimpleNamespace(wait=method(failed_wait),
                                           stderr=SimpleNamespace(read=method(read_error)))
    with pytest.raises(HTTPException) as exc:
        await service.create(t, "commit-fails", Exec(argv=["true"]))
    assert exc.value.status_code in {409, 502}
    assert set(boxes) <= {"agent"}
    stored = await service.records.get.aio(t["key"])
    assert stored["creating"] == 0
    assert stored["boxes"] == []
    record = await service.records.get.aio(f"request:{t['id']}:commit-fails")
    assert record["state"] == "failed"


async def test_worker_keeps_task_network_and_environment_secret(system, monkeypatch):
    service, boxes = system
    monkeypatch.setattr(code.modal.Secret, "from_name", lambda name: name)
    session, _ = await trial(
        service,
        outbound_domain_allowlist=["data.example.test"],
        outbound_cidr_allowlist=[],
        environment_secret_name="task-env",
    )
    job = await service.create(session, "request", Exec(argv=["true"]))
    options = boxes[job["box"]].options
    assert options["outbound_domain_allowlist"] == ["data.example.test"]
    assert options["outbound_cidr_allowlist"] == []
    assert options["secrets"] == ["task-env"]
