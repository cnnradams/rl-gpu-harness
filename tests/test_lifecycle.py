import json
from types import SimpleNamespace

import pytest
from modal.volume import FileEntryType
from test_service import Box, Dict, method

import gpu_harness.lifecycle as code
from gpu_harness.lifecycle import close_and_export
from gpu_harness.service import token_hash


@pytest.fixture
def launcher(monkeypatch):
    records, boxes, traces = Dict(), {}, {}

    async def lookup(box_id):
        return boxes[box_id]

    async def listdir(path, **kwargs):
        entries = [
            SimpleNamespace(path=p, type=FileEntryType.FILE) for p in traces if p.startswith(path + "/")
        ]
        if not entries:
            raise code.modal.exception.NotFoundError(path)
        return entries

    async def read_file(path):
        yield traces[path]

    monkeypatch.setattr(code.modal.Dict, "from_name", lambda *args: records)
    monkeypatch.setattr(code.modal.Sandbox, "from_id", method(lookup))
    monkeypatch.setattr(
        code.modal.Volume,
        "from_name",
        lambda *args: SimpleNamespace(listdir=method(listdir), read_file=method(read_file)),
    )
    return records, boxes, traces


async def prepare(launcher, session_id, box_ids):
    records, boxes, _ = launcher
    session = {"trial": session_id, "token": f"secret-{session_id}", "url": "https://gpu.modal.run"}
    await records.put.aio(f"trial:{token_hash(session['token'])}", {"id": session_id, "boxes": box_ids})
    for box_id in box_ids:
        box = Box(box_id)

        async def wait(**kwargs):
            return 0

        box.wait = method(wait)
        boxes[box_id] = box
        await records.put.aio(f"box:{box_id}", {"id": box_id, "trial": session_id, "ready_at": 1})
    return session


async def test_cleanup_is_scoped_terminal_and_exports_real_bytes(launcher, tmp_path):
    records, boxes, traces = launcher
    a = await prepare(launcher, "a", ["sb-a"])
    await prepare(launcher, "b", ["sb-b"])
    traces["a/events.jsonl"] = b'{"kind":"exec","command":"cmd-a","request_id":"req-a"}\n'
    traces["a/cmd-a.stdout"] = b"actual command output\n"
    traces["b/events.jsonl"] = b"other session\n"
    await records.put.aio("command:cmd-a", {"id": "cmd-a", "trial": "a", "state": "succeeded"})
    await records.put.aio("request:a:req-a", {"state": "done"})
    snapshot = await close_and_export(a, tmp_path)
    assert boxes["sb-a"].ended
    assert not boxes["sb-b"].ended
    assert await records.get.aio("closed:a") is True
    assert await records.get.aio("closed:b") is None
    assert snapshot["errors"] == []
    assert snapshot["gpu_time_accounting_complete"] is False  # No confirmed-readiness metadata.
    assert set(snapshot["records"]) == {"command:cmd-a", "request:a:req-a"}
    assert (tmp_path / "traces" / "cmd-a.stdout").read_bytes() == traces["a/cmd-a.stdout"]
    assert a["token"] not in (tmp_path / "session.json").read_text()
    assert "other session" not in str(list(tmp_path.rglob("*")))
    await close_and_export(a, tmp_path)


async def test_unused_session_can_close_without_trace_directory(launcher, tmp_path):
    session = await prepare(launcher, "a", [])
    snapshot = await close_and_export(session, tmp_path)
    assert snapshot["closed"] is True
    assert snapshot["errors"] == []
    assert snapshot["boxes"] == []
    assert snapshot["gpu_time_accounting_complete"] is True


async def test_one_stop_failure_does_not_skip_other_boxes_and_is_saved(launcher, tmp_path):
    _, boxes, _ = launcher
    session = await prepare(launcher, "a", ["sb-fail", "sb-ok"])
    launcher[2]["a/events.jsonl"] = b""

    async def fail():
        raise RuntimeError("provider unavailable")

    boxes["sb-fail"].terminate = method(fail)
    with pytest.raises(RuntimeError, match="provider unavailable"):
        await close_and_export(session, tmp_path)
    assert boxes["sb-ok"].ended
    snapshot = json.loads((tmp_path / "session.json").read_text())
    assert snapshot["closed"] is True
    assert len(snapshot["errors"]) == 1
