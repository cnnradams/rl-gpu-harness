"""Exercise the actual shell/data path, with local processes in place of Modal."""

import asyncio
import hashlib
import os
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient
from test_service import Dict, method

import gpu_harness.service as service_code
from gpu_harness.client import Client
from gpu_harness.service import Limits, Service, create_api, token_hash
from gpu_harness.transfer import upload


@pytest.fixture
def transport(tmp_path, monkeypatch):
    processes = []

    async def execute(*argv, **options):
        process = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        processes.append(process)

        class Reader:
            read = method(process.stdout.read)

            async def __aiter__(self):
                while chunk := await process.stdout.read(1024 * 1024):
                    yield chunk

        return SimpleNamespace(
            stdin=SimpleNamespace(
                write=process.stdin.write,
                write_eof=process.stdin.write_eof,
                drain=method(process.stdin.drain),
            ),
            stdout=Reader(),
            stderr=SimpleNamespace(read=method(process.stderr.read)),
            wait=method(process.wait),
        )

    async def lookup(box):
        return SimpleNamespace(exec=method(execute))

    async def commit():
        pass

    monkeypatch.setattr(service_code.modal, "Sandbox", SimpleNamespace(from_id=method(lookup)))
    records = Dict()
    token = "owner-token"
    key = "trial:" + token_hash(token)
    records.data[key] = {
        "id": "owner",
        "key": key,
        "boxes": ["box"],
        "limits": Limits(image_id="im-test").model_dump(),
    }
    records.data["box:box"] = {"id": "box", "trial": "owner"}
    other_key = "trial:" + token_hash("other-token")
    records.data[other_key] = {
        "id": "other", "key": other_key, "boxes": [],
        "limits": Limits(image_id="im-test").model_dump(),
    }
    service = Service(
        records, None, SimpleNamespace(commit=method(commit)), str(tmp_path / "traces"),
        policy_index=Dict(),
    )
    api = TestClient(create_api(service))

    client = Client("http://test", token)
    client.http.close()
    client.http = api
    api.headers["Authorization"] = "Bearer " + token
    yield client, api, processes
    client.http.close()
    assert all(process.returncode is not None for process in processes)


@pytest.mark.parametrize("size", [0, 13, 40 * 1024 * 1024 + 17])
def test_stream_round_trip_and_unrestricted_quoted_paths(transport, tmp_path, size):
    client, _, _ = transport
    source = tmp_path / "source"
    remote = tmp_path / "outside workspace ' $(false) 🦊"
    destination = tmp_path / "download"
    block = bytes(range(256)) * 4096
    digest = hashlib.sha256()
    with source.open("wb") as file:
        remaining = size
        while remaining:
            chunk = block[: min(remaining, len(block))]
            file.write(chunk)
            digest.update(chunk)
            remaining -= len(chunk)
    uploaded = client.transfer("box", source, str(remote), "push")
    downloaded = client.transfer("box", destination, str(remote), "pull")
    assert uploaded == {"bytes": size, "sha256": digest.hexdigest()}
    assert downloaded["sha256"] == digest.hexdigest()
    with destination.open("rb") as file:
        assert hashlib.file_digest(file, "sha256").hexdigest() == digest.hexdigest()
    assert not list(tmp_path.glob("*.gpu-*"))


def test_incomplete_push_preserves_existing_destination(transport, tmp_path):
    _, api, _ = transport
    target = tmp_path / "existing"
    target.write_bytes(b"keep me")
    response = api.put(
        "/boxes/box/file", params={"path": str(target)}, content=b"partial", headers={"Content-Length": "100"}
    )
    assert response.status_code == 400
    assert "Incomplete upload" in response.text
    assert target.read_bytes() == b"keep me"
    assert not list(tmp_path.glob("*.gpu-*"))


@pytest.mark.asyncio
async def test_small_http_fragments_are_coalesced_before_native_stdin_writes():
    writes = []

    async def complete():
        return 0

    process = SimpleNamespace(
        stdin=SimpleNamespace(
            write=lambda data: writes.append(bytes(data)),
            write_eof=lambda: None,
            drain=method(complete),
        ),
        wait=method(complete),
    )

    async def execute(*args, **kwargs):
        return process

    fragment = b"x" * 8192

    async def fragments():
        for _ in range(2049):
            yield fragment

    result = await upload(
        SimpleNamespace(exec=method(execute)), "/tmp/data", 2049 * len(fragment), fragments()
    )
    assert len(writes) < 10  # Avoid thousands of native stream writes for one HTTP body.
    assert b"".join(writes) == fragment * 2049
    assert result["bytes"] == 2049 * len(fragment)


@pytest.mark.asyncio
@pytest.mark.parametrize("error_type", [ConnectionError, ConnectionResetError])
async def test_disconnected_push_preserves_existing_destination(transport, tmp_path, error_type):
    _, _, processes = transport
    target = tmp_path / "existing"
    target.write_bytes(b"keep me")
    sb = await service_code.modal.Sandbox.from_id.aio("box")

    async def broken_stream():
        yield b"partial"
        raise error_type("Network disconnected")

    with pytest.raises(error_type, match="Network disconnected"):
        await upload(sb, str(target), 100, broken_stream())
    assert processes[0].returncode == 1
    assert target.read_bytes() == b"keep me"
    assert not list(tmp_path.glob("*.gpu-*"))


def writer_already_finished(monkeypatch, *, before_stdin):
    lookup = service_code.modal.Sandbox.from_id.aio

    async def box(box_id):
        sandbox = await lookup(box_id)

        async def execute(*argv, **options):
            process = await sandbox.exec.aio(*argv, **options)
            if before_stdin:
                await process.wait.aio()
            else:
                write_eof, drain = process.stdin.write_eof, process.stdin.drain.aio
                eof = False

                def finish():
                    nonlocal eof
                    eof = True
                    write_eof()

                async def completed():
                    await drain()
                    if eof:
                        await process.wait.aio()
                        raise service_code.modal.exception.ConflictError("Exec has already completed")

                process.stdin.write_eof = finish
                process.stdin.drain = method(completed)
            return process

        return SimpleNamespace(exec=method(execute))

    monkeypatch.setattr(service_code.modal.Sandbox.from_id, "aio", box)


@pytest.mark.parametrize("size", [13, 9 * 1024 * 1024])
def test_failed_writer_before_stdin_reports_filesystem_error(transport, tmp_path, monkeypatch, size):
    _, api, _ = transport
    writer_already_finished(monkeypatch, before_stdin=True)
    parent = tmp_path / "blocked-parent"
    parent.write_bytes(b"not a directory")
    target = parent / "file"
    response = api.put(
        "/boxes/box/file",
        params={"path": str(target)},
        content=b"x" * size,
        headers={"Content-Length": str(size)},
    )
    assert response.status_code == 400
    assert "directory" in response.text
    assert parent.read_bytes() == b"not a directory"
    assert not target.exists()


def test_push_creates_nested_quoted_destination_parents(transport, tmp_path):
    client, _, _ = transport
    source = tmp_path / "source"
    source.write_bytes(b"nested file")
    target = tmp_path / "new parent ' $(false) 🦊" / "nested" / "file"
    result = client.transfer("box", source, str(target), "push")
    assert result["bytes"] == source.stat().st_size
    assert target.read_bytes() == source.read_bytes()


@pytest.mark.parametrize("directory", [False, True])
def test_completed_writer_result_wins_over_closed_stdin(transport, tmp_path, monkeypatch, directory):
    client, api, _ = transport
    writer_already_finished(monkeypatch, before_stdin=False)
    source, target = tmp_path / "source", tmp_path / "target"
    source.write_bytes(b"complete file")
    if directory:
        target.mkdir()
        (target / "keep").write_bytes(b"unchanged")
        response = api.put(
            "/boxes/box/file",
            params={"path": str(target)},
            content=source.read_bytes(),
            headers={"Content-Length": str(source.stat().st_size)},
        )
        assert response.status_code == 400
        assert "directory" in response.text
        assert (target / "keep").read_bytes() == b"unchanged"
    else:
        result = client.transfer("box", source, str(target), "push")
        assert result["bytes"] == source.stat().st_size
        assert target.read_bytes() == source.read_bytes()
    assert not list(tmp_path.glob("*.gpu-*"))


async def test_cancelled_push_closes_writer_and_preserves_destination(transport, tmp_path):
    _, _, processes = transport
    target = tmp_path / "existing"
    target.write_bytes(b"keep me")
    sb = await service_code.modal.Sandbox.from_id.aio("box")
    waiting = asyncio.Event()

    async def cancelled_stream():
        yield b"partial"
        waiting.set()
        await asyncio.Event().wait()

    task = asyncio.create_task(upload(sb, str(target), 100, cancelled_stream()))
    await waiting.wait()
    task.cancel("caller cancelled")
    with pytest.raises(asyncio.CancelledError, match="caller cancelled"):
        await task
    assert processes[0].returncode == 1
    assert target.read_bytes() == b"keep me"
    assert not list(tmp_path.glob("*.gpu-*"))


def test_missing_remote_file_preserves_existing_local_file(transport, tmp_path):
    client, _, _ = transport
    target = tmp_path / "existing"
    target.write_bytes(b"keep me")
    with pytest.raises(httpx.HTTPStatusError, match="No such file"):
        client.transfer("box", target, str(tmp_path / "missing"), "pull")
    assert target.read_bytes() == b"keep me"
    assert set(tmp_path.iterdir()) == {target}


@pytest.mark.parametrize("kind", ["directory", "directory-symlink", "fifo"])
def test_pull_nonregular_source_returns_clear_error_before_streaming(transport, tmp_path, kind):
    client, api, processes = transport
    source = tmp_path / "source ' $(false) 🦊"
    if kind == "fifo":
        os.mkfifo(source)
    else:
        source.mkdir()
        if kind == "directory-symlink":
            link = tmp_path / "source-link"
            link.symlink_to(source)
            source = link
    response = api.get("/boxes/box/file", params={"path": str(source)})
    assert response.status_code == 400
    assert response.json() == {
        "detail": f"gpu pull requires a regular file; source is not a regular file: {source}"
    }
    assert len(processes) == 1  # Reject before starting cat or sending a file response.
    destination = tmp_path / "destination"
    destination.write_bytes(b"keep me")
    with pytest.raises(httpx.HTTPStatusError, match="requires a regular file"):
        client.transfer("box", destination, str(source), "pull")
    assert destination.read_bytes() == b"keep me"


def test_pull_follows_source_symlink_to_regular_file(transport, tmp_path):
    client, api, _ = transport
    source, link, destination = (tmp_path / name for name in ("source", "link", "destination"))
    source.write_bytes(b"\x00\xffregular file\n")
    link.symlink_to(source)
    response = api.get("/boxes/box/file", params={"path": str(link)})
    assert response.status_code == 200
    assert response.content == source.read_bytes()
    assert response.headers["Content-Length"] == str(source.stat().st_size)
    client.transfer("box", destination, str(link), "pull")
    assert destination.read_bytes() == source.read_bytes()


@pytest.mark.parametrize("token,detail", [("invalid", "Invalid trial"), ("other-token", "Unknown box")])
def test_transfers_enforce_session_ownership(transport, tmp_path, token, detail):
    client, _, processes = transport
    client.http.headers["Authorization"] = "Bearer " + token
    with pytest.raises(httpx.HTTPStatusError) as error:
        client.transfer("box", tmp_path / "download", "/tmp/remote", "pull")
    assert error.value.response.status_code in {401, 404}
    assert detail in error.value.response.text
    assert not processes


def test_push_preserves_mode_and_follows_destination_symlink(transport, tmp_path):
    client, _, _ = transport
    source, target, link = (tmp_path / name for name in ("source", "target", "link"))
    source.write_bytes(b"replacement")
    target.write_bytes(b"old")
    target.chmod(0o755)
    link.symlink_to(target)
    client.transfer("box", source, str(link), "push")
    assert link.is_symlink()
    assert target.read_bytes() == b"replacement"
    assert target.stat().st_mode & 0o777 == 0o755
