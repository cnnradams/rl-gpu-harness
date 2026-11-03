"""Trusted launcher cleanup and collection, using operator Modal credentials."""

import json
import time
from pathlib import Path, PurePosixPath

import modal
from modal.volume import FileEntryType

from . import RECORDS_NAME, TRACE_VOLUME_NAME
from .service import token_hash


async def close_and_export(session, destination: Path):
    """End access, stop owned boxes, and save the currently committed traces.

    Closing is terminal. A create already provisioning is stopped by the service
    when it checks the close marker before admitting the ready box.
    """
    destination.mkdir(parents=True, exist_ok=True)
    records = modal.Dict.from_name(RECORDS_NAME)
    session_id = session["trial"]
    await records.put.aio(f"closed:{session_id}", True)
    owner = await records.get.aio(f"trial:{token_hash(session['token'])}")
    errors, boxes = [], []
    for box_id in owner["boxes"]:
        box = await records.get.aio(f"box:{box_id}")
        try:
            sb = await modal.Sandbox.from_id.aio(box_id)
            await sb.terminate.aio()
            await sb.wait.aio(raise_on_termination=False)
            box = await records.get.aio(f"box:{box_id}")
            box.setdefault("ended_at", time.time())
            await records.put.aio(f"box:{box_id}", box)
            boxes.append({**box, "observed_exit_code": await sb.poll.aio()})
        except Exception as error:
            errors.append(f"{box_id}: {type(error).__name__}: {error}")
            boxes.append(box)

    volume = modal.Volume.from_name(TRACE_VOLUME_NAME)
    try:
        try:
            entries = await volume.listdir.aio(session_id, recursive=True)
        except (FileNotFoundError, modal.exception.NotFoundError):
            # A session that never created a box has no trace directory yet.
            entries = []
            if owner["boxes"]:
                raise RuntimeError("No committed trace directory for a session that created boxes") from None
        for entry in entries:
            if entry.type != FileEntryType.FILE:
                continue
            relative = PurePosixPath(entry.path).relative_to(session_id)
            target = destination / "traces" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("wb") as output:
                async for chunk in volume.read_file.aio(entry.path):
                    output.write(chunk)
    except Exception as error:
        errors.append(f"trace export: {type(error).__name__}: {error}")

    # Fetch records referenced by this session's committed events, rather than
    # scanning the shared Dict's history for every completed Harbor trial.
    scoped_records = {}
    events = destination / "traces" / "events.jsonl"
    if events.exists():
        for line in events.read_text().splitlines():
            event = json.loads(line)
            for field, prefix in [("command", "command:"), ("request_id", f"request:{session_id}:")]:
                if field in event:
                    key = prefix + event[field]
                    if key not in scoped_records:
                        scoped_records[key] = await records.get.aio(key)

    snapshot = {
        "session": {key: value for key, value in session.items() if key != "token"},
        "closed": True,
        "snapshot_at": time.time(),
        "creating_at_close": owner.get("creating", 0),
        "boxes": boxes,
        "gpu_time_accounting_complete": all(box.get("readiness_observed", False) for box in boxes),
        "records": scoped_records,
        "errors": errors,
        "trace_note": "Committed files at collection time; in-flight commands may finish logging later.",
    }
    (destination / "session.json").write_text(json.dumps(snapshot, indent=2) + "\n")
    if errors:
        raise RuntimeError("GPU session cleanup/export failed: " + "; ".join(errors))
    return snapshot
