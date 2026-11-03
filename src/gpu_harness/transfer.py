"""Ordinary HTTP bodies bridged to native Sandbox exec streams."""

import hashlib
from stat import S_ISREG

import modal
from fastapi import HTTPException

WRITE = """set -eu
target=$(readlink -m -- "$1")
mkdir -p -- "$(dirname -- "$target")"
tmp=$(mktemp -- "${target}.gpu-XXXXXX")
trap 'rm -f -- "$tmp"' EXIT
cat > "$tmp"
actual=$(wc -c < "$tmp")
if [ "$actual" != "$2" ]; then
    echo "Incomplete upload: expected $2 bytes, received $actual" >&2
    exit 1
fi
if [ -f "$target" ]; then chmod --reference="$target" "$tmp"; else chmod 644 "$tmp"; fi
mv -fT -- "$tmp" "$target"
"""


async def checked(process):
    if await process.wait.aio() != 0:
        raise HTTPException(400, (await process.stderr.read.aio()).decode(errors="replace").strip())


async def upload(sb, path, size, chunks):
    process = await sb.exec.aio("sh", "-c", WRITE, "gpu-push", path, str(size), text=False)
    count, buffer, digest = 0, bytearray(), hashlib.sha256()
    stdin_closed = False
    try:
        async for chunk in chunks:
            # Batch native stdin writes rather than one RPC per HTTP fragment.
            for start in range(0, len(chunk), 1024 * 1024):
                part = chunk[start : start + 1024 * 1024]
                count += len(part)
                digest.update(part)
                buffer.extend(part)
                if len(buffer) >= 8 * 1024 * 1024:
                    try:
                        process.stdin.write(buffer)
                        await process.stdin.drain.aio()
                    except (modal.exception.ConflictError, BrokenPipeError, ConnectionResetError):
                        stdin_closed = True
                        break
                    buffer.clear()
            if stdin_closed:
                break
    finally:
        if not stdin_closed:
            try:
                if buffer:
                    process.stdin.write(buffer)
                process.stdin.write_eof()
                await process.stdin.drain.aio()
            except (modal.exception.ConflictError, BrokenPipeError, ConnectionResetError):
                # A finished writer may reject stdin before its filesystem
                # error (or successful completion) is collected below.
                pass
        await process.wait.aio()
    await checked(process)
    return {"bytes": count, "sha256": digest.hexdigest()}


async def download(sb, path):
    stat = await sb.exec.aio("stat", "-Lc", "%f %s", "--", path, text=False)
    output = await stat.stdout.read.aio()
    await checked(stat)
    mode, size = output.split()
    if not S_ISREG(int(mode, 16)):
        raise HTTPException(400, f"gpu pull requires a regular file; source is not a regular file: {path}")
    process = await sb.exec.aio("cat", "--", path, text=False)

    async def chunks():
        async for chunk in process.stdout:
            yield chunk
        # Content-Length exposes a truncated read; the client keeps its old file.
        await checked(process)

    return int(size), chunks()
