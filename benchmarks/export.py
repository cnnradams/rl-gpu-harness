"""Capture owned source packages and scan review artifacts without whole-file buffers."""

import shutil
from pathlib import Path

_SCAN_CHUNK_BYTES = 8 * 1024 * 1024


def copy_sources(destination):
    import benchmarks
    import gpu_harness

    for package in (benchmarks, gpu_harness):
        shutil.copytree(
            Path(package.__file__).parent,
            destination / package.__name__,
            ignore=shutil.ignore_patterns("__pycache__"),
        )


def contains_secret(path: Path, values: list[str]) -> bool:
    patterns = tuple(value.encode() for value in values if value)
    if not patterns:
        return False
    overlap = max(map(len, patterns)) - 1
    tail = b""
    with path.open("rb") as stream:
        while chunk := stream.read(_SCAN_CHUNK_BYTES):
            data = tail + chunk
            if any(pattern in data for pattern in patterns):
                return True
            tail = data[-overlap:] if overlap else b""
    return False
