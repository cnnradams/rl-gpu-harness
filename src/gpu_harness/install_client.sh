#!/bin/sh
set -eu
runtime=""
for candidate in python3 python; do
    if command -v "$candidate" >/dev/null 2>&1 && \
        "$candidate" -c 'import sys, venv; assert sys.version_info >= (3, 10)' 2>/dev/null; then
        runtime="$candidate"
        break
    fi
done
if [ -z "$runtime" ]; then
    echo 'gpu-harness requires Python 3.10+ with the venv module in the task image.' >&2
    exit 1
fi
if ! "$runtime" -m pip --help 2>/dev/null | grep -q -- '--python'; then
    echo 'gpu-harness requires pip 22.3+ for the selected Python to install its isolated client.' >&2
    exit 1
fi
"$runtime" -m venv --without-pip /opt/gpu-harness/venv
"$runtime" -m pip --python /opt/gpu-harness/venv/bin/python install httpx==0.28.1
