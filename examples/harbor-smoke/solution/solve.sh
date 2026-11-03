#!/bin/sh
set -eu
gpu policy > /workspace/policy.json
if python3 -c 'import json; assert json.load(open("/workspace/policy.json"))["limits"]["shared_workspace"]'; then
    test "$(cat /workspace/seeded.txt)" = initial-task-file
    cp /solution/compute.py /workspace/compute.py
    echo first > /workspace/input.txt
    gpu sync
    gpu run --detach -- sh -c 'test "$(cat /workspace/input.txt)" = first; python /workspace/compute.py; cp /tmp/report.json /workspace/report.json; echo remote > /workspace/reply.txt; sleep 15' > /tmp/run.json
    BOX_ID=$(python3 -c 'import json; print(json.load(open("/tmp/run.json"))["box"])')
    COMMAND_ID=$(python3 -c 'import json; print(json.load(open("/tmp/run.json"))["id"])')
    gpu exec "$BOX_ID" -- nvidia-smi
    gpu wait "$COMMAND_ID"
    gpu sync
    test "$(cat /workspace/reply.txt)" = remote
    echo second > /workspace/input.txt
    gpu sync
    gpu run -- sh -c 'test "$(cat /workspace/input.txt)" = second; echo updated > /workspace/reply.txt'
    gpu sync
    test "$(cat /workspace/reply.txt)" = updated
else
    gpu run --detach -- sleep infinity > /tmp/run.json
    BOX_ID=$(python3 -c 'import json; print(json.load(open("/tmp/run.json"))["box"])')
    gpu push "$BOX_ID" /solution/compute.py /tmp/compute.py
    gpu exec "$BOX_ID" -- python /tmp/compute.py
    gpu pull "$BOX_ID" /tmp/report.json /workspace/report.json
fi
