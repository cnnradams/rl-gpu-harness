"""Adapt native PostTrainBench generation to the study's common workspace."""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

from harbor.models.task.config import TaskConfig

from benchmarks.prepare import check_revision

REVISION = "ea8d65bc375e61c8ac77a7aea86effcc995378fb"
TASK = "posttrainbench-gsm8k-qwen3-1.7b"
COLLECTOR = "src/harbor_adapter/template/environment/ptb_collect.sh"
WRAPPER = """#!/bin/bash
set -euo pipefail
if [ -n "${PTB_SHARED_WORKSPACE:-}" ]; then
    PTB_WORKSPACE="$PTB_SHARED_WORKSPACE" PTB_VOLUME_DIR=/tmp/ptb-model \\
        PTB_AGENT_LOGS_DST=/tmp/ptb-runtime-logs bash /usr/local/bin/ptb_collect.sh
    export PTB_MODEL_DIR=/tmp/ptb-model
    cp -a /logs/artifacts/workspace /logs/verifier/workspace
fi
exec bash /tests/ptb-native-test.sh "$@"
"""


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"Expected one occurrence of {old!r}")
    return text.replace(old, new)


def prepare(source: Path, output: Path, hours: int):
    check_revision(source, REVISION)
    task = output / TASK
    if task.exists():
        raise FileExistsError(task)
    subprocess.run(
        [
            sys.executable,
            str(source / "src/harbor_adapter/run_adapter.py"),
            "--benchmark",
            "gsm8k",
            "--model",
            "qwen3-1.7b",
            "--agent-name",
            "terminus-2",
            "--num-hours",
            str(hours),
            "--output",
            str(output.resolve()),
        ],
        check=True,
    )
    original_instruction = (task / "instruction.md").read_bytes()
    native_files = {p.relative_to(task): p.read_bytes() for p in (task / "tests").rglob("*") if p.is_file()}
    instruction = original_instruction.decode()
    for old, new in [
        (
            "- The machine is equipped with an Nvidia H100 GPU.",
            "- One NVIDIA H100 is available for GPU work.",
        ),
        (
            r"Store your best trained model in the folder \`final_model\`.",
            r"Store your best trained model in the agent's \`/workspace/final_model\`.",
        ),
        (r"calling \`bash timer.sh\` in the current directory.", r"calling \`bash /workspace/timer.sh\`."),
        (
            "Work only in the current directory and subdirectories.",
            "Work only in /workspace and its subdirectories.",
        ),
    ]:
        instruction = replace_once(instruction, old, new)
    instruction = instruction.replace("../test_data.json", "/home/agent/test_data.json")
    instruction = instruction.replace("../contamination_check.py", "/home/agent/contamination_check.py")
    (task / "instruction.md").write_text(instruction)
    for role in ("environment", "tests"):
        path = task / role / "Dockerfile"
        recipe = path.read_text()
        if role == "environment":
            start, end = recipe.index("# Node.js 22.x"), recipe.index("# uv\n")
            recipe = recipe[:start] + recipe[end:]
            start, end = recipe.index("# Agent CLIs,"), recipe.index("RUN mkdir -p /opt")
            recipe = recipe[:start] + recipe[end:]
            recipe = replace_once(
                recipe, "    build-essential \\\n", "    build-essential tmux asciinema ripgrep procps \\\n"
            )
        else:
            shutil.copyfile(source / COLLECTOR, task / "tests/ptb_collect.sh")
            recipe += "\nCOPY ptb_collect.sh /usr/local/bin/ptb_collect.sh\n"
        recipe += (
            "\nRUN rm -rf /workspace && mv /home/agent/workspace /workspace "
            "&& ln -s /workspace /home/agent/workspace\nWORKDIR /workspace\n"
            "RUN python3 -m pip freeze > /opt/task-pip-freeze.txt\n"
        )
        path.write_text(recipe)
    (task / "tests/test.sh").rename(task / "tests/ptb-native-test.sh")
    (task / "tests/test.sh").write_text(WRAPPER)
    config = TaskConfig.model_validate_toml((task / "task.toml").read_text())
    config.environment.gpus = 0
    config.environment.workdir = "/workspace"
    # These particular model/dataset assets are public; no Hub credential is needed.
    config.environment.env.pop("HF_TOKEN", None)
    config.verifier.env.pop("HF_TOKEN", None)
    config.verifier.env["PTB_AGENT_NAME"] = "terminus-2"
    config.metadata.update(
        benchmark="PostTrainBench", source_revision=REVISION, prepared_agent_seconds=hours * 3600
    )
    (task / "task.toml").write_text(config.model_dump_toml())
    preserved = {}
    for original, contents in sorted(native_files.items()):
        if original == Path("tests/Dockerfile"):
            continue
        destination = Path("tests/ptb-native-test.sh") if original == Path("tests/test.sh") else original
        assert (task / destination).read_bytes() == contents
        preserved[str(destination)] = hashlib.sha256(contents).hexdigest()
    (task / "source.json").write_text(
        json.dumps(
            {
                "benchmark": "PostTrainBench",
                "revision": REVISION,
                "hours": hours,
                "native_instruction_sha256": hashlib.sha256(original_instruction).hexdigest(),
                "preserved_native_verifier_sha256": preserved,
                "adaptations": [
                    "common /workspace paths and factual GPU availability, identical across interfaces",
                    "absolute native decontamination-tool paths",
                    "replace unused agent CLIs with Terminus terminal tools",
                "public GSM8K and Qwen assets do not require a Hub credential",
                    "native collection from a fresh verifier mount for shared GPU outputs",
                    "unchanged Terminus ATIF delivered through the trusted Harbor verification hook",
                ],
            },
            indent=2,
        )
        + "\n"
    )
    print(task)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/tasks"))
    parser.add_argument("--hours", type=int, default=1)
    args = parser.parse_args()
    prepare(args.source, args.output, args.hours)
