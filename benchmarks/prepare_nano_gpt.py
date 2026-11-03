"""Adapt the pinned RSI nanoGPT task to a short Harbor trial and separate verifier."""

import argparse
import json
import shutil
from pathlib import Path

from harbor.models.task.config import TaskConfig, VerifierEnvironmentMode

from benchmarks.prepare import check_revision

REVISION = "781a8091723b7a3651a3599e6b6d6063c82182bf"


def inline_timer(config):
    # A background `bash file.sh` keeps the Volume busy for its entire life.
    # Execute the same timer text without retaining the script file handle.
    if config.environment.healthcheck:
        config.environment.healthcheck.command = config.environment.healthcheck.command.replace(
            "bash /workspace/timer.sh", 'bash -c "$(cat /workspace/timer.sh)"'
        )


def prepare(source: Path, output: Path):
    check_revision(source, REVISION)
    upstream = source / "samples/nano-gpt-data-curation"
    task = output / "nano-gpt-data-curation"
    if task.exists():
        raise FileExistsError(task)
    shutil.copytree(upstream, task)
    recipe = (task / "environment/Dockerfile").read_text()
    # Keep the frozen training, tokenizer and dataset build. Agent CLIs belong
    # to Harbor; this image needs only the terminal tools used by Terminus.
    start, end = recipe.index("# Agent CLIs."), recipe.index("COPY workspace/model.py")
    recipe = recipe[:start] + recipe[end:]
    recipe = recipe.replace(
        "    && curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \\\n"
        "    && apt-get install -y --no-install-recommends nodejs \\\n",
        "    && apt-get install -y --no-install-recommends tmux asciinema \\\n",
    )
    (task / "environment/Dockerfile").write_text(recipe)
    verifier = task / "tests"
    for name in ["workspace", "assets"]:
        shutil.copytree(task / "environment" / name, verifier / name)
    shutil.copyfile(task / "environment/build_pool.py", verifier / "build_pool.py")
    (verifier / "Dockerfile").write_text(recipe + "\nCOPY . /tests\n")
    config = TaskConfig.model_validate_toml((task / "task.toml").read_text())
    config.schema_version = "1.4"
    config.agent.timeout_sec = 3600
    config.environment.gpus = 0
    config.environment.env["TASK_BUDGET_SECS"] = "3600"
    inline_timer(config)
    config.verifier.environment_mode = VerifierEnvironmentMode.SEPARATE
    config.verifier.timeout_sec = 600
    config.verifier.environment = config.environment.model_copy(deep=True)
    config.verifier.environment.gpus = 1
    config.verifier.environment.gpu_types = ["H100!"]
    config.verifier.environment.workdir = "/tests"
    config.verifier.environment.healthcheck = None
    config.metadata.update(source_revision=REVISION, harness_study=True, budget_wall_clock_hours=1)
    (task / "task.toml").write_text(config.model_dump_toml())
    (task / "source.json").write_text(
        json.dumps(
            {
                "benchmark": "RSI nano-gpt-data-curation",
                "revision": REVISION,
                "agent_horizon_seconds": 3600,
                "separate_verifier": True,
                "native_network_policy_preserved": True,
                "timer_launch": "execute original script text without retaining an open Volume file",
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
    args = parser.parse_args()
    prepare(args.source, args.output)
