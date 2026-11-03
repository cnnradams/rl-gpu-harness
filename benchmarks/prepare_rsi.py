"""Prepare native RSI training/inference tasks for the controlled Harbor runner."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from harbor.models.task.config import TaskConfig, VerifierEnvironmentMode

from benchmarks.prepare import check_revision
from benchmarks.prepare_nano_gpt import REVISION, inline_timer

TASKS = ("moe-router-design", "kv-cache-pruning-sglang")


def prepare(source: Path, output: Path, name: str):
    check_revision(source, REVISION)
    upstream = source / "tasks" / name
    target = output / name
    if target.exists():
        raise FileExistsError(target)
    shutil.copytree(upstream, target)
    dockerfile = target / "environment/Dockerfile"
    recipe = dockerfile.read_text()
    # Agent CLIs are supplied by Harbor. Keep the native engine, model,
    # training, data builders and verifier intact; only add terminal tooling.
    if name == "moe-router-design":
        start, end = recipe.index("# Agent CLIs."), recipe.index("# Corpus built")
        recipe = recipe[:start] + recipe[end:]
        recipe = recipe.replace(
            "    && curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \\\n"
            "    && apt-get install -y --no-install-recommends nodejs \\\n",
            "",
        )
    else:
        start, end = recipe.index("# Node from the official tarball"), recipe.index("# Frozen engine source")
        recipe = recipe[:start] + recipe[end:]
    assert recipe.count("ripgrep procps") == 1
    recipe = recipe.replace("ripgrep procps", "ripgrep procps tmux asciinema")
    dockerfile.write_text(recipe)
    config = TaskConfig.model_validate_toml((target / "task.toml").read_text())
    assert config.verifier.environment_mode == VerifierEnvironmentMode.SEPARATE
    config.environment.gpus = 0
    inline_timer(config)
    # The runner supplies the actual horizon for each trial. Native target
    # training/scoring budgets and the isolated verifier remain unchanged.
    config.metadata.update(source_revision=REVISION, harness_study=True)
    (target / "task.toml").write_text(config.model_dump_toml())
    protected = [target / "instruction.md", *sorted((target / "tests").rglob("*"))]
    hashes = {}
    for path in protected:
        if path.is_file():
            relative = path.relative_to(target)
            assert path.read_bytes() == (upstream / relative).read_bytes()
            hashes[str(relative)] = hashlib.sha256(path.read_bytes()).hexdigest()
    (target / "source.json").write_text(
        json.dumps(
            {
                "benchmark": "RSI " + name,
                "revision": REVISION,
                "adaptations": [
                    "replace unused agent CLIs with Terminus terminal tools",
                    "GPU supplied by harness",
                    "execute original timer text without retaining an open Volume script file",
                ],
                "native_instruction_and_verifier_sha256": hashes,
                "native_scoring_and_training_preserved": True,
                "timer_launch": "execute original script text without retaining an open Volume file",
            },
            indent=2,
        )
        + "\n"
    )
    print(target)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/tasks"))
    parser.add_argument("--task", choices=TASKS, required=True)
    args = parser.parse_args()
    prepare(args.source, args.output, args.task)
