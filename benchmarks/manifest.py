"""Describe a reproducible interface cohort without starting any trials."""

import argparse
import hashlib
import json
import math
import random
import subprocess
from pathlib import Path

from harbor.models.task.config import TaskConfig

from benchmarks.configuration import trial_instructions
from gpu_harness.service import Limits

INTERFACES = ("attached", "remote", "shared")


def file_hashes(folder):
    result = {}
    for path in sorted(folder.rglob("*")):
        if path.is_file():
            with path.open("rb") as stream:
                result[str(path.relative_to(folder))] = hashlib.file_digest(stream, "sha256").hexdigest()
    return result


def source_hashes(root):
    paths = [root / "pyproject.toml", root / "uv.lock"]
    paths.extend(sorted((root / "src").rglob("*.py")))
    paths.extend(sorted((root / "benchmarks").rglob("*.py")))
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
        if path.is_file()
    }


def build(catalog, tasks, root):
    """Catalogs contain public settings and evidence references, never credentials."""
    repeats = catalog["repeats"]
    if not isinstance(repeats, int) or repeats < 1:
        raise ValueError("repeats must be a positive integer")
    names = [task["name"] for task in catalog["tasks"]]
    if not names or len(set(names)) != len(names):
        raise ValueError("Choose distinct task names")
    cells, definitions = [], {}
    for task in catalog["tasks"]:
        name, seconds = task["name"], task["seconds"]
        folder = tasks / name
        config = TaskConfig.model_validate_toml((folder / "task.toml").read_text())
        if config.metadata.get("benchmark") == "PostTrainBench":
            if config.metadata["prepared_agent_seconds"] != seconds:
                raise ValueError(f"{name}: native PostTrainBench prompt/timer horizon differs")
        definitions[name] = {
            **task,
            "files": file_hashes(folder),
            "requires_verifier_secret": any(
                "OPENAI_API_KEY" in value for value in config.verifier.env.values()
            ),
        }
        for interface in INTERFACES:
            limits = Limits(
                image_id=task["gpu_image"],
                gpu="H100!",
                max_boxes=1,
                box_seconds=seconds,
                cpu=config.environment.cpus,
                memory=config.environment.memory_mb,
                shared_workspace=interface == "shared",
                block_network=config.environment.network_mode.value != "public",
            )
            text = trial_instructions(seconds, limits if interface != "attached" else None)
            for repeat in range(1, repeats + 1):
                cells.append(
                    {
                        "task": name,
                        "interface": interface,
                        "repeat": repeat,
                        "agent_seconds": seconds,
                        "verifier_seconds": config.verifier.timeout_sec,
                        "agent_gpus": int(interface == "attached"),
                        "gpu_limits": limits.model_dump() if interface != "attached" else None,
                        "extra_instructions": text,
                    }
                )
    random.Random(catalog["order_seed"]).shuffle(cells)
    return {
        "schema_version": 1,
        "scope": "Candidate interface screening; this file does not admit tasks or start trials",
        "source_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
        ).strip(),
        "source_files": source_hashes(root),
        "agent": catalog["agent"],
        "service": catalog["service"],
        "concurrency": catalog["concurrency"],
        "repeats": repeats,
        "order_seed": catalog["order_seed"],
        "tasks": definitions,
        "trials": cells,
    }


def verify(manifest, tasks, root, *, actual_source=None):
    """Report changed, added and removed inputs, including instructions and scorers."""
    differences = []
    groups = [
        ("source", manifest["source_files"], source_hashes(root) if actual_source is None else actual_source)
    ]
    groups.extend(
        (name, config["files"], file_hashes(tasks / name)) for name, config in manifest["tasks"].items()
    )
    for name, before, after in groups:
        differences.extend(
            f"{name}/{path}"
            for path in sorted(before.keys() | after.keys())
            if before.get(path) != after.get(path)
        )
    return differences


def runtime_sources(metadata=Path("/study-meta")):
    import benchmarks
    import gpu_harness

    files = {}
    for prefix, folder in [
        ("benchmarks", Path(benchmarks.__file__).parent),
        ("src/gpu_harness", Path(gpu_harness.__file__).parent),
    ]:
        for path in sorted(folder.rglob("*.py")):
            files[f"{prefix}/{path.relative_to(folder)}"] = hashlib.sha256(path.read_bytes()).hexdigest()
    for name in ["pyproject.toml", "uv.lock"]:
        files[name] = hashlib.sha256((metadata / name).read_bytes()).hexdigest()
    return files


def validate_launch(manifest):
    concurrency = manifest["concurrency"]
    if not isinstance(concurrency, int) or isinstance(concurrency, bool) or concurrency <= 0:
        raise ValueError("Concurrency must be a positive integer")
    for task in manifest["tasks"].values():
        if not task.get("gpu_image") or not task.get("verifier_image"):
            raise ValueError("Record both task and verifier base images before launch")
    for row in manifest["trials"]:
        if not math.isfinite(row["agent_seconds"]) or row["agent_seconds"] <= 0:
            raise ValueError("Agent horizon must be positive")
        if row["interface"] != "attached":
            if row["gpu_limits"]["image_id"] != manifest["tasks"][row["task"]]["gpu_image"]:
                raise ValueError("GPU limits do not match the recorded task image")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["build", "verify"])
    parser.add_argument("path", type=Path, help="Catalog for build; manifest for verify")
    parser.add_argument("--tasks", type=Path, default=Path("artifacts/tasks"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    data = json.loads(args.path.read_text())
    if args.action == "build":
        if args.output is None:
            parser.error("build requires --output")
        result = build(data, args.tasks, root)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(f"Wrote {len(result['trials'])} candidate trials; no work launched")
    else:
        changed = verify(data, args.tasks, root)
        if changed:
            raise SystemExit("Changed inputs:\n" + "\n".join(changed))
        print("Task and harness source files match; runtime/model/deployment checks remain separate")


if __name__ == "__main__":
    main()
