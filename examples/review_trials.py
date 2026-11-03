"""Prepare exported native trial folders for the upstream Harbor viewer."""

import argparse
import hashlib
import json
import math
import shlex
import shutil
import uuid
from pathlib import Path

from harbor.models.job.result import JobResult, JobStats
from harbor.models.trial.result import TrialResult


def prepare(source: Path, output: Path) -> Path:
    source, output = source.resolve(), output.resolve()
    if output.is_relative_to(source):
        raise ValueError("Output must be outside the source directory")
    if output.exists():
        raise FileExistsError(f"Choose a new output directory: {output}")
    trials = []
    for folder in sorted(source.iterdir()):
        if not (folder / "result.json").is_file():
            continue
        if output.is_relative_to(folder.resolve()):
            raise ValueError("Output must be outside every source trial directory")
        raw = (folder / "result.json").read_bytes()
        result = TrialResult.model_validate_json(raw)
        if not result.started_at or not result.finished_at or result.trial_name != folder.name:
            raise ValueError(f"Expected a completed native trial with matching directory name: {folder}")
        trials.append((folder, raw, result))
    if not trials:
        raise ValueError("No completed native trial exports found")

    output.mkdir(parents=True)
    rows, projections = [], {}
    for folder, raw, result in trials:
        target = output / folder.name
        shutil.copytree(folder, target, symlinks=True)
        rewards = result.verifier_result.rewards if result.verifier_result else None
        speedup = rewards.get("speedup") if rewards else None
        if (
            result.task_name.startswith("kernelbench-")
            and rewards is not None
            and "reward" not in rewards
            and rewards.get("correct") in (0, 1)
            and isinstance(speedup, (float, int))
            and math.isfinite(speedup)
        ):
            original = target / "result.native.json"
            if original.exists():
                raise ValueError(f"Reserved review file already exists: {original}")
            original.write_bytes(raw)
            data = json.loads(raw)
            data["verifier_result"]["rewards"]["reward"] = speedup if rewards["correct"] == 1 else 0
            (target / "result.json").write_text(json.dumps(data, indent=2) + "\n")
            result = TrialResult.model_validate(data)
            projections[folder.name] = hashlib.sha256(raw).hexdigest()
        rows.append(result)
    job = JobResult(
        id=uuid.uuid5(uuid.NAMESPACE_URL, str(output)),
        started_at=min(row.started_at for row in rows),
        updated_at=max(row.finished_at for row in rows),
        finished_at=max(row.finished_at for row in rows),
        n_total_trials=len(rows),
        stats=JobStats.from_trial_results(rows),
    )
    (output / "result.json").write_text(job.model_dump_json(indent=2) + "\n")
    (output / "review.json").write_text(
        json.dumps(
            {
                "scope": "Local view of these exports only; not the original study completion status.",
                "kernel_projection": "UI reward = speedup if correct=1, else 0; native result preserved.",
                "projected_native_result_sha256": projections,
            },
            indent=2,
        )
        + "\n"
    )
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Directory containing completed native trial folders")
    parser.add_argument("--output", required=True, type=Path, help="New viewer job directory")
    args = parser.parse_args()
    output = prepare(args.source, args.output)
    print(f"Prepared {output}.")
    print(f"uv run --extra harbor harbor view {shlex.quote(str(output.parent))} --jobs")


if __name__ == "__main__":
    main()
