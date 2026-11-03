"""Create controlled Harbor tasks from pinned benchmark checkouts."""

import argparse
import json
import shutil
import subprocess
from pathlib import Path

KERNELBENCH_REVISION = "423217d9fda91e0c2d67e4a43bf62f96f6d104f1"
KERNELS = {
    "softmax": "level1/23_Softmax.py",
    "fused-matmul": "level2/40_Matmul_Scaling_ResidualAdd.py",
    "irregular-matmul": "level1/8_Matmul_with_irregular_shapes_.py",
    "layernorm": "level1/40_LayerNorm.py",
}

CONFIG = """schema_version = "1.4"
artifacts = ["/workspace/solution.py"]

[metadata]
benchmark = "KernelBench"
source_revision = "{revision}"
problem = "{problem}"
metric = "correctness-gated reference/runtime speedup"
timing_unit = "milliseconds"

[agent]
timeout_sec = 3600

[environment]
cpus = 4
memory_mb = 16384
gpus = 0
gpu_types = ["H100!"]
workdir = "/workspace"
build_timeout_sec = 1800

[verifier]
timeout_sec = 600
environment_mode = "separate"

[verifier.environment]
cpus = 4
memory_mb = 16384
gpus = 1
gpu_types = ["H100!"]
workdir = "/tests"
build_timeout_sec = 1800
"""

INSTRUCTION = """Optimize the GPU computation in `/opt/task/reference.py` for one NVIDIA H100.
The input shapes, initial parameters and FP32 precision are defined by that file.

The submission is `/workspace/solution.py`, a Python module exporting `ModelNew`
with the reference model's constructor and forward signature. The final score is
reference runtime divided by submission runtime when correctness passes, and
zero for an incorrect implementation. Higher speedup is better.

`/opt/task/starter.py` provides a functional reference implementation.
`python3 /opt/task/evaluate.py /workspace/solution.py` on a GPU environment
provides correctness and timing feedback; it supports repeated calls and an
optional `--output PATH`. Its default is five correctness trials and twenty
timing trials. Final verification uses five correctness trials and one hundred
timing trials in a separate GPU environment. Reported times are milliseconds.
"""


def check_revision(source: Path, expected: str):
    actual = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if actual != expected:
        raise ValueError(f"Expected benchmark revision {expected}, found {actual}")
    if subprocess.check_output(["git", "-C", str(source), "status", "--porcelain"], text=True).strip():
        raise ValueError("Benchmark checkout has local changes; use a clean pinned source")


def kernelbench(source: Path, output: Path, names: list[str]):
    check_revision(source, KERNELBENCH_REVISION)
    template = Path(__file__).parent / "kernelbench"
    for name in names:
        task = output / f"kernelbench-{name}"
        if task.exists():
            raise FileExistsError(task)
        for role in ["environment", "tests"]:
            folder = task / role
            folder.mkdir(parents=True)
            shutil.copyfile(template / "evaluate.py", folder / "evaluate.py")
            shutil.copyfile(source / "KernelBench" / KERNELS[name], folder / "reference.py")
            (folder / "starter.py").write_text("from reference import Model as ModelNew\n")
            dockerfile = (template / "Dockerfile").read_text()
            if role == "tests":
                dockerfile += "RUN mkdir -p /tests && cp /opt/task/test.sh /tests/test.sh\n"
                (folder / "test.sh").write_text(
                    "#!/bin/sh\nset -eu\npython3 /opt/task/evaluate.py /workspace/solution.py "
                    "--correct-trials 5 --perf-trials 100 --output /logs/verifier/evaluation.json "
                    "--reward /logs/verifier/reward.json\n"
                )
            (folder / "Dockerfile").write_text(dockerfile)
        (task / "task.toml").write_text(CONFIG.format(revision=KERNELBENCH_REVISION, problem=KERNELS[name]))
        (task / "instruction.md").write_text(INSTRUCTION)
        solution = task / "solution"
        solution.mkdir()
        (solution / "solve.sh").write_text(
            "#!/bin/sh\nset -eu\ncp /opt/task/starter.py /workspace/solution.py\n"
        )
        (task / "source.json").write_text(
            json.dumps(
                {
                    "benchmark": "KernelBench",
                    "revision": KERNELBENCH_REVISION,
                    "problem": KERNELS[name],
                },
                indent=2,
            )
            + "\n"
        )
        print(task)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Clean pinned KernelBench checkout")
    parser.add_argument("--output", type=Path, default=Path("artifacts/tasks"))
    parser.add_argument("--kernels", nargs="+", choices=KERNELS, default=["softmax", "fused-matmul"])
    args = parser.parse_args()
    kernelbench(args.source, args.output, args.kernels)
