"""Repeated development feedback and independent final KernelBench scoring."""

import argparse
import hashlib
import json
from pathlib import Path


def evaluate(solution: Path, *, correct_trials: int, perf_trials: int):
    import torch
    from kernelbench.eval import eval_kernel_against_ref

    reference = Path(__file__).with_name("reference.py")
    candidate = solution.read_text()
    result = eval_kernel_against_ref(
        reference.read_text(),
        candidate,
        seed_num=42,
        num_correct_trials=correct_trials,
        num_perf_trials=perf_trials,
        measure_performance=True,
        timing_method="cuda_event",
        backend="triton",
        precision=torch.float32,
    )
    if result is None:
        raise RuntimeError("Native evaluator returned no result; inspect evaluation output")
    report = result.model_dump()
    report.update(
        solution_sha256=hashlib.sha256(candidate.encode()).hexdigest(),
        reference_sha256=hashlib.sha256(reference.read_bytes()).hexdigest(),
        timing_unit="milliseconds",
        correct_trials=correct_trials,
        perf_trials=perf_trials,
    )
    report["speedup"] = (
        result.ref_runtime / result.runtime
        if result.compiled and result.correctness and result.runtime > 0 and result.ref_runtime > 0
        else 0.0
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("solution", type=Path)
    parser.add_argument("--correct-trials", type=int, default=5)
    parser.add_argument("--perf-trials", type=int, default=20)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--reward", type=Path, help="Final verifier's flat numeric reward JSON")
    args = parser.parse_args()
    if args.correct_trials < 1 or args.perf_trials < 1:
        parser.error("Trial counts must be positive")
    report = evaluate(args.solution, correct_trials=args.correct_trials, perf_trials=args.perf_trials)
    serialized = json.dumps(report, indent=2, default=str)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n")
    if args.reward:
        args.reward.parent.mkdir(parents=True, exist_ok=True)
        args.reward.write_text(
            json.dumps({"speedup": report["speedup"], "correct": int(report["correctness"])})
        )
    print(serialized)


if __name__ == "__main__":
    main()
