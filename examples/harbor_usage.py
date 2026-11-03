"""Summarize exported GPU allocation alongside Harbor trial wall time."""

import argparse
import json
from datetime import datetime
from pathlib import Path


def duration(phase):
    if not phase or not phase.get("started_at") or not phase.get("finished_at"):
        return None
    return (
        datetime.fromisoformat(phase["finished_at"]) - datetime.fromisoformat(phase["started_at"])
    ).total_seconds()


def summarize(trial):
    result = json.loads((trial / "result.json").read_text())
    session_path = trial / "gpu-harness/session.json"
    session = json.loads(session_path.read_text()) if session_path.exists() else None
    boxes = session["boxes"] if session else []
    complete = bool(
        session
        and session.get("gpu_time_accounting_complete")
        and not session.get("creating_at_close")
        and all(b.get("ready_at") and b.get("ended_at") for b in boxes)
    )
    allocated = (
        sum(
            max(0, b["ended_at"] - b["ready_at"] - b.get("unavailable_seconds", 0)) * b["gpus"] for b in boxes
        )
        if complete
        else None
    )
    startup = sum(max(0, b["ready_at"] - b["requested_at"]) for b in boxes) if complete else None
    wall = duration(result.get("agent_execution"))
    return {
        "trial": result["trial_name"],
        "agent_wall_seconds": wall,
        "total_trial_wall_seconds": duration(result),
        "agent_allocated_gpu_seconds": allocated,
        "gpu_allocation_per_agent_wall_second": allocated / wall if allocated is not None and wall else None,
        "summed_gpu_startup_seconds": startup,
        "gpu_boxes": len(boxes),
        "accounting_complete": complete,
        "verifier_phase_wall_seconds": duration(result.get("verifier")),
        "reward": (result.get("verifier_result") or {}).get("rewards"),
        "error": (result.get("exception_info") or {}).get("exception_type"),
        "note": (
            "Allocation includes idle time, not CUDA utilization. "
            "Verifier phase time is not verifier GPU allocation. "
            "Startup is summed across boxes and may overlap."
        ),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trial", type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.trial), indent=2))
