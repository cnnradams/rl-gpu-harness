import shutil
from collections import Counter
from pathlib import Path

import pytest

pytest.importorskip("harbor")

from benchmarks.manifest import build, verify

ROOT = Path(__file__).resolve().parent.parent


def inputs(tmp_path, count=5):
    tasks = tmp_path / "tasks"
    specs = []
    for index in range(count):
        name = f"task-{index}"
        folder = tasks / name
        (folder / "tests").mkdir(parents=True)
        (folder / "task.toml").write_text(
            '[metadata]\nbenchmark = "fixture"\n'
            "[environment]\ncpus = 4\nmemory_mb = 16384\n"
            '[verifier]\ntimeout_sec = 7200\nenvironment_mode = "separate"\n'
        )
        (folder / "instruction.md").write_text("Original task instructions\n")
        (folder / "tests/test.sh").write_text("Original scorer\n")
        specs.append({"name": name, "seconds": 3600, "gpu_image": "im-fixture"})
    catalog = {
        "tasks": specs,
        "repeats": 3,
        "order_seed": 20261102,
        "concurrency": 2,
        "agent": {"name": "terminus-2", "model_name": "fixture"},
        "service": {"url": "https://fixture.invalid", "source_revision": "fixture"},
    }
    return catalog, tasks


def test_screening_design_has_all_45_distinct_cells_and_reproducible_order(tmp_path):
    catalog, tasks = inputs(tmp_path)
    manifest = build(catalog, tasks, ROOT)
    rows = manifest["trials"]
    assert len(rows) == 45
    assert Counter((r["task"], r["interface"]) for r in rows) == {
        (f"task-{index}", interface): 3
        for index in range(5)
        for interface in ("attached", "remote", "shared")
    }
    assert len({(r["task"], r["interface"], r["repeat"]) for r in rows}) == 45
    assert build(catalog, tasks, ROOT)["trials"] == rows
    for row in rows:
        if row["interface"] == "attached":
            assert row["agent_gpus"] == 1 and row["gpu_limits"] is None
        else:
            limits = row["gpu_limits"]
            assert limits["max_boxes"] == 1 and limits["gpu_seconds"] is None
            assert limits["idle_seconds"] is None
            assert limits["shared_workspace"] == (row["interface"] == "shared")


def test_manifest_detects_task_prompt_scorer_additions_and_removals(tmp_path):
    catalog, tasks = inputs(tmp_path, 1)
    manifest = build(catalog, tasks, ROOT)
    assert verify(manifest, tasks, ROOT) == []
    (tasks / "task-0/instruction.md").write_text("Changed prompt")
    (tasks / "task-0/tests/test.sh").unlink()
    (tasks / "task-0/tests/extra.py").write_text("Changed scoring inputs")
    assert verify(manifest, tasks, ROOT) == [
        "task-0/instruction.md",
        "task-0/tests/extra.py",
        "task-0/tests/test.sh",
    ]


def test_posttrain_native_horizon_must_match_the_planned_horizon(tmp_path):
    catalog, tasks = inputs(tmp_path, 1)
    config = tasks / "task-0/task.toml"
    config.write_text(
        config.read_text().replace(
            'benchmark = "fixture"',
            'benchmark = "PostTrainBench"\nprepared_agent_seconds = 14400',
        )
    )
    with pytest.raises(ValueError, match="prompt/timer horizon differs"):
        build(catalog, tasks, ROOT)


def test_manifest_detects_harness_edits_and_removed_dependency_lock(tmp_path):
    catalog, tasks = inputs(tmp_path, 1)
    manifest = build(catalog, tasks, ROOT)
    changed_root = tmp_path / "source"
    for name in ["src", "benchmarks"]:
        shutil.copytree(ROOT / name, changed_root / name)
    shutil.copyfile(ROOT / "pyproject.toml", changed_root / "pyproject.toml")
    (changed_root / "src/gpu_harness/service.py").write_text("changed runtime behavior")
    assert verify(manifest, tasks, changed_root) == [
        "source/src/gpu_harness/service.py",
        "source/uv.lock",
    ]


def test_native_judge_credential_requirement_comes_from_the_task_env(tmp_path):
    catalog, tasks = inputs(tmp_path, 1)
    assert not build(catalog, tasks, ROOT)["tasks"]["task-0"]["requires_verifier_secret"]
    config = tasks / "task-0/task.toml"
    config.write_text(config.read_text() + '\n[verifier.env]\nJUDGE_KEY = "${OPENAI_API_KEY:-}"\n')
    assert build(catalog, tasks, ROOT)["tasks"]["task-0"]["requires_verifier_secret"]
