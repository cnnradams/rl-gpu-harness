import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

pytest.importorskip("harbor")

from fastapi.testclient import TestClient
from harbor.models.task.id import LocalTaskId
from harbor.models.trial.config import TaskConfig, TrialConfig
from harbor.models.trial.result import AgentInfo, ExceptionInfo, TrialResult
from harbor.models.verifier.result import VerifierResult
from harbor.viewer import create_app

from examples.review_trials import prepare


def hashes(root):
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.rglob("*")
        if p.is_file()
    }


def exported(source, name, rewards, *, error=False, finished=True):
    folder = source / name
    folder.mkdir(parents=True)
    config = TrialConfig(task=TaskConfig(path=Path("tasks") / name), trial_name=name)
    now = datetime.now(UTC)
    result = TrialResult(
        task_name=name,
        trial_name=name,
        trial_uri=folder.as_uri(),
        task_id=LocalTaskId(path=config.task.path),
        task_checksum="a" * 64,
        config=config,
        agent_info=AgentInfo(name="terminus-2", version="2.0"),
        verifier_result=VerifierResult(rewards=rewards) if rewards is not None else None,
        exception_info=ExceptionInfo.from_exception(RuntimeError("evaluation unavailable"))
        if error
        else None,
        started_at=now,
        finished_at=now + timedelta(seconds=10) if finished else None,
    )
    (folder / "result.json").write_text(result.model_dump_json(indent=2))
    (folder / "config.json").write_text(config.model_dump_json())
    (folder / "agent").mkdir()
    (folder / "agent/trajectory.json").write_text(json.dumps({"steps": [{"step_id": 1}]}))
    (folder / "submission.bin").write_bytes(b"retained native artifact\0\xff")
    return folder


def test_native_viewer_preserves_sources_missing_grades_and_original_kernel_result(tmp_path):
    source = tmp_path / "exports"
    kernel = exported(source, "kernelbench-example", {"correct": 1, "speedup": 2.5})
    exported(source, "posttrainbench-example", None, error=True)
    exported(source, "serving-example", {"invalid": 1, "reward": 0, "speedup": 0})
    before = hashes(source)
    output = prepare(source, tmp_path / "jobs/snapshot")
    assert hashes(source) == before
    for relative, expected in before.items():
        copied = output / relative
        if relative == f"{kernel.name}/result.json":
            copied = copied.with_name("result.native.json")
        assert hashlib.sha256(copied.read_bytes()).hexdigest() == expected
    with TestClient(create_app(output.parent)) as client:
        response = client.get("/api/jobs")
        assert response.status_code == 200 and response.json()["items"][0]["name"] == "snapshot"
        for folder in source.iterdir():
            native = json.loads((folder / "result.json").read_text())
            response = client.get(f"/api/jobs/snapshot/trials/{folder.name}")
            assert response.status_code == 200
            shown = response.json()
            assert shown["id"] == native["id"]
            assert shown["exception_info"] == native["exception_info"]
            if folder == kernel:
                assert shown["verifier_result"]["rewards"]["reward"] == 2.5
            else:
                assert shown["verifier_result"] == native["verifier_result"]
            trajectory = client.get(f"/api/jobs/snapshot/trials/{folder.name}/trajectory")
            assert trajectory.status_code == 200 and trajectory.json() == {"steps": [{"step_id": 1}]}
    assert json.loads((output / "review.json").read_text())["projected_native_result_sha256"] == {
        kernel.name: before[f"{kernel.name}/result.json"]
    }


def test_review_rejects_overwriting_sources_existing_output_and_pending_trials(tmp_path):
    source = tmp_path / "exports"
    exported(source, "example", None, finished=False)
    before = hashes(source)
    with pytest.raises(ValueError, match="outside"):
        prepare(source, source / "nested")
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError):
        prepare(source, existing)
    output = tmp_path / "jobs/new"
    with pytest.raises(ValueError, match="completed"):
        prepare(source, output)
    assert hashes(source) == before and not output.exists()


def test_source_alias_cannot_hide_nested_output(tmp_path):
    source = tmp_path / "original"
    folder = exported(source, "example", None)
    aliases = tmp_path / "aliases"
    aliases.mkdir()
    (aliases / folder.name).symlink_to(folder, target_is_directory=True)
    before = hashes(source)
    with pytest.raises(ValueError, match="outside"):
        prepare(aliases, folder / "nested")
    assert hashes(source) == before
