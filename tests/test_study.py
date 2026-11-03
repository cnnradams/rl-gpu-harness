import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("harbor")

from harbor.environments.factory import EnvironmentFactory
from harbor.environments.modal import ModalEnvironment
from harbor.models.task.config import EnvironmentConfig as TaskEnvironmentConfig
from harbor.models.trial.config import EnvironmentConfig
from harbor.models.trial.paths import TrialPaths

from benchmarks import environment
from benchmarks.manifest import runtime_sources, source_hashes, validate_launch
from gpu_harness.harbor import GPUEnvironment


@pytest.mark.parametrize(
    "kind,session,role",
    [
        ("StudyAttachedEnvironment", "trial__env", "agent"),
        ("StudyGPUEnvironment", "trial__env", "agent"),
        ("StudyGPUEnvironment", "trial__verifier_env", "verifier"),
    ],
)
async def test_native_factory_selects_pinned_phase_image_and_keeps_cleanup(
    tmp_path, monkeypatch, kind, session, role
):
    definition = tmp_path / "environment"
    definition.mkdir()
    (tmp_path / "task.toml").write_text('[verifier]\nenvironment_mode = "separate"\n')
    (definition / "Dockerfile").write_text("FROM intentionally-unused-image\n")
    trial_dir = tmp_path / "trial"
    trial_dir.mkdir()
    options = {"image_ids": {"agent": "im-agent", "verifier": "im-verifier"}}
    if kind == "StudyGPUEnvironment":
        options.update(
            gpu_limits={"image_id": "im-agent", "gpu": "H100"}, mode="manual", gpu_proxy_secret="proxy"
        )
    env = EnvironmentFactory.create_environment_from_config(
        config=EnvironmentConfig(import_path="benchmarks.environment:" + kind, kwargs=options),
        environment_dir=definition,
        environment_name="fixture",
        session_id=session,
        trial_paths=TrialPaths(trial_dir=trial_dir),
        task_env_config=TaskEnvironmentConfig(cpus=1, memory_mb=1024, gpus=0),
    )
    calls = []

    async def create(self, **kwargs):
        calls.append((self._image.object_id, kwargs))
        self._image = SimpleNamespace(object_id="im-native-derived")
        return SimpleNamespace(object_id="sb-native")

    async def start(self, force_build):
        assert force_build is False
        self._sandbox = await self._create_sandbox(entrypoint=["native", "entrypoint"])
        assert not (trial_dir / f"{role}-image.json").exists()

    async def stop(self, delete):
        calls.append(("native-stop", delete))

    async def sdk_exec(self, *args, **kwargs):
        return SimpleNamespace(return_code=1)

    monkeypatch.setattr(environment.modal.Image, "from_id", lambda name: SimpleNamespace(object_id=name))
    parent = GPUEnvironment if kind == "StudyGPUEnvironment" else ModalEnvironment
    monkeypatch.setattr(parent, "_create_sandbox", create)
    monkeypatch.setattr(parent, "start", start)
    monkeypatch.setattr(parent, "stop", stop)
    monkeypatch.setattr(parent, "_sdk_exec", sdk_exec)
    await env.start(False)
    assert env._sandbox.object_id == "sb-native"
    assert calls == [("im-" + role, {"entrypoint": ["native", "entrypoint"]})]
    initial = json.loads((trial_dir / f"{role}-image.json").read_text())
    assert initial["sandbox"] == "sb-native"
    assert initial["base_image"] == "im-" + role
    assert initial["runtime_image"] == "im-native-derived"
    assert "stop_completed" not in initial
    assert not (trial_dir / f"{role}-image.tmp").exists()
    assert (await env._sdk_exec("false")).return_code == 1
    first_access = env._study_first_exec
    await env._sdk_exec("another native command")
    assert env._study_first_exec == first_access
    await env.stop(True)
    assert calls[-1] == ("native-stop", True)
    observed = json.loads((trial_dir / f"{role}-image.json").read_text())
    assert observed["base_image"] == "im-" + role
    assert observed["runtime_image"] == "im-native-derived"
    assert observed["first_exec_returned_at"] == first_access
    assert observed["stop_completed"]
    await env.stop(True)
    assert json.loads((trial_dir / f"{role}-image.json").read_text()) == observed


async def test_start_metadata_error_preserves_owned_sandbox_for_native_cleanup(tmp_path, monkeypatch):
    trial_dir = tmp_path / "trial"
    trial_dir.mkdir()
    env = environment.StudyAttachedEnvironment.__new__(environment.StudyAttachedEnvironment)
    env.trial_paths = TrialPaths(trial_dir=trial_dir)
    env._study_runtime = None
    env._study_first_exec = None
    stopped = []

    async def start(self, force_build):
        self._sandbox = SimpleNamespace(object_id="sb-owned")
        self._study_runtime = {"role": "agent", "sandbox": self._sandbox.object_id}

    async def stop(self, delete):
        stopped.append((self._sandbox.object_id, delete))

    write = environment.StudyImages._write_study_runtime

    def failed_write(self):
        assert self._sandbox.object_id == "sb-owned"
        raise OSError("metadata filesystem unavailable")

    monkeypatch.setattr(ModalEnvironment, "start", start)
    monkeypatch.setattr(ModalEnvironment, "stop", stop)
    monkeypatch.setattr(environment.StudyImages, "_write_study_runtime", failed_write)
    with pytest.raises(OSError, match="metadata filesystem unavailable"):
        await env.start(False)
    assert env._sandbox.object_id == "sb-owned"
    monkeypatch.setattr(environment.StudyImages, "_write_study_runtime", write)
    await env.stop(True)
    assert stopped == [("sb-owned", True)]
    assert json.loads((trial_dir / "agent-image.json").read_text())["stop_completed"] is True


def test_runtime_hash_layout_matches_local_harness_and_dependency_files():
    root = Path(__file__).resolve().parent.parent
    assert runtime_sources(root) == source_hashes(root)


def test_launch_rejects_missing_images_and_image_drift():
    with pytest.raises(ValueError, match="both task and verifier"):
        validate_launch({"concurrency": 2, "tasks": {"task": {"gpu_image": "im-task"}}})
    with pytest.raises(ValueError, match="do not match"):
        validate_launch(
            {
                "concurrency": 2,
                "tasks": {"task": {"gpu_image": "im-task", "verifier_image": "im-tests"}},
                "trials": [
                    {
                        "task": "task",
                        "interface": "shared",
                        "agent_seconds": 3600,
                        "gpu_limits": {"image_id": "im-other"},
                    }
                ],
            }
        )


@pytest.mark.parametrize("concurrency", [1, 2, 32, 64, 65, 1024])
def test_launch_accepts_positive_concurrency(concurrency):
    validate_launch({"concurrency": concurrency, "tasks": {}, "trials": []})


@pytest.mark.parametrize("concurrency", [0, -1, 1.5, True])
def test_launch_rejects_nonpositive_or_noninteger_concurrency(concurrency):
    with pytest.raises(ValueError, match="positive integer"):
        validate_launch({"concurrency": concurrency, "tasks": {}, "trials": []})


@pytest.mark.parametrize("seconds", [0, -1, float("nan"), float("inf")])
def test_launch_rejects_invalid_agent_horizons(seconds):
    with pytest.raises(ValueError, match="positive"):
        validate_launch(
            {"concurrency": 1, "tasks": {}, "trials": [{"interface": "attached", "agent_seconds": seconds}]}
        )


@pytest.mark.parametrize("seconds", [1, 30, 14401, 86400])
def test_launch_has_no_study_specific_horizon_range(seconds):
    validate_launch(
        {"concurrency": 1, "tasks": {}, "trials": [{"interface": "attached", "agent_seconds": seconds}]}
    )


def test_agent_provider_uses_operator_config_and_keeps_judge_key(monkeypatch):
    from benchmarks.run import agent_provider

    monkeypatch.setenv("AGENT_API_KEY", "agent-only")
    monkeypatch.setenv("AGENT_API_BASE", "https://model.example/v1/")
    monkeypatch.setenv("AGENT_MODEL", "openai/operator-model")
    monkeypatch.setenv("OPENAI_API_KEY", "judge-only")
    assert agent_provider(SimpleNamespace(kwargs={}, model_name=None)) == (
        "https://model.example/v1",
        "agent-only",
        "operator-model",
    )
    assert os.environ["OPENAI_API_KEY"] == "judge-only"


def test_recorded_agent_provider_uses_manifest_model_and_endpoint(monkeypatch):
    from benchmarks.run import agent_provider

    monkeypatch.setenv("AGENT_API_KEY", "agent-only")
    monkeypatch.setenv("AGENT_API_BASE", "https://unrecorded.example/v1")
    monkeypatch.setenv("AGENT_MODEL", "unrecorded")
    config = SimpleNamespace(kwargs={"api_base": "https://recorded.example/v1"}, model_name="openai/pinned")
    assert agent_provider(config, recorded=True) == ("https://recorded.example/v1", "agent-only", "pinned")


def test_agent_provider_requires_explicit_operator_configuration(monkeypatch):
    from benchmarks.run import agent_provider

    for name in ("AGENT_API_KEY", "AGENT_API_BASE", "AGENT_MODEL"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(ValueError, match="AGENT_API_KEY"):
        agent_provider(SimpleNamespace(kwargs={}, model_name=None))


def test_operator_model_proxy_and_judge_secrets_are_configurable(monkeypatch):
    from benchmarks import run

    monkeypatch.setattr(run.modal.Secret, "from_name", lambda name: name)
    assert run.operator_secrets("agent-provider", "gpu-proxy", "judge-provider") == [
        "agent-provider",
        "gpu-proxy",
        "judge-provider",
    ]
    assert run.operator_secrets(proxy_secret="") == []
