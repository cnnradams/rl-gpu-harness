import asyncio
import json
import stat
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

pytest.importorskip("harbor")

from harbor.environments.factory import EnvironmentFactory
from harbor.environments.modal import ModalEnvironment
from harbor.models.task.config import EnvironmentConfig as TaskEnvironmentConfig
from harbor.models.trial.config import EnvironmentConfig
from harbor.models.trial.paths import TrialPaths

import gpu_harness.harbor as code
from gpu_harness.harbor import GPUEnvironment


def make_environment(tmp_path, *, session_id="trial__env", **options):
    definition = tmp_path / "environment"
    definition.mkdir(exist_ok=True)
    (definition / "Dockerfile").write_text("FROM python:3.12-slim\nWORKDIR /workspace\n")
    return EnvironmentFactory.create_environment_from_config(
        config=EnvironmentConfig(
            import_path="gpu_harness.harbor:GPUEnvironment",
            kwargs={
                "gpu_limits": {"image_id": "im-task", "gpu": "T4"},
                "gpu_proxy_secret": "proxy-secret",
                **options,
            },
        ),
        environment_dir=definition,
        environment_name="smoke",
        session_id=session_id,
        trial_paths=TrialPaths(trial_dir=tmp_path / "trial"),
        task_env_config=TaskEnvironmentConfig(cpus=2, memory_mb=1024, gpus=0),
    )


async def test_shared_default_exec_stays_outside_mount_and_respects_explicit_cwd(tmp_path, monkeypatch):
    env = make_environment(tmp_path, gpu_limits={"image_id": "im-task", "shared_workspace": True})
    calls = []

    async def execute(command, **kwargs):
        calls.append(kwargs["cwd"])

    monkeypatch.setattr(env._strategy, "exec", execute)
    await env.exec("gpu sync")
    await env.exec("pwd", cwd="/workspace")
    assert calls == ["/", "/workspace"]


async def test_shared_start_prepares_gpu_mountpoint_before_minting(tmp_path, monkeypatch):
    env = make_environment(tmp_path, gpu_limits={"image_id": "im-task", "shared_workspace": True})
    calls = []

    async def build(app):
        calls.append("build")
        return SimpleNamespace(object_id="im-empty-workspace")

    def commands(command):
        assert command == "rm -rf /workspace && mkdir /workspace"
        return SimpleNamespace(build=SimpleNamespace(aio=build))

    async def lookup(*args):
        return object()

    async def mint(limits, run_id):
        calls.append(("mint", limits["image_id"]))
        return {"trial": "test", "limits": limits}

    async def start(self, force_build):
        calls.append("start")

    monkeypatch.setattr(code.modal.Image, "from_id", lambda image_id: SimpleNamespace(run_commands=commands))
    monkeypatch.setattr(code.modal.App.lookup, "aio", lookup)
    monkeypatch.setattr(
        code.modal.Function, "from_name", lambda *args: SimpleNamespace(remote=SimpleNamespace(aio=mint))
    )
    monkeypatch.setattr(ModalEnvironment, "start", start)
    await env.start(False)
    assert calls == ["build", ("mint", "im-empty-workspace"), "start"]


async def test_harbor_factory_links_trial_and_closes_session(tmp_path, monkeypatch):
    calls = []
    env = make_environment(tmp_path)
    env.context_id = uuid4()
    session = {"trial": "session-a", "url": "https://gpu.modal.run", "token": "private"}

    async def mint(limits, run_id):
        calls.append(("mint", limits, run_id))
        return session

    async def start(self, force_build):
        calls.append(("start", force_build))

    async def stop(self, delete):
        calls.append(("stop", delete))

    async def close(config, path):
        calls.append(("close", config, path))

    monkeypatch.setattr(
        code.modal.Function, "from_name", lambda *args: SimpleNamespace(remote=SimpleNamespace(aio=mint))
    )
    monkeypatch.setattr(ModalEnvironment, "start", start)
    monkeypatch.setattr(ModalEnvironment, "stop", stop)
    monkeypatch.setattr(code, "close_and_export", close)
    await env.start(False)
    await env.stop(True)
    assert calls[0][0] == "mint"
    assert calls[0][1]["gpu"] == "T4"
    assert calls[0][2] == str(env.context_id)
    assert calls[1:3] == [("start", False), ("stop", True)]
    assert calls[3] == ("close", session, env.trial_paths.trial_dir / "gpu-harness")
    assert "private" not in (env.trial_paths.trial_dir / "gpu-harness" / "instructions.md").read_text()
    await env.stop(True)
    assert sum(c[0] == "close" for c in calls) == 1


@pytest.mark.parametrize("failure", [RuntimeError("build failed"), asyncio.CancelledError()])
async def test_failed_or_cancelled_setup_closes_minted_session(tmp_path, monkeypatch, failure):
    env = make_environment(tmp_path)
    closed = []

    async def mint(*args):
        return {"trial": "a", "token": "private"}

    async def start(self, force_build):
        raise failure

    async def stop(self, delete):
        pass

    async def close(session, path):
        closed.append(session["trial"])

    monkeypatch.setattr(
        code.modal.Function, "from_name", lambda *args: SimpleNamespace(remote=SimpleNamespace(aio=mint))
    )
    monkeypatch.setattr(ModalEnvironment, "start", start)
    monkeypatch.setattr(ModalEnvironment, "stop", stop)
    monkeypatch.setattr(code, "close_and_export", close)
    with pytest.raises(type(failure)):
        await env.start(False)
    assert closed == ["a"]


@pytest.mark.parametrize("suffix", ["grade", "env", "a__env", "a" * 80])
@pytest.mark.parametrize("file_delivery", [False, True])
async def test_separate_verifier_does_not_mint_or_receive_agent_access(
    tmp_path, monkeypatch, suffix, file_delivery
):
    options = {"gpu_proxy_secret": None, "gpu_proxy_file": "absent.json"} if file_delivery else {}
    env = make_environment(tmp_path, session_id=f"trial__verifier__{suffix}", **options)
    calls = []

    async def start(self, force_build):
        calls.append("start")

    async def stop(self, delete):
        calls.append("stop")

    def forbidden(*args, **kwargs):
        raise AssertionError("Verifier tried to access the agent GPU session")

    monkeypatch.setattr(code.modal.Function, "from_name", forbidden)
    monkeypatch.setattr(code, "credential_policy", forbidden)
    monkeypatch.setattr(code, "close_and_export", forbidden)
    monkeypatch.setattr(ModalEnvironment, "start", start)
    monkeypatch.setattr(ModalEnvironment, "stop", stop)
    await env.start(False)
    await env.stop(True)
    assert calls == ["start", "stop"]
    assert env.gpu_session is None
    assert not (env.trial_paths.trial_dir / "gpu-harness").exists()


@pytest.mark.parametrize("agent_runtime", [None, "gvisor", "vm"])
async def test_native_create_keeps_credentials_outside_agent_and_preserves_resources(
    tmp_path, monkeypatch, agent_runtime
):
    env = make_environment(tmp_path, region="us-west", agent_runtime=agent_runtime)
    captured = {}
    image = SimpleNamespace()
    image.pip_install = image.add_local_file = image.run_commands = lambda *a, **k: image
    env._image, env._app = image, object()
    env.gpu_session = {"url": "https://gpu.modal.run", "token": "private"}

    async def create(*args, **kwargs):
        captured.update(kwargs)
        return "native-sandbox"

    monkeypatch.setattr(code.modal.Sandbox.create, "aio", create)
    monkeypatch.setattr(code, "credential_policy", lambda *args: "outside-agent-policy")
    assert await env._create_sandbox(entrypoint=["sleep", "infinity"]) == "native-sandbox"
    assert captured["name"] == "trial__env"
    assert captured["region"] == "us-west"
    assert captured.get("runtime") == agent_runtime
    assert captured["cpu"] == (2, 2)
    assert captured["memory"] == 1024
    assert "gpu" not in captured
    assert captured["env"]["GPU_HARNESS_TOKEN"] == "dummy"
    assert captured["env"]["GPU_HARNESS_URL"] == "https://gpu.modal.run"
    assert "private" not in str(captured)
    assert captured["_experimental_outbound_policy"] == "outside-agent-policy"
    assert issubclass(GPUEnvironment, ModalEnvironment)


@pytest.mark.parametrize("options", [{"gpu_proxy_secret": None}, {"gpu_proxy_file": "proxy.json"}])
def test_agent_requires_one_credential_delivery_method(tmp_path, options):
    with pytest.raises(ValueError, match="exactly one"):
        make_environment(tmp_path, **options)


async def test_file_delivery_uses_private_file_not_image_or_environment(tmp_path, monkeypatch):
    proxy = tmp_path / "operator-proxy.json"
    proxy.write_text(json.dumps({"Modal-Key": "key", "Modal-Secret": "secret"}))
    env = make_environment(tmp_path, gpu_proxy_secret=None, gpu_proxy_file=str(proxy))
    env.gpu_session = {"url": "https://gpu.modal.run", "token": "session-token"}
    captured = {}

    async def mint(*args):
        return env.gpu_session

    async def start(self, force_build):
        pass

    async def execute(command, **kwargs):
        captured.setdefault("exec", []).append((command, kwargs))
        return SimpleNamespace(return_code=0)

    async def upload(source, target):
        path = Path(source)
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        captured["credentials"] = json.loads(path.read_text())
        captured["target"] = target
        captured["temporary"] = path

    monkeypatch.setattr(
        code.modal.Function, "from_name", lambda *a: SimpleNamespace(remote=SimpleNamespace(aio=mint))
    )
    monkeypatch.setattr(ModalEnvironment, "start", start)
    monkeypatch.setattr(env, "exec", execute)
    monkeypatch.setattr(env, "upload_file", upload)
    await env.start(False)
    assert captured["credentials"] == {
        "url": "https://gpu.modal.run",
        "token": "session-token",
        "proxy": {"Modal-Key": "key", "Modal-Secret": "secret"},
    }
    assert captured["target"] == "/run/secrets/gpu-harness.json"
    assert not captured["temporary"].exists()
    assert captured["exec"][-1] == ("chmod 600 /run/secrets/gpu-harness.json", {"user": "root"})
    assert "session-token" not in (env.trial_paths.trial_dir / "gpu-harness/instructions.md").read_text()


async def test_file_delivery_keeps_native_network_allowlist(tmp_path, monkeypatch):
    from harbor.models.task.config import NetworkPolicy

    env = make_environment(
        tmp_path,
        gpu_proxy_secret=None,
        gpu_proxy_file="proxy.json",
        network_policy=NetworkPolicy(network_mode="allowlist", allowed_hosts=["gpu.modal.run"]),
    )
    image = SimpleNamespace()
    image.pip_install = image.add_local_file = image.run_commands = lambda *a, **k: image
    env._image, env._app = image, object()
    env.gpu_session = {"url": "https://gpu.modal.run", "token": "private"}
    captured = {}

    async def create(*args, **kwargs):
        captured.update(kwargs)
        return "native-sandbox"

    def forbidden(*args):
        raise AssertionError("File delivery must not enable header injection")

    monkeypatch.setattr(code.modal.Sandbox.create, "aio", create)
    monkeypatch.setattr(code, "credential_policy", forbidden)
    await env._create_sandbox()
    assert captured["_experimental_outbound_policy"] is None
    assert captured["outbound_domain_allowlist"] == ["gpu.modal.run"]
    assert "private" not in str(captured)


@pytest.mark.parametrize("binding_fails", [False, True])
async def test_shared_environment_seeds_and_binds_a_session_volume(tmp_path, monkeypatch, binding_fails):
    env = make_environment(tmp_path, gpu_limits={"image_id": "im-task", "shared_workspace": True})
    calls = []
    image = SimpleNamespace()
    image.pip_install = image.add_local_file = lambda *args, **kwargs: image

    def build(*args):
        calls.append(("image", args))
        return image

    image.run_commands = build
    env._image, env._app = image, object()
    env.gpu_session = {"url": "https://gpu.modal.run", "token": "private", "workspace_volume": "session-work"}

    async def wait():
        return 0

    async def execute(*argv):
        calls.append(("exec", argv))
        return SimpleNamespace(wait=SimpleNamespace(aio=wait))

    async def terminate():
        calls.append(("terminate",))

    sandbox = SimpleNamespace(
        object_id="agent-box", exec=SimpleNamespace(aio=execute), terminate=SimpleNamespace(aio=terminate)
    )

    async def create(*args, **kwargs):
        assert kwargs["workdir"] == "/"
        calls.append(("create", kwargs["volumes"]))
        return sandbox

    async def bind(*args):
        calls.append(("bind", args))
        if binding_fails:
            raise RuntimeError("Binding failed")

    monkeypatch.setattr(code.modal.Sandbox.create, "aio", create)
    monkeypatch.setattr(code.modal.Volume, "from_name", lambda name: name)
    monkeypatch.setattr(
        code.modal.Function, "from_name", lambda *args: SimpleNamespace(remote=SimpleNamespace(aio=bind))
    )
    monkeypatch.setattr(code, "credential_policy", lambda *args: None)
    if binding_fails:
        with pytest.raises(RuntimeError, match="Binding failed"):
            await env._create_sandbox()
        assert calls[-1] == ("terminate",)
    else:
        assert await env._create_sandbox() is sandbox
        assert ("create", {"/workspace": "session-work"}) in calls
        assert calls[-1] == ("bind", ("private", "agent-box"))
        assert "cp -a" in calls[-2][1][-1]


def test_idle_instructions_describe_behavior_without_strategy():
    from gpu_harness.service import Limits

    text = code.instructions(Limits(image_id="im-task", idle_seconds=300, idle_probe="cuda"))
    assert "300-second CUDA-idle timer resets" in text
    assert "is observed. Expiry stops the box." in text
    assert "command starts or finishes" not in text
    assert "CUDA-process checks are sampled about every 10 seconds." in text
    assert "failed GPU-process probe does not trigger" in text


def test_run_only_instructions_describe_lifetime_and_sync_without_extra_commands():
    from gpu_harness.service import Limits

    text = code.instructions(Limits(image_id="im-task", shared_workspace=True, run_only=True))
    assert "committed before each run" in text
    assert "box stops" in text and "CPU workspace view" in text
    assert "open" in text and "working directories" in text
    for operation in ["exec", "push", "pull", "refresh"]:
        assert f"gpu {operation}" not in text
    assert "should" not in text


@pytest.mark.parametrize("file_delivery", [False, True])
async def test_run_only_config_reaches_cpu_agent_for_both_credential_methods(
    tmp_path, monkeypatch, file_delivery
):
    proxy = tmp_path / "proxy.json"
    proxy.write_text(json.dumps({"Modal-Key": "key", "Modal-Secret": "secret"}))
    options = {"gpu_proxy_secret": None, "gpu_proxy_file": str(proxy)} if file_delivery else {}
    env = make_environment(
        tmp_path, gpu_limits={"image_id": "im-task", "shared_workspace": True, "run_only": True}, **options
    )
    captured = {}

    async def build(*args):
        return SimpleNamespace(object_id="im-prepared")

    async def mint(limits, run_id):
        return {"trial": "test", "url": "https://service.modal.run", "token": "private", "limits": limits}

    async def start(self, force_build):
        pass

    async def execute(command, **kwargs):
        return SimpleNamespace(return_code=0)

    async def upload(source, target):
        captured["config"] = json.loads(Path(source).read_text())
        captured["target"] = target
        assert stat.S_IMODE(Path(source).stat().st_mode) == 0o600

    monkeypatch.setattr(
        code.modal.Image,
        "from_id",
        lambda *args: SimpleNamespace(
            run_commands=lambda *args: SimpleNamespace(build=SimpleNamespace(aio=build))
        ),
    )
    monkeypatch.setattr(code.modal.App.lookup, "aio", build)
    monkeypatch.setattr(
        code.modal.Function, "from_name", lambda *args: SimpleNamespace(remote=SimpleNamespace(aio=mint))
    )
    monkeypatch.setattr(ModalEnvironment, "start", start)
    monkeypatch.setattr(env, "exec", execute)
    monkeypatch.setattr(env, "upload_file", upload)
    await env.start(False)
    assert captured["target"] == "/run/secrets/gpu-harness.json"
    assert captured["config"]["run_only"] is True
    assert captured["config"]["url"] == "https://service.modal.run"
    assert captured["config"]["token"] == ("private" if file_delivery else "dummy")
    assert captured["config"]["proxy"] == (
        {"Modal-Key": "key", "Modal-Secret": "secret"}
        if file_delivery
        else {"Modal-Key": "dummy", "Modal-Secret": "dummy"}
    )
    if not file_delivery:
        assert "private" not in json.dumps(captured)
