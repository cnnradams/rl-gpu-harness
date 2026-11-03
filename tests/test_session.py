import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

import gpu_harness.app as deployment
import gpu_harness.session as launcher
from gpu_harness.client import Client
from gpu_harness.service import Limits, token_hash


def test_sessions_discover_url_and_keep_independent_resource_settings(monkeypatch):
    records = {}
    urls = iter(["https://first.modal.run", "https://second.modal.run"])
    lookups = []

    def lookup(app_name, function_name):
        lookups.append((app_name, function_name))
        return SimpleNamespace(get_url=lambda: next(urls))

    monkeypatch.setattr(deployment, "records", records)
    monkeypatch.setattr(deployment.modal.Server, "from_name", lookup)
    a = deployment.mint_session.local(
        Limits(image_id="im-a", gpu="H100", max_boxes=1).model_dump(), "harbor-run-a"
    )
    b = deployment.mint_session.local(
        Limits(
            image_id="im-b", gpu="A100", gpus_per_box=4, max_boxes=2, box_seconds=600, cpu=16, memory=65536
        ).model_dump(),
        "harbor-run-b",
    )
    assert a["url"] == "https://first.modal.run"
    assert b["url"] == "https://second.modal.run"
    assert lookups == [(deployment.APP_NAME, "gpu_api")] * 2
    assert a["token"] != b["token"]
    assert a["limits"]["max_boxes"] == 1
    assert b["limits"]["gpus_per_box"] == 4
    assert b["limits"]["memory"] == 65536
    assert b["limits"]["box_seconds"] == 600
    assert records[f"trial:{token_hash(b['token'])}"]["run_id"] == "harbor-run-b"
    assert all(session["token"] not in json.dumps(records) for session in (a, b))


def test_launcher_mints_through_authenticated_modal_rpc(monkeypatch):
    called = {}

    def lookup(app_name, function_name):
        called.update(app=app_name, function=function_name)

        def remote(limits, run_id):
            called.update(limits=limits, run_id=run_id)
            return {"trial": "minted-remotely"}

        return SimpleNamespace(remote=remote)

    monkeypatch.setattr(launcher.modal.Function, "from_name", lookup)
    limits = Limits(image_id="im-test", max_boxes=3)
    assert launcher.create_session(limits, "rollout-123") == {"trial": "minted-remotely"}
    assert called == {
        "app": launcher.APP_NAME,
        "function": "mint_session",
        "limits": limits.model_dump(),
        "run_id": "rollout-123",
    }


def test_client_uses_credential_files_and_ignores_secret_environment_variables(tmp_path, monkeypatch):
    path = tmp_path / "credentials.json"
    path.write_text(
        json.dumps(
            {
                "url": "https://service.modal.run",
                "token": "file-token",
                "proxy": {
                    "Modal-Key": "file-proxy-key",
                    "Modal-Secret": "file-proxy-secret",
                },
            }
        )
    )
    monkeypatch.setenv("GPU_HARNESS_TOKEN", "wrong-env-token")
    monkeypatch.setenv("GPU_HARNESS_PROXY_SECRET", "wrong-env-secret")
    client = Client(config_file=path)
    try:
        assert client.http.headers["Authorization"] == "Bearer file-token"
        assert client.http.headers["Modal-Secret"] == "file-proxy-secret"
    finally:
        client.http.close()


def test_url_only_config_leaves_auth_to_outbound_injection(tmp_path, monkeypatch):
    for key in ("GPU_HARNESS_TOKEN", "GPU_HARNESS_PROXY_KEY", "GPU_HARNESS_PROXY_SECRET"):
        monkeypatch.delenv(key, raising=False)
    path = tmp_path / "public-config.json"
    path.write_text(json.dumps({"url": "https://service.modal.run"}))
    client = Client(config_file=path)
    try:
        assert "Authorization" not in client.http.headers
        assert "Modal-Key" not in client.http.headers
        assert "Modal-Secret" not in client.http.headers
    finally:
        client.http.close()


def test_caller_can_supply_headers_from_its_own_credential_system():
    client = Client(
        "https://service.modal.run",
        headers={
            "Authorization": "Bearer caller-token",
            "Modal-Key": "caller-key",
            "Modal-Secret": "caller-secret",
        },
    )
    try:
        assert client.http.headers["Authorization"] == "Bearer caller-token"
        assert client.http.headers["Modal-Key"] == "caller-key"
    finally:
        client.http.close()


def test_injection_policy_targets_only_the_discovered_service_host(monkeypatch):
    import modal.experimental

    captured = {}

    class Policy:
        def with_header_replacement(self, **kwargs):
            captured.update(kwargs)
            return self

    monkeypatch.setattr(modal.experimental, "OutboundPolicy", Policy)
    monkeypatch.setattr(deployment.modal.Secret, "from_name", lambda name: name)
    launcher.credential_policy({"url": "https://exact.modal.run", "token": "session-token"}, "proxy-secret")
    assert captured["domain"] == "exact.modal.run"
    assert captured["secret"] == "proxy-secret"
    assert captured["headers"]["Authorization"] == "Bearer session-token"


def test_modal_launcher_can_supply_dummy_env_credentials(monkeypatch):
    monkeypatch.setenv("GPU_HARNESS_URL", "https://service.modal.run")
    for key in ("GPU_HARNESS_TOKEN", "GPU_HARNESS_PROXY_KEY", "GPU_HARNESS_PROXY_SECRET"):
        monkeypatch.setenv(key, "dummy")
    client = Client()
    try:
        assert str(client.http.base_url) == "https://service.modal.run"
        assert client.http.headers["Authorization"] == "Bearer dummy"
        assert client.http.headers["Modal-Key"] == "dummy"
        assert client.http.headers["Modal-Secret"] == "dummy"
    finally:
        client.http.close()


def test_client_runs_as_a_standalone_file():
    import gpu_harness.client as client

    result = subprocess.run([sys.executable, client.__file__, "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "policy" in result.stdout and "exec" in result.stdout


def test_only_policy_sessions_registered_before_credential_return(monkeypatch):
    records, markers, order = {}, {}, []

    class Index(dict):
        def __setitem__(self, key, value):
            assert key in records
            order.append(key)
            super().__setitem__(key, value)

    monkeypatch.setattr(deployment, "records", records)
    monkeypatch.setattr(deployment, "policy_index", Index())
    monkeypatch.setattr(
        deployment.modal.Server, "from_name", lambda *args: SimpleNamespace(get_url=lambda: "https://test")
    )
    for settings in ({}, {"gpu_seconds": 10}, {"idle_seconds": 300, "idle_probe": "cuda"}):
        session = deployment.mint_session.local(Limits(image_id="im-test", **settings).model_dump())
        key = "trial:" + token_hash(session["token"])
        if settings:
            assert order[-1] == key
            markers[key] = True
        else:
            assert key not in deployment.policy_index
    assert dict(deployment.policy_index) == markers
    assert len(records) == 3


def test_index_registration_failure_does_not_return_a_session_credential(monkeypatch):
    class UnavailableIndex:
        def __setitem__(self, key, value):
            raise RuntimeError("Index unavailable")

    monkeypatch.setattr(deployment, "records", {})
    monkeypatch.setattr(deployment, "policy_index", UnavailableIndex())
    monkeypatch.setattr(
        deployment.modal.Server, "from_name", lambda *args: SimpleNamespace(get_url=lambda: "https://test")
    )
    with pytest.raises(RuntimeError, match="Index unavailable"):
        deployment.mint_session.local(Limits(image_id="im-test", gpu_seconds=10).model_dump())


def test_run_only_operator_option_requires_shared_before_any_build():
    with pytest.raises(ValueError, match="requires shared_workspace"):
        launcher.save_session(run_only=True)


def test_run_only_operator_mints_shared_limits_and_saves_client_configuration(tmp_path, monkeypatch):
    captured = []

    def mint(limits):
        captured.append(limits)
        return {"trial": "test", "url": "https://test", "limits": limits.model_dump()}

    monkeypatch.setattr(launcher, "create_session", mint)
    path = tmp_path / "session.json"
    launcher.save_session(image_id="im-test", shared_workspace=True, run_only=True, output=str(path))
    assert captured[0].run_only and captured[0].shared_workspace
    assert json.loads(path.read_text())["limits"]["run_only"] is True
    assert path.stat().st_mode & 0o777 == 0o600
