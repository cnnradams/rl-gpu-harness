import sys
from types import SimpleNamespace

import pytest

import gpu_harness.client as code


@pytest.mark.parametrize(
    "argv,usage",
    [
        (["status"], "gpu status COMMAND"),
        (["wait"], "gpu wait COMMAND"),
        (["stop"], "gpu stop BOX"),
        (["sync", "BOX"], "gpu sync"),
        (["run"], "run requires a remote command after --"),
        (["push", "BOX", "local.py"], "gpu push BOX LOCAL REMOTE"),
        (["exec", "BOX"], "exec requires a remote command after --"),
        (["exec", "BOX", "nvidia-smi"], "gpu exec BOX [options] -- PROGRAM [ARG ...]"),
    ],
)
def test_invalid_arguments_explain_usage_before_opening_http_client(monkeypatch, capsys, argv, usage):
    def forbidden(**kwargs):
        raise AssertionError("Invalid invocation tried to open the HTTP client")

    monkeypatch.setattr(code, "Client", forbidden)
    monkeypatch.setattr(sys, "argv", ["gpu", *argv])
    with pytest.raises(SystemExit) as error:
        code.main()
    assert error.value.code == 2
    assert usage in capsys.readouterr().err


def test_run_starts_gpu_command_without_a_gpu_flag_or_create_step(monkeypatch, capsys):
    calls = []

    def request(method, path, **kwargs):
        calls.append((method, path, kwargs))
        return SimpleNamespace(json=lambda: {"id": "command", "box": "box", "state": "running"})

    monkeypatch.setattr(
        code,
        "Client",
        lambda **kwargs: SimpleNamespace(request=request, http=SimpleNamespace(close=lambda: None)),
    )
    monkeypatch.setattr(sys, "argv", ["gpu", "run", "--detach", "--key", "same", "--", "python", "train.py"])
    code.main()
    assert calls == [
        (
            "POST",
            "/run",
            {
                "key": "same",
                "json": {
                    "argv": ["python", "train.py"],
                    "cwd": "/workspace",
                    "timeout": None,
                },
            },
        )
    ]
    assert "box: box" in capsys.readouterr().err


def test_valid_detached_exec_preserves_remote_argv_and_options(monkeypatch, capsys):
    calls = []

    def request(method, path, **kwargs):
        calls.append((method, path, kwargs))
        return SimpleNamespace(json=lambda: {"id": "command-a", "state": "running"})

    closed = []
    monkeypatch.setattr(
        code,
        "Client",
        lambda **kwargs: SimpleNamespace(
            request=request, http=SimpleNamespace(close=lambda: closed.append(True))
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "gpu",
            "exec",
            "sb-a",
            "--detach",
            "--timeout",
            "17",
            "--cwd",
            "/tmp",
            "--key",
            "same-request",
            "--",
            "python3",
            "-c",
            "print('value with spaces')",
        ],
    )
    code.main()
    assert calls == [
        (
            "POST",
            "/boxes/sb-a/exec",
            {
                "key": "same-request",
                "json": {
                    "argv": ["python3", "-c", "print('value with spaces')"],
                    "cwd": "/tmp",
                    "timeout": 17,
                },
            },
        )
    ]
    assert closed == [True]
    assert '"id": "command-a"' in capsys.readouterr().out


def test_exec_does_not_add_an_unrequested_command_timeout(monkeypatch):
    payloads = []

    def request(method, path, **kwargs):
        payloads.append(kwargs["json"])
        return SimpleNamespace(json=lambda: {"id": "command-a", "state": "running"})

    monkeypatch.setattr(
        code,
        "Client",
        lambda **kwargs: SimpleNamespace(request=request, http=SimpleNamespace(close=lambda: None)),
    )
    monkeypatch.setattr(sys, "argv", ["gpu", "exec", "box", "--detach", "--", "python", "train.py"])
    code.main()
    assert payloads == [{"argv": ["python", "train.py"], "cwd": "/workspace", "timeout": None}]


@pytest.mark.parametrize("operation", ["run", "exec", "wait"])
@pytest.mark.parametrize("exit_code", [0, 17])
def test_foreground_output_can_be_piped_without_trailing_metadata(monkeypatch, capsys, operation, exit_code):
    def request(*args, **kwargs):
        return SimpleNamespace(json=lambda: {"id": "command", "box": "box", "state": "running"})

    def wait(command):
        print("native progress\nmetric: 123")
        return {
            "id": command,
            "box": "box",
            "trial": "trial",
            "state": "succeeded" if exit_code == 0 else "failed",
            "exit_code": exit_code,
        }

    monkeypatch.setattr(
        code,
        "Client",
        lambda **kwargs: SimpleNamespace(
            request=request,
            wait=wait,
            http=SimpleNamespace(close=lambda: None),
        ),
    )
    argv = {
        "run": ["run", "--", "python", "evaluate.py"],
        "exec": ["exec", "box", "--", "python", "evaluate.py"],
        "wait": ["wait", "command"],
    }[operation]
    monkeypatch.setattr(sys, "argv", ["gpu", *argv])
    if exit_code:
        with pytest.raises(SystemExit) as error:
            code.main()
        assert error.value.code == exit_code
    else:
        code.main()
    output = capsys.readouterr()
    assert output.out == "native progress\nmetric: 123\n"
    assert output.out.splitlines()[-1] == "metric: 123"


def test_foreground_service_failure_remains_visible(monkeypatch, capsys):
    monkeypatch.setattr(
        code,
        "Client",
        lambda **kwargs: SimpleNamespace(
            wait=lambda command: {"state": "unknown", "error": "Service restarted; inspect the box directly"},
            http=SimpleNamespace(close=lambda: None),
        ),
    )
    monkeypatch.setattr(sys, "argv", ["gpu", "wait", "command"])
    with pytest.raises(SystemExit) as error:
        code.main()
    assert error.value.code == 1
    output = capsys.readouterr()
    assert output.out == ""
    assert "Service restarted" in output.err


@pytest.mark.parametrize("exit_code", [0, 7])
def test_capture_error_is_printed_without_overriding_native_exit(monkeypatch, capsys, exit_code):
    detail = "Output capture failed (stdout: UnicodeDecodeError: invalid start byte)"
    monkeypatch.setattr(
        code,
        "Client",
        lambda **kwargs: SimpleNamespace(
            wait=lambda command: {
                "state": "succeeded" if exit_code == 0 else "failed",
                "exit_code": exit_code,
                "error": detail,
            },
            http=SimpleNamespace(close=lambda: None),
        ),
    )
    monkeypatch.setattr(sys, "argv", ["gpu", "wait", "command"])
    if exit_code:
        with pytest.raises(SystemExit) as error:
            code.main()
        assert error.value.code == exit_code
    else:
        code.main()
    assert capsys.readouterr().err == detail + "\n"


@pytest.mark.parametrize("nested", [False, True])
def test_run_only_help_omits_unavailable_operations(tmp_path, monkeypatch, capsys, nested):
    import json

    config = tmp_path / "run-only.json"
    config.write_text(json.dumps({"limits": {"run_only": True}} if nested else {"run_only": True}))
    monkeypatch.setattr(sys, "argv", ["gpu", "--config", str(config), "--help"])
    with pytest.raises(SystemExit) as error:
        code.main()
    assert error.value.code == 0
    help_text = capsys.readouterr().out
    assert "gpu run [options] -- PROGRAM" in help_text
    assert "gpu sync" in help_text
    for command in ["exec", "push", "pull", "refresh"]:
        assert f"gpu {command}" not in help_text
        assert command not in help_text.split("Commands:")[0]


@pytest.mark.parametrize("operation", ["exec", "push", "pull", "refresh"])
def test_run_only_rejects_hidden_operations_before_http(tmp_path, monkeypatch, capsys, operation):
    config = tmp_path / "run-only.json"
    config.write_text('{"run_only": true}')
    monkeypatch.setenv("GPU_HARNESS_CONFIG", str(config))
    monkeypatch.setattr(sys, "argv", ["gpu", operation])

    def forbidden(**kwargs):
        raise AssertionError("Unsupported command opened an HTTP client")

    monkeypatch.setattr(code, "Client", forbidden)
    with pytest.raises(SystemExit) as error:
        code.main()
    assert error.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


def test_run_only_preserves_run_argv_and_existing_wait_semantics(tmp_path, monkeypatch):
    config = tmp_path / "run-only.json"
    config.write_text('{"run_only": true}')
    calls = []

    def request(method, path, **kwargs):
        calls.append((method, path, kwargs))
        return SimpleNamespace(json=lambda: {"id": "main", "box": "box", "state": "running"})

    monkeypatch.setattr(
        code,
        "Client",
        lambda **kwargs: SimpleNamespace(request=request, http=SimpleNamespace(close=lambda: None)),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["gpu", "--config", str(config), "run", "--detach", "--key", "repeat", "--", "python", "train.py"],
    )
    code.main()
    assert calls == [
        (
            "POST",
            "/run",
            {"key": "repeat", "json": {"argv": ["python", "train.py"], "cwd": "/workspace", "timeout": None}},
        )
    ]


@pytest.mark.parametrize("operation", ["run", "exec"])
def test_successful_submission_displays_command_id_without_retry_key(monkeypatch, capsys, operation):
    command_id = "c" * 32
    retry_key = "a" * 32
    requests = []

    def request(*args, **kwargs):
        requests.append(kwargs["key"])
        return SimpleNamespace(json=lambda: {
            "id": command_id, "box": "sb-test", "trial": "internal-trial", "state": "running"})

    monkeypatch.setattr(code, "Client", lambda **kw: SimpleNamespace(
        request=request, http=SimpleNamespace(close=lambda: None)))
    monkeypatch.setattr(sys, "argv", ["gpu", operation, *(["sb-test"] if operation == "exec" else []),
                                    "--detach", "--key", retry_key, "--", "true"])
    code.main()
    output = capsys.readouterr()
    assert requests == [retry_key]
    assert f"command: {command_id}" in output.err
    assert retry_key not in output.out + output.err
    assert "request ID" not in output.err
    assert "internal-trial" not in output.out + output.err


@pytest.mark.parametrize("operation", ["run", "exec"])
@pytest.mark.parametrize("failure", [RuntimeError("connection lost"), KeyboardInterrupt()])
def test_unknown_submission_prints_generated_retry_key_without_retrying(
    monkeypatch, capsys, operation, failure
):
    calls, closed = [], []

    def request(*args, **kwargs):
        calls.append(kwargs)
        raise failure

    monkeypatch.setattr(code.uuid, "uuid4", lambda: SimpleNamespace(hex="generated-key"))
    monkeypatch.setattr(code, "Client", lambda **kw: SimpleNamespace(
        request=request, http=SimpleNamespace(close=lambda: closed.append(True))))
    monkeypatch.setattr(sys, "argv", ["gpu", operation, *(["sb-test"] if operation == "exec" else []),
                                    "--", "true"])
    with pytest.raises(KeyboardInterrupt if isinstance(failure, KeyboardInterrupt) else SystemExit):
        code.main()
    assert len(calls) == 1
    assert calls[0]["key"] == "generated-key"
    assert closed == [True]
    output = capsys.readouterr()
    assert "retry the same command with --key generated-key" in output.err
    assert "command:" not in output.err


@pytest.mark.parametrize("failure", [RuntimeError("wait failed"), KeyboardInterrupt()])
def test_known_command_wait_failure_keeps_id_without_retry_key(monkeypatch, capsys, failure):
    def wait(command_id):
        assert command_id == "known-command"
        raise failure

    monkeypatch.setattr(code, "Client", lambda **kw: SimpleNamespace(
        request=lambda *a, **k: SimpleNamespace(json=lambda: {
            "id": "known-command", "box": "sb-test", "state": "running"}),
        wait=wait, http=SimpleNamespace(close=lambda: None)))
    monkeypatch.setattr(sys, "argv", ["gpu", "run", "--key", "secret-retry-key", "--", "true"])
    with pytest.raises(KeyboardInterrupt if isinstance(failure, KeyboardInterrupt) else SystemExit):
        code.main()
    output = capsys.readouterr()
    assert "command: known-command" in output.err
    assert "--key" not in output.err
    assert "secret-retry-key" not in output.out + output.err


@pytest.mark.parametrize("status,retry_expected", [(403, False), (408, True), (503, True)])
def test_submission_http_rejection_only_suggests_retry_for_unknown_outcomes(
    monkeypatch, capsys, status, retry_expected
):
    request = code.httpx.Request("POST", "https://gpu.test/run")
    response = code.httpx.Response(status, request=request, text="service detail")

    # Capture the real response checker before replacing the CLI client constructor.
    checked = code.Client.checked

    def rejected(*args, **kwargs):
        return checked(response)

    monkeypatch.setattr(code, "Client", lambda **kw: SimpleNamespace(
        request=rejected, http=SimpleNamespace(close=lambda: None)))
    monkeypatch.setattr(sys, "argv", ["gpu", "run", "--key", "retry-id", "--", "true"])
    with pytest.raises(SystemExit) as error:
        code.main()
    assert error.value.code == 1
    output = capsys.readouterr()
    assert f"GPU service HTTP {status}: service detail" in output.err
    assert ("retry the same command with --key retry-id" in output.err) is retry_expected
    if not retry_expected:
        assert "response unavailable" not in output.err
        assert "retry-id" not in output.err
