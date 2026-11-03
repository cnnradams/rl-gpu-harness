import argparse
import hashlib
import json
import os
import sys
import time
import uuid
from pathlib import Path
from tempfile import NamedTemporaryFile

import httpx


def configuration(config_file=None):
    path = Path(config_file or os.environ.get("GPU_HARNESS_CONFIG", "/run/secrets/gpu-harness.json"))
    if config_file and not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text()) if path.exists() else {}


class Client:
    def __init__(self, url=None, token=None, *, headers=None, config_file=None, proxy_file=None):
        config = configuration(config_file)
        url = url or config.get("url") or os.environ.get("GPU_HARNESS_URL")
        token = token or config.get("token") or os.environ.get("GPU_HARNESS_TOKEN")
        if not url:
            raise ValueError("Provide the service URL via arguments, config, or GPU_HARNESS_URL")
        auth_headers = dict(config.get("headers", {}))
        if token:
            auth_headers["Authorization"] = f"Bearer {token}"
        proxy = json.loads(Path(proxy_file).read_text()) if proxy_file else config.get("proxy", {})
        for key, env in [
            ("Modal-Key", "GPU_HARNESS_PROXY_KEY"),
            ("Modal-Secret", "GPU_HARNESS_PROXY_SECRET"),
        ]:
            if key not in proxy and os.environ.get(env):
                proxy[key] = os.environ[env]
        auth_headers.update({key: proxy[key] for key in ("Modal-Key", "Modal-Secret") if key in proxy})
        auth_headers.update(headers or {})
        # External injectors may supply auth outside the client; no delivery mode is required.
        self.http = httpx.Client(base_url=url, headers=auth_headers, timeout=600)

    def transfer(self, box, local, remote, operation):
        local = Path(local).resolve()
        endpoint = f"/boxes/{box}/file"
        digest, count = hashlib.sha256(), 0
        if operation == "push":
            with local.open("rb") as source:
                size = os.fstat(source.fileno()).st_size

                def chunks():
                    nonlocal count
                    while chunk := source.read(1024 * 1024):
                        digest.update(chunk)
                        count += len(chunk)
                        yield chunk

                result = self.request(
                    "PUT",
                    endpoint,
                    params={"path": remote},
                    content=chunks(),
                    headers={"Content-Length": str(size)},
                ).json()
            if result != {"bytes": count, "sha256": digest.hexdigest()}:
                raise RuntimeError("Upload integrity check failed")
            return result
        temporary = None
        try:
            with self.http.stream("GET", endpoint, params={"path": remote}) as response:
                if not response.is_success:
                    response.read()
                    self.checked(response)
                with NamedTemporaryFile(dir=local.parent, delete=False) as output:
                    temporary = Path(output.name)
                    for chunk in response.iter_bytes(1024 * 1024):
                        output.write(chunk)
                        digest.update(chunk)
                        count += len(chunk)
            temporary.chmod(local.stat().st_mode & 0o777 if local.exists() else 0o644)
            temporary.replace(local)
            return {"bytes": count, "sha256": digest.hexdigest(), "saved": str(local)}
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def request(self, method, path, *, key=None, **kwargs):
        headers = kwargs.pop("headers", {})
        if key:
            headers["Idempotency-Key"] = key
        response = self.http.request(method, path, headers=headers, **kwargs)
        return self.checked(response)

    @staticmethod
    def checked(response):
        if not response.is_success:
            raise httpx.HTTPStatusError(
                f"GPU service HTTP {response.status_code}: {response.text}",
                request=response.request,
                response=response,
            )
        return response

    def wait(self, command):
        offsets = {"stdout": 0, "stderr": 0}
        while True:
            status = self.request("GET", f"/commands/{command}").json()
            for name, output in [("stdout", sys.stdout), ("stderr", sys.stderr)]:
                while True:
                    chunk = self.request(
                        "GET", f"/commands/{command}/logs", params={"stream": name, "offset": offsets[name]}
                    ).content
                    if not chunk:
                        break
                    offsets[name] += len(chunk)
                    output.buffer.write(chunk)
                    output.flush()
            if status["state"] != "running":
                return status
            time.sleep(0.5)


ARGUMENTS = {
    "policy": (),
    "run": (),
    "stop": ("BOX",),
    "exec": ("BOX",),
    "status": ("COMMAND",),
    "wait": ("COMMAND",),
    "push": ("BOX", "LOCAL", "REMOTE"),
    "pull": ("BOX", "REMOTE", "LOCAL"),
    "refresh": ("BOX",),
    "sync": (),
}


def main():
    raw = sys.argv[1:]
    option_args = raw[: raw.index("--")] if "--" in raw else raw
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config")
    configured, _ = config_parser.parse_known_args(option_args)
    try:
        config = configuration(configured.config)
    except (OSError, ValueError) as error:
        config_parser.error(str(error))
    run_only = config.get("run_only", config.get("limits", {}).get("run_only", False)) is True
    arguments = {
        name: params
        for name, params in ARGUMENTS.items()
        if not run_only or name in {"policy", "run", "stop", "status", "wait", "sync"}
    }
    parser = argparse.ArgumentParser(
        prog="gpu",
        description="GPU boxes through a shared Modal service",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            """Commands:
  gpu policy
  gpu run [options] -- PROGRAM [ARG ...]
  gpu stop BOX
  gpu status COMMAND
  gpu wait COMMAND
  gpu sync

run starts a GPU box for its command and returns command and box IDs.
The box stops when that command exits. run waits unless --detach is set.
CPU /workspace writes are committed before each run; GPU writes commit on exit.
gpu sync commits and reloads the CPU workspace view.
status and wait take the returned command IDs.
Remote programs receive EOF on stdin; local stdin is not forwarded.
The -- separator starts the remote program and its arguments.
"""
            if run_only
            else """Commands:
  gpu policy
  gpu run [options] -- PROGRAM [ARG ...]
  gpu stop BOX
  gpu exec BOX [options] -- PROGRAM [ARG ...]
  gpu status COMMAND
  gpu wait COMMAND
  gpu push BOX LOCAL REMOTE
  gpu pull BOX REMOTE LOCAL
  gpu refresh BOX
  gpu sync

run starts a GPU box with its first command and returns command and box IDs.
The run command is the sandbox entrypoint; when it exits, the box stops.
exec is available while that box is running.
run and exec wait unless --detach is set.
status and wait take the command IDs they return.
Remote programs receive EOF on stdin; local stdin is not forwarded.
The -- separator starts the remote program and its arguments.
"""
        ),
    )
    parser.add_argument("operation", choices=arguments)
    parser.add_argument("args", nargs="*")
    parser.add_argument("--key", help="Reuse this request ID after a lost response")
    parser.add_argument("--timeout", type=int, help="Optional timeout for the remote command, in seconds")
    parser.add_argument("--cwd", default="/workspace")
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--config", help="Optional JSON config instead of environment configuration")
    argv = raw[raw.index("--") + 1 :] if "--" in raw else []
    args = parser.parse_args(option_args)
    names = arguments[args.operation]
    usage = " ".join(["gpu", args.operation, *names])
    if args.operation in {"run", "exec"}:
        usage += " [options] -- PROGRAM [ARG ...]"
    if len(args.args) != len(names):
        parser.error(f"usage: {usage}")
    if args.operation in {"run", "exec"} and not argv:
        parser.error(f"{args.operation} requires a remote command after --; usage: {usage}")
    client = Client(config_file=args.config)
    try:
        key = args.key or uuid.uuid4().hex
        def submit(path):
            try:
                result = client.request(
                    "POST", path, key=key,
                    json={"argv": argv, "cwd": args.cwd, "timeout": args.timeout},
                ).json()
                result["id"]  # A response without a command ID is still an unknown submission outcome.
            except (Exception, KeyboardInterrupt) as error:
                if isinstance(error, httpx.HTTPStatusError):
                    status = error.response.status_code
                    if 400 <= status < 500 and status != 408:
                        raise
                print(
                    f"Command response unavailable; retry the same command with --key {key}",
                    file=sys.stderr, flush=True,
                )
                raise
            return result

        match args.operation:
            case "policy":
                result = client.request("GET", "/policy").json()
            case "run":
                result = submit("/run")
                print(f"box: {result['box']}", file=sys.stderr, flush=True)
                print(f"command: {result['id']}", file=sys.stderr, flush=True)
                if not args.detach:
                    result = client.wait(result["id"])
            case "stop":
                result = client.request("DELETE", f"/boxes/{args.args[0]}").json()
            case "exec":
                result = submit(f"/boxes/{args.args[0]}/exec")
                print(f"command: {result['id']}", file=sys.stderr, flush=True)
                if not args.detach:
                    result = client.wait(result["id"])
            case "status":
                result = client.request("GET", f"/commands/{args.args[0]}").json()
            case "wait":
                result = client.wait(args.args[0])
            case "refresh":
                result = client.request("POST", f"/boxes/{args.args[0]}/refresh").json()
            case "sync":
                result = client.request("POST", "/sync").json()
            case "push":
                result = client.transfer(args.args[0], args.args[1], args.args[2], "push")
            case "pull":
                result = client.transfer(args.args[0], args.args[2], args.args[1], "pull")
        foreground = args.operation == "wait" or args.operation in {"run", "exec"} and not args.detach
        if not foreground:
            visible = result
            if args.operation in {"run", "exec"}:
                visible = {k: v for k, v in result.items() if k != "trial"}
            print(json.dumps(visible, indent=2))
        elif result.get("error"):
            print(result["error"], file=sys.stderr)
        if foreground and result.get("exit_code") not in {None, 0}:
            raise SystemExit(result["exit_code"])
        if result.get("state") in {"failed", "unknown"}:
            raise SystemExit(1)
    except (httpx.HTTPError, OSError, RuntimeError, IndexError) as error:
        parser.exit(1, f"{error}\n")
    finally:
        client.http.close()


if __name__ == "__main__":
    main()
