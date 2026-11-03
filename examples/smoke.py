"""Short live check; requires operator-created trial and proxy credentials on disk."""

import argparse
import json
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory

import httpx

from gpu_harness.client import Client


def run(url, trial_file, proxy_file):
    config = json.loads(Path(trial_file).read_text())
    client = Client(url, config["token"], proxy_file=proxy_file)
    assert httpx.get(f"{url}/policy").status_code in {401, 403}
    box = None

    def execute(argv):
        return client.request(
            "POST",
            f"/boxes/{box}/exec",
            key=uuid.uuid4().hex,
            json={"argv": argv, "cwd": "/workspace", "timeout": 30},
        ).json()

    def wait(command):
        deadline = time.monotonic() + 45
        while time.monotonic() < deadline:
            result = client.request("GET", f"/commands/{command['id']}").json()
            if result["state"] != "running":
                assert result["state"] == "succeeded", result
                return result
            time.sleep(0.25)
        raise TimeoutError("Command did not complete")

    try:
        key = uuid.uuid4().hex
        created = client.request("POST", "/boxes", key=key).json()
        box = created["id"]
        assert client.request("POST", "/boxes", key=key).json() == created
        with TemporaryDirectory() as scratch:
            source, destination = Path(scratch) / "source", Path(scratch) / "destination"
            source.write_bytes(b"ordinary files work")
            client.transfer(box, source, "/tmp/harness-smoke.txt", "push")
            client.transfer(box, destination, "/tmp/harness-smoke.txt", "pull")
            assert destination.read_bytes() == source.read_bytes()
        gpu = execute(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"])
        wait(gpu)
        gpu_name = client.request("GET", f"/commands/{gpu['id']}/logs").text.strip()
        with ThreadPoolExecutor(2) as pool:
            long = pool.submit(
                execute, ["python", "-u", "-c", "import time; print('training'); time.sleep(10)"]
            )
            inspect = pool.submit(execute, ["python", "-c", "print('inspection')"])
            long, inspect = long.result(), inspect.result()
        wait(inspect)
        assert client.request("GET", f"/commands/{long['id']}").json()["state"] == "running"
        wait(long)
        report = {
            "trial": config["trial"],
            "box": box,
            "gpu": gpu_name,
            "checks": [
                "proxy auth",
                "idempotent create",
                "files outside workspace",
                "native GPU exec",
                "concurrent commands",
                "persisted logs",
            ],
        }
        print(json.dumps(report, indent=2))
        Path(trial_file).with_suffix(".report.json").write_text(json.dumps(report, indent=2))
    finally:
        if box:
            client.request("DELETE", f"/boxes/{box}")
        client.http.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--trial-file", required=True)
    parser.add_argument("--proxy-file", required=True)
    run(**vars(parser.parse_args()))
