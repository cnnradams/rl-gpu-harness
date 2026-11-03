"""Create a Modal proxy token and store it in a named Secret."""

import argparse
import json
import os
import subprocess
import sys
from tempfile import NamedTemporaryFile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--secret", default="rl-gpu-harness-proxy")
    parser.add_argument("--environment", default=os.environ.get("MODAL_ENVIRONMENT"), required=False)
    args = parser.parse_args()
    if not args.environment:
        parser.error("set MODAL_ENVIRONMENT or pass --environment")
    cli = [sys.executable, "-m", "modal"]
    existing = json.loads(
        subprocess.check_output(cli + ["secret", "list", "--env", args.environment, "--json"], text=True)
    )
    if any(secret["name"] == args.secret for secret in existing):
        parser.error(f"Secret {args.secret!r} already exists; reuse it or choose another name")
    token = json.loads(
        subprocess.check_output(
            cli + ["workspace", "proxy-tokens", "create", "--name", args.secret, "--json"], text=True
        )
    )
    token_id = token["Modal-Key"]
    try:
        subprocess.run(cli + ["workspace", "proxy-tokens", "allow", token_id, args.environment], check=True)
        with NamedTemporaryFile(mode="w", suffix=".json") as credentials:
            json.dump(
                {"MODAL_PROXY_TOKEN_ID": token_id, "MODAL_PROXY_TOKEN_SECRET": token["Modal-Secret"]},
                credentials,
            )
            credentials.flush()
            subprocess.run(
                cli
                + [
                    "secret",
                    "create",
                    args.secret,
                    "--env",
                    args.environment,
                    "--from-json",
                    credentials.name,
                ],
                check=True,
            )
    except BaseException:
        subprocess.run(cli + ["workspace", "proxy-tokens", "delete", token_id, "--yes"], check=False)
        raise
    print(f"Use --ek gpu_proxy_secret={args.secret} with Harbor.")


if __name__ == "__main__":
    main()
