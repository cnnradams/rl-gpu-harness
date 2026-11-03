"""Run one short native Harbor verification trial on a trusted Modal function."""

import json
import math
import os
import shutil
import tarfile
import uuid
from pathlib import Path
from tempfile import TemporaryDirectory, TemporaryFile
from urllib.parse import urlsplit

import modal

app = modal.App("gpu-harness-benchmark-check")
results = modal.Volume.from_name("gpu-harness-validation-results", create_if_missing=True, version=2)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("harbor[modal]==0.23.0", "modal==1.6.3", "fastapi==0.162.0", "litellm[mcp]==1.106.6")
    .add_local_python_source("gpu_harness", copy=True)
    .add_local_python_source("benchmarks", copy=True)
    .add_local_dir("artifacts/tasks", "/tasks", copy=True)
)


def agent_provider(agent_config, *, recorded=False):
    """Read trusted operator credentials without overriding native judge credentials."""
    key = os.environ.get("AGENT_API_KEY")
    url = agent_config.kwargs.get("api_base") if recorded else os.environ.get("AGENT_API_BASE")
    model = agent_config.model_name if recorded else os.environ.get("AGENT_MODEL")
    if not key or not url or not model:
        raise ValueError("Configure AGENT_API_KEY and model/API base via the provider Secret or manifest")
    return url.rstrip("/"), key, model.removeprefix("openai/")


def operator_secrets(agent_secret="", proxy_secret="rl-gpu-harness-proxy", verifier_secret=""):
    return [modal.Secret.from_name(name) for name in (agent_secret, proxy_secret, verifier_secret) if name]


async def execute(
    task: str,
    gpu_image: str = "",
    interface: str = "remote",
    agent: str = "oracle",
    seconds: int = 900,
    *,
    study: dict | None = None,
    agent_runtime: str | None = None,
):
    import httpx
    import litellm
    from harbor.models.task.config import NetworkMode
    from harbor.models.task.config import TaskConfig as TaskDefinition
    from harbor.models.trial.config import AgentConfig, EnvironmentConfig, TaskConfig, TrialConfig
    from harbor.trial.hooks import TrialEvent
    from harbor.trial.trial import Trial

    from benchmarks.configuration import trial_instructions
    from benchmarks.export import contains_secret, copy_sources
    from gpu_harness import APP_NAME, SERVER_NAME
    from gpu_harness.service import Limits

    if interface not in {"attached", "remote", "shared"} or agent not in {"oracle", "terminus-2"}:
        raise ValueError("Choose attached/remote/shared and oracle/terminus-2")
    if interface == "attached" and agent_runtime is not None:
        raise ValueError("agent_runtime applies to CPU agents with remote GPU tools")
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("Agent horizon must be positive")
    if interface != "attached" and not gpu_image:
        raise ValueError("Remote GPU access requires a built task image")
    secret = ""
    agent_config = AgentConfig.model_validate(study["agent"]) if study else AgentConfig(name=agent)
    model = None
    if agent == "terminus-2":
        url, secret, wanted = agent_provider(agent_config, recorded=study is not None)
        litellm.openai_key = secret
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(url + "/models", headers={"Authorization": "Bearer " + secret})
            response.raise_for_status()
            models = response.json()["data"]
        model = next(m for m in models if m["id"] == wanted)
        context = model["context_length"]
        if study:
            info = agent_config.kwargs["model_info"]
            if info["max_input_tokens"] + info["max_output_tokens"] > context:
                raise ValueError("Live model context is smaller than the recorded configuration")
        else:
            agent_config.model_name = "openai/" + model["id"]
            agent_config.kwargs = {
                "api_base": url,
                "enable_summarize": True,
                "proactive_summarization_threshold": 8000,
                "record_terminal_session": True,
                "store_all_messages": True,
                "trajectory_config": {"linear_history": True},
                "model_info": {
                    "max_input_tokens": context - 16384,
                    "max_output_tokens": 16384,
                    "input_cost_per_token": 0.0,
                    "output_cost_per_token": 0.0,
                },
                "llm_call_kwargs": {"max_tokens": 16384},
            }
    with TemporaryDirectory() as credentials:
        folder = Path(credentials) / Path(task).name
        shutil.copytree(Path("/tasks") / task, folder)
        definition = TaskDefinition.model_validate_toml((folder / "task.toml").read_text())
        posttrain = definition.metadata.get("benchmark") == "PostTrainBench"
        if posttrain and definition.metadata["prepared_agent_seconds"] != seconds:
            raise ValueError("Prepare PostTrainBench's native prompt and timer for the requested horizon")
        definition.agent.timeout_sec = seconds
        if study:
            if definition.verifier.timeout_sec != study["row"]["verifier_seconds"]:
                raise ValueError("Native verifier horizon differs from the manifest")
        definition.environment.gpus = int(interface == "attached")
        definition.environment.gpu_types = ["H100!"]
        definition.verifier.environment.gpu_types = ["H100!"]
        if "TASK_BUDGET_SECS" in definition.environment.env:
            definition.environment.env["TASK_BUDGET_SECS"] = str(seconds)
            definition.metadata["budget_wall_clock_hours"] = seconds / 3600
        if posttrain and interface == "shared":
            # The fresh verifier collects committed GPU files directly; the
            # agent's mounted snapshot may still be stale at trial end.
            definition.verifier.collect = []
        (folder / "task.toml").write_text(definition.model_dump_toml())
        proxy = {
            "Modal-Key": os.environ["MODAL_PROXY_TOKEN_ID"],
            "Modal-Secret": os.environ["MODAL_PROXY_TOKEN_SECRET"],
        }
        proxy_path = Path(credentials) / "proxy.json"
        proxy_path.touch(mode=0o600)
        proxy_path.write_text(json.dumps(proxy))
        private_values = [secret, *proxy.values()]
        private_values.extend(
            os.environ[name]
            for name in ("OPENAI_API_KEY", "CODEX_API_KEY", "HF_TOKEN")
            if os.environ.get(name)
        )
        options = {"sandbox_timeout_secs": int(max(seconds, definition.verifier.timeout_sec)) + 300}
        if posttrain and interface != "shared":
            from benchmarks.posttrain import model_volume

            options["volumes"] = {"/mnt/ptb_final_model": await model_volume()}
        if study:
            options["image_ids"] = {
                "agent": study["task"]["gpu_image"],
                "verifier": study["task"]["verifier_image"],
            }
        allowed_hosts = []
        if interface != "attached":
            options["agent_runtime"] = agent_runtime
            service_url = await modal.Server.from_name(APP_NAME, SERVER_NAME).get_url.aio()
            if study and service_url != study["service"]["url"]:
                raise ValueError("Deployed GPU service URL differs from the manifest")
            if definition.environment.network_mode == NetworkMode.ALLOWLIST:
                allowed_hosts = [urlsplit(service_url).hostname]
            options.update(
                gpu_proxy_file=str(proxy_path),
                gpu_limits={
                    "image_id": gpu_image,
                    "gpu": "H100!",
                    "max_boxes": 1,
                    "box_seconds": seconds,
                    "cpu": definition.environment.cpus,
                    "memory": definition.environment.memory_mb,
                    "block_network": definition.environment.network_mode != NetworkMode.PUBLIC,
                    "shared_workspace": interface == "shared",
                },
            )
        if study and interface != "attached":
            options["gpu_limits"] = study["row"]["gpu_limits"]
        environment_class = "gpu_harness.harbor:GPUEnvironment" if interface != "attached" else None
        if study:
            environment_class = (
                "benchmarks.environment:StudyGPUEnvironment"
                if interface != "attached"
                else "benchmarks.environment:StudyAttachedEnvironment"
            )
        # Harbor appends "__env"; Modal sandbox names must be shorter than 64 characters.
        trial_prefix = f"{task}-{interface}-{agent}"
        trial = await Trial.create(
            TrialConfig(
                task=TaskConfig(path=folder),
                trial_name=f"{trial_prefix[:49]}-{uuid.uuid4().hex[:8]}",
                trials_dir=Path("/results") / study["cohort"] if study else Path("/results"),
                agent=agent_config,
                extra_instructions=(
                    study["row"]["extra_instructions"]
                    if study
                    else trial_instructions(
                        seconds, Limits(**options["gpu_limits"]) if interface != "attached" else None
                    )
                ),
                environment=EnvironmentConfig(
                    import_path=environment_class,
                    type="modal",
                    cpu_enforcement_policy="request",
                    extra_allowed_hosts=allowed_hosts,
                    kwargs=options,
                ),
            )
        )
        if posttrain:
            from benchmarks.posttrain import add_verification_hook

            add_verification_hook(trial, interface=interface, agent=agent)

        async def verify_access(event):
            if interface != "attached":
                private_values.append(trial.agent_environment.gpu_session["token"])
            code = 'test -z "$OPENAI_API_KEY$AGENT_API_KEY$MODAL_PROXY_TOKEN_ID$MODAL_PROXY_TOKEN_SECRET"'
            if interface != "attached":
                code += " && test $(stat -c %a /run/secrets/gpu-harness.json) = 600 && gpu policy"
            if interface == "shared":
                code += " && gpu sync"
            observed = await trial.agent_environment.exec(code)
            (trial.paths.trial_dir / "credential-isolation.json").write_text(
                json.dumps(
                    {
                        "return_code": observed.return_code,
                        "stdout": observed.stdout,
                        "stderr": observed.stderr,
                    },
                    indent=2,
                )
                + "\n"
            )
            if observed.return_code:
                raise RuntimeError("Agent startup check failed; see credential-isolation.json")
            if interface == "attached":
                hardware = await trial.agent_environment.exec(
                    "nvidia-smi --query-gpu=name,uuid --format=csv,noheader"
                )
                (trial.paths.trial_dir / "hardware.txt").write_text(hardware.stdout or "")
                if hardware.return_code or "H100" not in (hardware.stdout or ""):
                    raise RuntimeError("Attached reference requires an H100")
            environment = trial.agent_environment
            (trial.paths.trial_dir / "provider.json").write_text(
                json.dumps(
                    {
                        "agent_sandbox": environment._sandbox.object_id,
                        "agent_image": environment._image.object_id,
                        "gpu_worker_image": (
                            environment.gpu_session["limits"]["image_id"] if interface != "attached" else None
                        ),
                        "interface": interface,
                        "agent_runtime_requested": agent_runtime,
                        "gpu_protocol": "run-entrypoint-cpu-sync" if interface != "attached" else None,
                        "credential_delivery": "file" if interface != "attached" else None,
                    },
                    indent=2,
                )
                + "\n"
            )

        trial.add_hook(TrialEvent.AGENT_START, verify_access)
        (trial.paths.trial_dir / "effective-task.toml").write_text(definition.model_dump_toml())
        (trial.paths.trial_dir / "runner.py").write_text(Path(__file__).read_text())
        copy_sources(trial.paths.trial_dir / "runner-source")
        (trial.paths.trial_dir / "source.json").write_text((folder / "source.json").read_text())
        if model is not None:
            (trial.paths.trial_dir / "model-discovery.json").write_text(json.dumps(model, indent=2) + "\n")
        if study:
            (trial.paths.trial_dir / "study.json").write_text(json.dumps(study, indent=2) + "\n")
        print(json.dumps({"trial": str(trial.paths.trial_dir), "phase": "starting"}), flush=True)
        outcome = await trial.run()
    folder = trial.paths.trial_dir
    (folder / "cost-note.txt").write_text(
        "Endpoint price unknown; native zero cost fields are registration placeholders.\n"
    )
    for path in folder.rglob("*"):
        if path.is_file() and contains_secret(path, private_values):
            raise RuntimeError("Credential found in trial output; export stopped")
    await results.commit.aio()
    print(
        json.dumps(
            {
                "trial": str(folder),
                "exception": outcome.exception_info.exception_type if outcome.exception_info else None,
                "rewards": outcome.verifier_result.rewards if outcome.verifier_result else None,
            }
        ),
        flush=True,
    )
    archive = Path("/results/.exports") / (study["cohort"] if study else "") / f"{folder.name}.tar.gz"
    archive.parent.mkdir(parents=True, exist_ok=True)
    temporary = archive.with_suffix(".gz.partial")
    with tarfile.open(temporary, mode="w:gz") as output:
        output.add(folder, arcname=folder.name)
    temporary.replace(archive)
    await results.commit.aio()
    return {"trial": str(folder.relative_to("/results")), "archive": str(archive.relative_to("/results"))}


@app.function(
    image=image,
    volumes={"/results": results},
    cpu=1,
    memory=4096,
    timeout=10800,
    nonpreemptible=True,
)
async def check(
    task: str,
    gpu_image: str = "",
    interface: str = "remote",
    agent: str = "oracle",
    seconds: int = 900,
    *,
    run_id: str,
    agent_runtime: str | None = None,
):
    from benchmarks.admission import claim_once

    await claim_once("development/" + run_id)
    return await execute(task, gpu_image, interface, agent, seconds, agent_runtime=agent_runtime)


@app.local_entrypoint()
def main(
    task: str,
    gpu_image: str = "",
    interface: str = "remote",
    agent: str = "oracle",
    seconds: int = 900,
    verifier_secret: str = "",
    agent_secret: str = "",
    proxy_secret: str = "rl-gpu-harness-proxy",
    agent_runtime: str | None = None,
):
    if interface != "attached" and not gpu_image:
        from gpu_harness import APP_NAME

        folder = Path("artifacts/tasks") / task / "environment"
        gpu_image = (
            modal.Image.from_dockerfile(folder / "Dockerfile", context_dir=folder)
            .build(modal.App.lookup(APP_NAME))
            .object_id
        )
        print("GPU task image:", gpu_image)
    runner = check.with_options(secrets=operator_secrets(agent_secret, proxy_secret, verifier_secret))
    exported = runner.remote(
        task, gpu_image, interface, agent, seconds, run_id=uuid.uuid4().hex, agent_runtime=agent_runtime
    )
    print(json.dumps(exported), flush=True)
    with TemporaryFile() as downloaded:
        for chunk in results.read_file(exported["archive"]):
            downloaded.write(chunk)
        downloaded.seek(0)
        with tarfile.open(fileobj=downloaded, mode="r:gz") as archive:
            archive.extractall("artifacts/benchmark-results", filter="data")
    print("Actual Harbor outputs: artifacts/benchmark-results")
