"""GPU access for upstream Harbor trials (Harbor 0.23.0)."""

import asyncio
import json
import os
import posixpath
import uuid
from pathlib import Path, PurePosixPath
from tempfile import NamedTemporaryFile
from urllib.parse import urlsplit

import modal
from harbor.environments.modal import ModalEnvironment
from harbor.models.task.config import EnvironmentConfig, TaskConfig
from harbor.models.task.verifier_mode import task_has_any_shared_verifier
from harbor.verifier.verifier import Verifier

from . import APP_NAME, client
from .lifecycle import close_and_export
from .service import Limits
from .session import credential_policy

INSTRUCTIONS = """# GPU tools

`gpu` provides HTTP access to remote GPU boxes. The local environment and each
GPU box have separate filesystems. `gpu push` and `gpu pull` transfer files;
`gpu run -- PROGRAM [ARG ...]` starts a GPU box with its first command;
the box stops when that command exits. `gpu exec BOX -- PROGRAM [ARG ...]`
runs additional commands while the box is alive.
Run and exec wait unless `--detach` is set.
`gpu --help` lists the interface.
`gpu policy` reports this session's resources, limits and allocated GPU time.
GPU-time accounting starts after readiness, includes idle allocation, and sums
across GPUs. Commands can run concurrently on a box. A stopped or failed box
loses its processes and local files; work is not automatically restored.
"""

SHARED_INSTRUCTIONS = """# GPU tools

`gpu` provides HTTP access to remote GPU boxes. The agent and GPU boxes mount
the same session-specific Volume at `/workspace`.
`gpu run -- PROGRAM [ARG ...]` starts a GPU box with its first command;
the box stops when that command exits. `gpu exec BOX -- PROGRAM [ARG ...]`
runs additional commands while the box is alive.
Run and exec wait unless `--detach` is set. A new box mounts the latest
committed files. Modal commits GPU writes in the background and on shutdown.
`gpu sync` commits and reloads the agent's CPU-side Volume view; it does not
allocate a GPU or reload a running GPU box. It can run before any GPU exists
and after a GPU box stops. `gpu refresh BOX` reloads a running GPU box's Volume
view. Reload fails while any process holds open files or a working directory
on that Volume. GPU commands default to `/workspace`; the CPU agent starts in `/`.
Sync reports completed steps on failure without stopping processes.
Concurrent same-file edits use last-writer-wins
semantics. Paths outside `/workspace` are separate and GPU-local files there
are lost when the box stops.
`gpu --help` lists the interface; `gpu policy` reports resources and limits.
GPU time starts at readiness, includes idle allocation and sums across GPUs.
Commands can run concurrently. A failed box loses its running processes;
work is not automatically restarted.
"""


RUN_ONLY_INSTRUCTIONS = """# GPU tools

`gpu run -- PROGRAM [ARG ...]` starts a GPU box for that command; the box stops
when it exits. Run waits unless `--detach` is set.
GPU programs start only through `gpu run`. `gpu stop BOX` stops a box early;
`gpu status COMMAND` and `gpu wait COMMAND` read command output and state.
CPU `/workspace` writes are committed before each run. The new GPU box mounts
that workspace; GPU writes commit in the background and on shutdown.
`gpu sync` commits and reloads the CPU workspace view. Reload requires no open
files or working directories on the Volume. GPU-local paths outside `/workspace`
are lost when the box stops. `gpu --help` lists commands; `gpu policy` lists limits.
"""


def instructions(limits):
    text = (
        RUN_ONLY_INSTRUCTIONS
        if limits.run_only
        else (SHARED_INSTRUCTIONS if limits.shared_workspace else INSTRUCTIONS)
    )
    if limits.gpu_seconds is not None:
        text += f"\nThis session has an enforced budget of {limits.gpu_seconds:g} allocated GPU seconds.\n"
    if limits.idle_seconds is not None:
        if limits.idle_probe == "cuda":
            text += (
                f"\nThe {limits.idle_seconds}-second CUDA-idle timer resets when a CUDA-attached process "
                "is observed. "
                "Expiry stops the box. CUDA-process checks are sampled about every 10 seconds.\n"
            )
            text += "A failed GPU-process probe does not trigger idle release.\n"
        else:
            text += (
                f"\nBoxes stop after {limits.idle_seconds} consecutive seconds "
                "without a running tool command.\n"
            )
    return text


class GPUEnvironment(ModalEnvironment):
    """A CPU agent environment with separately configured HTTP GPU tools.

    Harbor owns trial creation, agent execution, verification and trajectories.
    Its documented session naming identifies the separate verifier, which uses
    the ordinary Modal environment without a GPU-tool session or credentials.
    """

    def __init__(
        self,
        *args,
        gpu_limits=None,
        gpu_proxy_secret=None,
        gpu_proxy_file=None,
        agent_runtime=None,
        mode=None,
        gpu_seconds=None,
        cuda_idle_seconds=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if agent_runtime not in (None, "gvisor", "vm"):
            raise ValueError("agent_runtime must be gvisor or vm")
        self.agent_runtime = agent_runtime
        self._gpu_session_closed = False
        self._snapshot_verifier = False
        self._task_environment = self.task_env_config.model_copy(deep=True)
        self._task_secret = None
        self._artifact_reader = None
        self.gpu_session = None
        self.gpu_limits = None
        self.gpu_proxy_secret = gpu_proxy_secret
        self.gpu_proxy_file = gpu_proxy_file
        self._gpu_agent = (
            self.session_id == f"{self.trial_paths.trial_dir.name}__env" and self._effective_gpus > 0
        )
        if self._gpu_agent:
            definition = self.environment_dir.parent / "task.toml"
            task = TaskConfig.model_validate_toml(definition.read_text())
            self._snapshot_verifier = task_has_any_shared_verifier(task)
            if self._snapshot_verifier and task.steps:
                raise ValueError(
                    "Snapshot verification supports single-step tasks; "
                    "use separate verifiers for multi-step tasks"
                )
            if self._compose_mode or self._vm_runtime_enabled:
                raise ValueError("GPUEnvironment requires a direct Linux Modal sandbox")
            options = dict(gpu_limits or {})
            # Explicit legacy flags remain supported; new callers default to run-only.
            if mode is None:
                if options.get("run_only"):
                    mode = "run-only"
                elif options.get("shared_workspace"):
                    mode = "shared"
                elif "shared_workspace" in options or "run_only" in options:
                    mode = "manual"
                else:
                    mode = "run-only"
            modes = {"run-only": (True, True), "shared": (True, False), "manual": (False, False)}
            if mode not in modes:
                raise ValueError("mode must be run-only, shared or manual")
            shared, run_only = modes[mode]
            for key, value in (("shared_workspace", shared), ("run_only", run_only)):
                if key in options and options[key] != value:
                    raise ValueError(f"{key} conflicts with mode={mode}")
                options[key] = value
            defaults = {
                "gpu": (self.task_env_config.gpu_types or ["any"])[0],
                "gpus_per_box": self._effective_gpus or 1,
                "cpu": self._effective_cpus or 4,
                "memory": self._effective_memory_mb or 16384,
                "image_id": "",  # Resolved from Harbor's task image before minting.
            }
            if gpu_seconds is not None:
                options["gpu_seconds"] = gpu_seconds
            if cuda_idle_seconds is not None:
                options.update(idle_seconds=cuda_idle_seconds, idle_probe="cuda")
            defaults.update(ModalEnvironment._dynamic_network_kwargs(self.network_policy))
            self.gpu_limits = Limits(**(defaults | options))
            if self.gpu_limits.shared_workspace:
                # Harbor supplies the task cwd to every exec, including the
                # parent shell of the agent. Sandbox.workdir alone is not enough.
                self.task_env_config = self.task_env_config.model_copy(update={"workdir": "/"})
            if bool(gpu_proxy_secret) == bool(gpu_proxy_file):
                raise ValueError(
                    "Set exactly one of gpu_proxy_secret (outbound injection) or "
                    "gpu_proxy_file (trusted runner JSON with Modal-Key and Modal-Secret)"
                )
            if gpu_proxy_secret and any(
                policy.network_mode != "public"
                for policy in [self.network_policy, *self._phase_network_policies]
            ):
                raise ValueError(
                    "Modal credential injection cannot be combined with a network allowlist; "
                    "use gpu_proxy_file"
                )

    async def _mint_session(self, image):
        if self._startup_env():
            name = f"gpu-harness-env-{uuid.uuid4().hex}"
            await modal.Secret.objects.create.aio(name, self._startup_env())
            self._task_secret = name
            self.gpu_limits = self.gpu_limits.model_copy(update={"environment_secret_name": name})
        if self.gpu_limits.shared_workspace:
            image = image.run_commands("rm -rf /workspace && mkdir /workspace")
        if self.gpu_limits.shared_workspace or not self.gpu_limits.image_id:
            image = await image.build.aio(await modal.App.lookup.aio(APP_NAME))
            self.gpu_limits = self.gpu_limits.model_copy(update={"image_id": image.object_id})
        self.gpu_session = await modal.Function.from_name(APP_NAME, "mint_session").remote.aio(
            self.gpu_limits.model_dump(), str(self.context_id or self.session_id)
        )
        folder = self.trial_paths.trial_dir / "gpu-harness"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "instructions.md").write_text(instructions(self.gpu_limits))

    async def start(self, force_build: bool):
        try:
            if self._gpu_agent:
                if self._snapshot_verifier:
                    config = json.loads(self.trial_paths.config_path.read_text())
                    verifier = config.get("verifier", {})
                    if (
                        not verifier.get("disable")
                        and verifier.get("import_path") != "gpu_harness.harbor:GPUVerifier"
                    ):
                        raise ValueError(
                            "Shared-verifier tasks require --verifier gpu_harness.harbor:GPUVerifier"
                        )
                if self.gpu_limits.image_id:
                    await self._mint_session(modal.Image.from_id(self.gpu_limits.image_id))
            await super().start(force_build)
            if self._gpu_agent and (self.gpu_proxy_file or self.gpu_limits.run_only):
                proxy = (
                    json.loads(Path(self.gpu_proxy_file).read_text())
                    if self.gpu_proxy_file
                    else {"Modal-Key": "dummy", "Modal-Secret": "dummy"}
                )
                credentials = {
                    "url": self.gpu_session["url"],
                    "token": self.gpu_session["token"] if self.gpu_proxy_file else "dummy",
                    "proxy": {key: proxy[key] for key in ("Modal-Key", "Modal-Secret")},
                }
                if self.gpu_limits.run_only:
                    credentials["run_only"] = True
                result = await self.exec("mkdir -p /run/secrets && chmod 700 /run/secrets", user="root")
                if result.return_code:
                    raise RuntimeError("Could not create GPU credential directory")
                with NamedTemporaryFile(mode="w+") as config:
                    json.dump(credentials, config)
                    config.flush()
                    await self.upload_file(config.name, "/run/secrets/gpu-harness.json")
                result = await self.exec("chmod 600 /run/secrets/gpu-harness.json", user="root")
                if result.return_code:
                    raise RuntimeError("Could not set GPU credential file permissions")
        except BaseException:
            # Harbor also calls stop on failure. Do not leave a minted session
            # active if setup failed before it reached agent execution.
            try:
                await asyncio.shield(self.stop(delete=True))
            except Exception:
                self.logger.exception("GPU environment cleanup after setup failure failed")
            raise

    async def _create_sandbox(self, *, entrypoint=None, block_network=None, experimental_options=None):
        if not self._gpu_agent:
            return await super()._create_sandbox(
                entrypoint=entrypoint, block_network=block_network, experimental_options=experimental_options
            )
        if self.gpu_session is None:
            await self._mint_session(self._image)
        # Harbor 0.23 has no outbound-policy hook. Keep this native create call
        # aligned with its ModalEnvironment; all other lifecycle/exec is inherited.
        self._image = (
            self._image.add_local_file(
                Path(__file__).with_name("install_client.sh"), "/opt/gpu-harness/install_client.sh", copy=True
            )
            .run_commands("sh /opt/gpu-harness/install_client.sh")
            .add_local_file(client.__file__, "/opt/gpu_harness/client.py", copy=True)
            .add_local_file(
                self.trial_paths.trial_dir / "gpu-harness/instructions.md", "/etc/gpu-harness.md", copy=True
            )
            .run_commands(
                "printf '%s\\n' '#!/bin/sh' "
                "'exec /opt/gpu-harness/venv/bin/python /opt/gpu_harness/client.py \"$@\"' "
                "> /usr/local/bin/gpu && chmod 755 /usr/local/bin/gpu"
            )
        )
        if self.gpu_limits.shared_workspace:
            if "/workspace" in self._volumes:
                raise ValueError("Shared GPU workspace supplies its own session Volume")
            # Mounting a Volume hides the image directory. Preserve the initial
            # task files in the image and seed the new Volume before healthchecks.
            self._image = self._image.run_commands(
                "mkdir -p /workspace && mv /workspace /opt/gpu-harness-seed && mkdir /workspace"
            )
            self._volumes["/workspace"] = self.gpu_session["workspace_volume"]
        kwargs = {"experimental_options": experimental_options} if experimental_options else {}
        if self.agent_runtime is not None:
            kwargs["runtime"] = self.agent_runtime
        if self.gpu_limits.shared_workspace:
            # Keep the sandbox's idle process outside the reloadable mount.
            kwargs["workdir"] = "/"
        for name, value in [
            ("cpu", self._cpu_config()),
            ("memory", self._memory_config()),
            ("tags", self._sandbox_labels()),
            ("region", self._kwargs.get("region")),
        ]:
            if value is not None:
                kwargs[name] = value
        # The tool service is part of the harness transport, not task Internet access.
        # Preserve the task's other network restrictions on both CPU and GPU boxes.
        block_network = False
        if not self.gpu_proxy_secret:
            kwargs.update(self._dynamic_network_kwargs(self.network_policy))
        elif self.network_policy.network_mode != "public":
            raise ValueError(
                "Modal credential injection cannot be combined with a network allowlist; use gpu_proxy_file"
            )
        os.environ["MODAL_SANDBOX_V2"] = "1"
        sandbox = await modal.Sandbox.create.aio(
            *(entrypoint or ()),
            app=self._app,
            image=self._image,
            name=self.session_id,
            timeout=self._sandbox_timeout,
            idle_timeout=self._sandbox_idle_timeout,
            block_network=block_network,
            secrets=self._secrets_config(),
            volumes=self._volumes_config(),
            env={
                **self._startup_env(),
                "GPU_HARNESS_URL": self.gpu_session["url"],
                "GPU_HARNESS_TOKEN": "dummy",
                "GPU_HARNESS_PROXY_KEY": "dummy",
                "GPU_HARNESS_PROXY_SECRET": "dummy",
            },
            _experimental_outbound_policy=(
                credential_policy(self.gpu_session, self.gpu_proxy_secret) if self.gpu_proxy_secret else None
            ),
            **kwargs,
        )
        if self.gpu_limits.shared_workspace:
            try:
                process = await sandbox.exec.aio(
                    "sh", "-c", "cp -a /opt/gpu-harness-seed/. /workspace/ && sync /workspace"
                )
                if await process.wait.aio() != 0:
                    raise RuntimeError("Could not seed the shared task workspace")
                await modal.Function.from_name(APP_NAME, "bind_agent").remote.aio(
                    self.gpu_session["token"], sandbox.object_id
                )
            except BaseException:
                await sandbox.terminate.aio()
                raise
        self._sandbox_termination_requested = False
        return sandbox

    def _dynamic_network_kwargs(self, network_policy):
        kwargs = ModalEnvironment._dynamic_network_kwargs(network_policy)
        if getattr(self, "_gpu_agent", False) and self.gpu_session:
            host = urlsplit(self.gpu_session["url"]).hostname
            domains = kwargs["outbound_domain_allowlist"]
            if "*" not in domains and host not in domains:
                kwargs["outbound_domain_allowlist"] = [*domains, host]
        return kwargs

    async def snapshot_for_verification(self):
        """Move disk state into a GPU verifier without migrating running processes."""
        await close_and_export(self.gpu_session, self.trial_paths.trial_dir / "gpu-harness")
        self._gpu_session_closed = True
        # Never bake operator proxy credentials into the verifier snapshot.
        process = await self._sandbox.exec.aio("sh", "-c", "rm -f /run/secrets/gpu-harness.json; sync")
        if await process.wait.aio() != 0:
            raise RuntimeError("Could not flush agent files before verifier snapshot")
        image = await self._sandbox.snapshot_filesystem.aio(ttl=86400)
        target = ModalEnvironment(
            environment_dir=self.environment_dir,
            environment_name=self.environment_name,
            session_id=f"{self.session_id}-verifier",
            trial_paths=self.trial_paths,
            task_env_config=self._task_environment.model_copy(deep=True),
            persistent_env=self._startup_env(),
            network_policy=self.network_policy,
        )
        target.default_user = self.default_user
        resources = {
            name: value
            for name, value in (
                ("gpu", self._gpu_config()),
                ("cpu", self._cpu_config()),
                ("memory", self._memory_config()),
                ("region", self._kwargs.get("region")),
            )
            if value is not None
        }
        target._sandbox = await modal.Sandbox.create.aio(
            "sleep",
            "infinity",
            app=self._app,
            image=image,
            timeout=self._sandbox_timeout,
            workdir=self._task_environment.workdir,
            volumes=self._volumes_config(),
            secrets=self._secrets_config(),
            env=self._startup_env(),
            **ModalEnvironment._dynamic_network_kwargs(self.network_policy),
            **resources,
        )
        (self.trial_paths.trial_dir / "gpu-harness/verifier.json").write_text(
            json.dumps(
                {
                    "mode": "filesystem-snapshot",
                    "sandbox": target._sandbox.object_id,
                    "image": image.object_id,
                    "gpu": resources.get("gpu"),
                    "workspace_volume": self.gpu_session.get("workspace_volume"),
                },
                indent=2,
            )
        )
        return target

    async def _submission_source(self, path):
        """Read shared submissions from a fresh Volume view, not the agent's stale mount."""
        if not (
            self._gpu_agent
            and self.gpu_limits.shared_workspace
            and PurePosixPath(posixpath.normpath(path)).is_relative_to("/workspace")
        ):
            return None
        # Commit CPU edits without reloading its mount or disturbing open shells.
        process = await self._sandbox.exec.aio("sync", "/workspace")
        if await process.wait.aio() != 0:
            raise RuntimeError("Could not commit the CPU workspace for artifact collection")
        if self._artifact_reader is None:
            reader = ModalEnvironment(
                environment_dir=self.environment_dir,
                environment_name=self.environment_name,
                session_id=f"{self.session_id}-artifacts",
                trial_paths=self.trial_paths,
                task_env_config=EnvironmentConfig(gpus=0, workdir="/"),
            )
            reader._sandbox = await modal.Sandbox.create.aio(
                "sleep",
                "infinity",
                app=self._app,
                image=modal.Image.debian_slim(),
                cpu=0.25,
                memory=512,
                timeout=self._sandbox_timeout,
                workdir="/",
                block_network=True,
                volumes={
                    "/workspace": modal.Volume.from_name(
                        self.gpu_session["workspace_volume"]
                    ).with_mount_options(read_only=True)
                },
            )
            self._artifact_reader = reader
        else:
            await self._artifact_reader._sandbox.reload_volumes.aio()
        return self._artifact_reader

    async def is_dir(self, path, user=None):
        source = await self._submission_source(path)
        return await (source or super()).is_dir(path, user=user)

    async def download_file(self, source_path, target_path):
        source = await self._submission_source(source_path)
        return await (source or super()).download_file(source_path, target_path)

    async def download_dir(self, source_dir, target_dir):
        source = await self._submission_source(source_dir)
        return await (source or super()).download_dir(source_dir, target_dir)

    async def download_dir_with_exclusions(self, *, source_dir, target_dir, exclude):
        source = await self._submission_source(source_dir)
        return await (source or super()).download_dir_with_exclusions(
            source_dir=source_dir, target_dir=target_dir, exclude=exclude
        )

    async def stop(self, delete: bool):
        try:
            try:
                if self._artifact_reader is not None:
                    await self._artifact_reader.stop(delete=True)
                    self._artifact_reader = None
            finally:
                await super().stop(delete)
        finally:
            try:
                if self.gpu_session:
                    if not self._gpu_session_closed:
                        await close_and_export(self.gpu_session, self.trial_paths.trial_dir / "gpu-harness")
                    self.gpu_session = None
            finally:
                if self._task_secret:
                    await modal.Secret.objects.delete.aio(self._task_secret, allow_missing=True)
                    self._task_secret = None


class GPUVerifier(Verifier):
    """Keep native grading; supply a filesystem-snapshot GPU for shared verifiers."""

    async def verify(self):
        agent_environment = self.environment
        if not isinstance(agent_environment, GPUEnvironment) or not agent_environment._gpu_agent:
            return await super().verify()
        target = await agent_environment.snapshot_for_verification()
        self.environment = target
        try:
            return await super().verify()
        finally:
            self.environment = agent_environment
            await target.stop(delete=True)
