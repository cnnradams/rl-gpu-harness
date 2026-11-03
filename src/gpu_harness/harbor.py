"""GPU access for upstream Harbor trials (Harbor 0.23.0)."""

import asyncio
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import modal
from harbor.environments.modal import ModalEnvironment

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
        self, *args, gpu_limits=None, gpu_proxy_secret=None, gpu_proxy_file=None, agent_runtime=None, **kwargs
    ):
        super().__init__(*args, **kwargs)
        if agent_runtime not in (None, "gvisor", "vm"):
            raise ValueError("agent_runtime must be gvisor or vm")
        self.agent_runtime = agent_runtime
        self.gpu_session = None
        self.gpu_limits = None
        self.gpu_proxy_secret = gpu_proxy_secret
        self.gpu_proxy_file = gpu_proxy_file
        self._gpu_agent = self.session_id == f"{self.trial_paths.trial_dir.name}__env"
        if self._gpu_agent:
            if self._compose_mode or self._vm_runtime_enabled:
                raise ValueError("GPUEnvironment requires a direct Linux Modal sandbox")
            if agent_runtime == "vm" and self._gpu_config() is not None:
                raise ValueError("The VM agent must be CPU-only; GPU boxes use the remote service")
            self.gpu_limits = Limits(**(gpu_limits or {}))
            if self.gpu_limits.shared_workspace:
                # Harbor supplies the task cwd to every exec, including the
                # parent shell of the agent. Sandbox.workdir alone is not enough.
                self.task_env_config = self.task_env_config.model_copy(update={"workdir": "/"})
            if bool(gpu_proxy_secret) == bool(gpu_proxy_file):
                raise ValueError(
                    "Set exactly one of gpu_proxy_secret (outbound injection) or "
                    "gpu_proxy_file (trusted runner JSON with Modal-Key and Modal-Secret)"
                )

    async def start(self, force_build: bool):
        try:
            if self._gpu_agent:
                if self.gpu_limits.shared_workspace:
                    # GPU Sandboxes reject mounting over a nonempty image path.
                    # Initial task files are already seeded by the CPU side.
                    image = await (
                        modal.Image.from_id(self.gpu_limits.image_id)
                        .run_commands("rm -rf /workspace && mkdir /workspace")
                        .build.aio(await modal.App.lookup.aio(APP_NAME))
                    )
                    self.gpu_limits = self.gpu_limits.model_copy(update={"image_id": image.object_id})
                self.gpu_session = await modal.Function.from_name(APP_NAME, "mint_session").remote.aio(
                    self.gpu_limits.model_dump(), str(self.context_id or self.session_id)
                )
                folder = self.trial_paths.trial_dir / "gpu-harness"
                folder.mkdir(parents=True, exist_ok=True)
                (folder / "instructions.md").write_text(instructions(self.gpu_limits))
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
        # Harbor 0.23 has no outbound-policy hook. Keep this native create call
        # aligned with its ModalEnvironment; all other lifecycle/exec is inherited.
        self._image = (
            self._image.pip_install("httpx==0.28.1")
            .add_local_file(client.__file__, "/opt/gpu_harness/client.py", copy=True)
            .run_commands(
                "printf '%s\\n' '#!/bin/sh' 'exec python3 /opt/gpu_harness/client.py \"$@\"' "
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
            ("gpu", self._gpu_config()),
            ("tags", self._sandbox_labels()),
            ("region", self._kwargs.get("region")),
        ]:
            if value is not None:
                kwargs[name] = value
        if block_network is None:
            block_network = self._network_disabled
        if self._dynamic_network:
            block_network = False
            kwargs.update(self._dynamic_network_kwargs(self.network_policy))
        elif self._network_is_allowlist:
            kwargs.update(self._allowlist_network_kwargs(self.network_policy))
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

    async def stop(self, delete: bool):
        try:
            await super().stop(delete)
        finally:
            if self.gpu_session:
                await close_and_export(self.gpu_session, self.trial_paths.trial_dir / "gpu-harness")
                self.gpu_session = None
