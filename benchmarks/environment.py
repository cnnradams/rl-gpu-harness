"""Pin study base images while retaining upstream Harbor environment behavior."""

import json
import time

import modal
from harbor.environments.modal import ModalEnvironment

from gpu_harness.harbor import GPUEnvironment


class StudyImages:
    def __init__(self, *args, image_ids, **kwargs):
        super().__init__(*args, **kwargs)
        if self._compose_mode or self._vm_runtime_enabled:
            raise ValueError("Study images require direct Modal Sandboxes")
        self._study_images = image_ids
        self._study_runtime = None
        self._study_first_exec = None

    async def start(self, force_build: bool):
        await super().start(force_build)
        self._write_study_runtime()

    def _write_study_runtime(self):
        if self._study_runtime is None:
            return
        role = self._study_runtime["role"]
        path = self.trial_paths.trial_dir / f"{role}-image.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self._study_runtime, indent=2) + "\n")
        temporary.replace(path)

    async def _create_sandbox(self, **kwargs):
        role = "agent" if self.session_id == f"{self.trial_paths.trial_dir.name}__env" else "verifier"
        base = self._study_images[role]
        self._image = modal.Image.from_id(base)
        requested_at = time.time()
        sandbox = await super()._create_sandbox(**kwargs)
        self._study_runtime = {
            "role": role,
            "base_image": base,
            "runtime_image": self._image.object_id,
            "sandbox": sandbox.object_id,
            "requested_at": requested_at,
        }
        return sandbox

    async def _sdk_exec(self, *args, **kwargs):
        result = await super()._sdk_exec(*args, **kwargs)
        if self._study_first_exec is None:
            self._study_first_exec = time.time()
        return result

    async def stop(self, delete):
        requested_at = time.time()
        completed = False
        try:
            await super().stop(delete)
            completed = True
        finally:
            if self._study_runtime:
                if "stop_requested_at" not in self._study_runtime:
                    self._study_runtime.update(
                        first_exec_returned_at=self._study_first_exec,
                        stop_requested_at=requested_at,
                        stop_finished_at=time.time(),
                        stop_completed=completed,
                    )
                self._write_study_runtime()


class StudyGPUEnvironment(StudyImages, GPUEnvironment):
    pass


class StudyAttachedEnvironment(StudyImages, ModalEnvironment):
    pass
