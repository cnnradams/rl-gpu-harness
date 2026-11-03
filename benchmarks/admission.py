"""Prevent a platform-retried controller from starting another rollout."""

import modal


async def claim_once(key):
    claims = modal.Dict.from_name("gpu-harness-rollout-claims", create_if_missing=True)
    if not await claims.put.aio(key, True, skip_if_exists=True):
        raise RuntimeError(f"Rollout {key} already started; refusing to start a replacement trial")
