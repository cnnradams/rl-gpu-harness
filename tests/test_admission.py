from types import SimpleNamespace

import pytest

from benchmarks import admission


async def test_repeated_controller_cannot_enter_trial_again(monkeypatch):
    started, keys = [], set()

    async def put(key, value, *, skip_if_exists):
        assert skip_if_exists
        if key in keys:
            return False
        keys.add(key)
        return True

    monkeypatch.setattr(
        admission.modal.Dict, "from_name", lambda *a, **kw: SimpleNamespace(put=SimpleNamespace(aio=put))
    )

    async def controller(key):
        await admission.claim_once(key)
        started.append(key)
        raise InterruptedError("controller preempted after starting its trial")

    with pytest.raises(InterruptedError):
        await controller("cohort/0")
    with pytest.raises(RuntimeError, match="refusing to start a replacement"):
        await controller("cohort/0")
    with pytest.raises(InterruptedError):
        await controller("cohort/1")
    assert started == ["cohort/0", "cohort/1"]
