import json
import os
import shutil
from pathlib import Path

report = json.loads(Path("/workspace/report.json").read_text())
assert report["sum"] == 499500
assert report["gpu"]
assert not any(name.startswith("GPU_HARNESS_") for name in os.environ)
assert shutil.which("gpu") is None
Path("/logs/verifier/reward.json").write_text(json.dumps({"reward": 1}))
print(json.dumps({"report": report, "verifier_has_no_gpu_session_or_client": True}))
