import json
import subprocess
from pathlib import Path

gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True).strip()
report = {"gpu": gpu, "sum": sum(range(1000))}
Path("/tmp/report.json").write_text(json.dumps(report))
print(json.dumps(report))
