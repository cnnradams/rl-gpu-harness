#!/usr/bin/env python3
import json, sys, re
inp, flagged, out = sys.argv[1], sys.argv[2], sys.argv[3]
bad = set()
try:
    with open(flagged) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            m = re.match(r"line:(\d+)", obj.get("source", ""))
            if m:
                bad.add(int(m.group(1)))
except FileNotFoundError:
    pass
kept = 0
with open(inp) as f, open(out, "w") as g:
    for i, line in enumerate(f, start=1):
        if i in bad:
            continue
        g.write(line)
        kept += 1
print(f"removed {len(bad)} flagged docs, kept {kept}")
