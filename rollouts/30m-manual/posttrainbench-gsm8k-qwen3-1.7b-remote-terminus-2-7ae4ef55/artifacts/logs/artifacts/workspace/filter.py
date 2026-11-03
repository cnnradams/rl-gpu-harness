import json
flagged = set()
for line in open("/workspace/contaminated.jsonl"):
    line = line.strip()
    if not line:
        continue
    try:
        obj = json.loads(line)
    except Exception:
        continue
    def walk(o):
        if isinstance(o, dict):
            for v in o.values(): walk(v)
        elif isinstance(o, list):
            for v in o: walk(v)
        elif isinstance(o, str):
            flagged.add(o)
    walk(obj)
qs = [l.rstrip("\n") for l in open("/workspace/train_questions.txt")]
bad = set()
for i, q in enumerate(qs):
    for fl in flagged:
        if fl and (fl in q or q in fl):
            bad.add(i)
            break
rows = [json.loads(l) for l in open("/workspace/train_raw.jsonl")]
clean = [r for i, r in enumerate(rows) if i not in bad]
with open("/workspace/train_clean.jsonl","w") as fo:
    for r in clean:
        fo.write(json.dumps(r)+"\n")
print("removed", len(bad), "kept", len(clean))
