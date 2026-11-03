import json, random
from datasets import load_dataset
from inspect_ai.dataset import hf_dataset
from inspect_evals.gsm8k.gsm8k import record_to_sample, sample_to_fewshot, MATH_PROMPT_TEMPLATE

ds = load_dataset("openai/gsm8k", "main", split="train")
print("train size", len(ds))

fewshots = hf_dataset(path="openai/gsm8k", data_dir="main", split="train", sample_fields=record_to_sample, shuffle=True, seed=42, limit=10)
system_msg = "\n\n".join([sample_to_fewshot(s) for s in fewshots])
print("SYSTEM MSG HEAD:", system_msg[:300])

def build(rec):
    q = rec["question"]
    parts = rec["answer"].split("####")
    target = parts[-1].strip()
    reasoning = "####".join(parts[:-1]).strip()
    user = MATH_PROMPT_TEMPLATE.format(prompt=q)
    completion = reasoning + "\n\nANSWER: " + target
    return q, user, completion

out = []
with open("/workspace/questions.txt", "w") as qf:
    for rec in ds:
        q, user, completion = build(rec)
        qf.write(q.replace("\n", " ") + "\n")
        out.append({"messages": [{"role": "user", "content": user}, {"role": "assistant", "content": completion}]})

idx = list(range(len(ds)))
random.Random(0).shuffle(idx)
for i in idx[:1500]:
    rec = ds[i]
    q, user, completion = build(rec)
    out.append({"messages": [{"role": "system", "content": system_msg}, {"role": "user", "content": user}, {"role": "assistant", "content": completion}]})

random.Random(1).shuffle(out)
with open("/workspace/train.jsonl", "w") as f:
    for r in out:
        f.write(json.dumps(r) + "\n")
print("wrote", len(out))
