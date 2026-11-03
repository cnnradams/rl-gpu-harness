import json, re, random
from datasets import load_dataset
from inspect_evals.gsm8k.gsm8k import MATH_PROMPT_TEMPLATE, record_to_sample, sample_to_fewshot
from inspect_ai.dataset import hf_dataset

fewshots = hf_dataset(path="openai/gsm8k", data_dir="main", split="train", sample_fields=record_to_sample, shuffle=True, seed=42, limit=10)
sys_msg = "\n\n".join([sample_to_fewshot(s) for s in fewshots])
print("SYSMSG_LEN", len(sys_msg))

ds = load_dataset("openai/gsm8k", "main", split="train")
rng = random.Random(0)
idxs = list(range(len(ds)))
rng.shuffle(idxs)
out = open("/workspace/train_data.jsonl", "w")
txt = open("/workspace/train_texts.txt", "w")
n = 0
for i in idxs[:3500]:
    rec = ds[i]
    q = rec["question"]
    parts = rec["answer"].split("####")
    target = parts[-1].strip()
    reasoning = "####".join(parts[:-1]).strip()
    reasoning = re.sub(r"<<[^>]*>>", "", reasoning)
    user = MATH_PROMPT_TEMPLATE.format(prompt=q)
    assistant = reasoning + "\n\nANSWER: " + target
    use_sys = rng.random() < 0.85
    out.write(json.dumps({"system": sys_msg if use_sys else None, "user": user, "assistant": assistant}) + "\n")
    txt.write((q + " " + rec["answer"]).replace("\n", " ") + "\n")
    n += 1
out.close(); txt.close()
print("WROTE", n)
