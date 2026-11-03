import json, re
from inspect_evals.gsm8k.gsm8k import record_to_sample, sample_to_fewshot, MATH_PROMPT_TEMPLATE
from inspect_ai.dataset import hf_dataset

fewshots = hf_dataset(path="openai/gsm8k", data_dir="main", split="train",
                      sample_fields=record_to_sample, shuffle=True, seed=42, limit=10)
sys_msg = "\n\n".join(sample_to_fewshot(s) for s in fewshots)
print("SYS MSG:", sys_msg[:500])

from datasets import load_dataset
ds = load_dataset("openai/gsm8k", "main", split="train")

def clean(r):
    return re.sub(r"<<[^>]*>>", "", r)

examples = []
texts = []
for i, rec in enumerate(ds):
    parts = rec["answer"].split("####")
    target = parts[-1].strip()
    reasoning = clean("####".join(parts[:-1])).strip()
    user = MATH_PROMPT_TEMPLATE.format(prompt=rec["question"])
    completion = reasoning + "\n\nANSWER: " + target
    examples.append({"idx": i, "system": sys_msg, "user": user, "completion": completion})
    texts.append((rec["question"] + " " + reasoning + " " + target).replace("\n", " "))

json.dump(examples, open("/workspace/sft_data.json", "w"))
with open("/workspace/train_texts.txt", "w") as f:
    for t in texts:
        f.write(t + "\n")
print("num examples:", len(examples))
