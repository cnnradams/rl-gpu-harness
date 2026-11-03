import json
from datasets import load_dataset
from inspect_ai.dataset import hf_dataset
from inspect_evals.gsm8k.gsm8k import sample_to_fewshot, record_to_sample, DATASET_PATH, MATH_PROMPT_TEMPLATE

fewshots = hf_dataset(path=DATASET_PATH, data_dir="main", split="train", sample_fields=record_to_sample, shuffle=True, seed=42, limit=10)
sys_msg = "\n\n".join([sample_to_fewshot(s) for s in fewshots])

ds = load_dataset("openai/gsm8k", "main", split="train")
n = 0
with open("/workspace/train_raw.jsonl","w") as f, open("/workspace/train_questions.txt","w") as fq:
    for rec in ds:
        q = rec["question"].strip()
        parts = rec["answer"].split("####")
        reasoning = "####".join(parts[:-1]).strip()
        target = parts[-1].strip()
        user = MATH_PROMPT_TEMPLATE.format(prompt=q)
        assistant = reasoning + "\n\nANSWER: " + target
        f.write(json.dumps({"system": sys_msg, "user": user, "assistant": assistant, "question": q})+"\n")
        fq.write(q.replace("\n"," ")+"\n")
        n += 1
print("done", n)
