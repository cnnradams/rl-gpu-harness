import json
from datasets import load_dataset
ds = load_dataset("openai/gsm8k", "main")["train"]
with open("/workspace/train_questions.jsonl", "w") as f:
    for ex in ds:
        f.write(json.dumps({"text": ex["question"]}) + "\n")
print(len(ds))
