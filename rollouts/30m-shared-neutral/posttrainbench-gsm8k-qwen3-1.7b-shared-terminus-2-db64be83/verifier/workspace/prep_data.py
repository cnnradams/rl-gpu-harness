import re, json
from datasets import load_dataset
from inspect_ai.dataset import hf_dataset
from inspect_evals.gsm8k.gsm8k import record_to_sample, sample_to_fewshot

fewshots = hf_dataset(path="openai/gsm8k", data_dir="main", split="train",
    sample_fields=record_to_sample, shuffle=True, seed=42, limit=10)
sys_prompt = "\n\n".join(sample_to_fewshot(s) for s in fewshots)

PROMPT = '''Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:'''

ds = load_dataset("openai/gsm8k", "main", split="train")
rows = []
check = []
for i, r in enumerate(ds):
    q = r["question"].strip()
    ans = r["answer"]
    parts = ans.split("####")
    target = parts[-1].strip()
    reasoning = "####".join(parts[:-1]).strip()
    reasoning = re.sub(r"<<[^>]*>>", "", reasoning)
    completion = f"{reasoning}\n\nANSWER: {target}"
    user = PROMPT.format(prompt=q)
    text = ""
    if i % 4 == 0:
        text += f"<|im_start|>system\n{sys_prompt}<|im_end|>\n"
    text += f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n{completion}<|im_end|>\n"
    rows.append({"text": text})
    check.append(q + "\n" + reasoning + "\n" + target)
with open("/workspace/train.jsonl", "w") as f:
    for row in rows:
        f.write(json.dumps(row) + "\n")
with open("/workspace/train_check.txt", "w") as f:
    for c in check:
        f.write(c.replace("\n", " ") + "\n")
print("num rows:", len(rows))
print(rows[1]["text"][:600])
