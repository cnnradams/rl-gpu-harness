import json, re
from datasets import load_dataset

MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

try:
    from inspect_ai.dataset import hf_dataset
    from inspect_evals.gsm8k.gsm8k import record_to_sample, sample_to_fewshot
    fewshots = hf_dataset(path="openai/gsm8k", data_dir="main", split="train", sample_fields=record_to_sample, shuffle=True, seed=42, limit=10)
    system_msg = "\n\n".join([sample_to_fewshot(s) for s in fewshots])
    print("built system message via inspect")
except Exception as e:
    print("fallback:", e)
    d = load_dataset("openai/gsm8k", "main", split="train").shuffle(seed=42)
    parts = []
    for i in range(10):
        r = d[i]
        a = r["answer"].split("####")
        tgt = a[-1].strip()
        reas = "####".join(a[:-1]).strip()
        parts.append(r["question"] + "\n\nReasoning:\n" + reas + "\n\nANSWER: " + tgt)
    system_msg = "\n\n".join(parts)

print(system_msg[:300])

ds = load_dataset("openai/gsm8k", "main", split="train")

def clean(ans):
    parts = ans.split("####")
    target = parts[-1].strip()
    reasoning = "####".join(parts[:-1]).strip()
    reasoning = re.sub(r"<<[^>]*>>", "", reasoning)
    return reasoning, target

n = 0
with open("/workspace/train_data.jsonl", "w") as f:
    for rec in ds:
        reasoning, target = clean(rec["answer"])
        user = MATH_PROMPT_TEMPLATE.replace("{prompt}", rec["question"])
        assistant = reasoning + "\n\nANSWER: " + target
        f.write(json.dumps({"system": system_msg, "user": user, "assistant": assistant}) + "\n")
        n += 1
print("wrote", n)
