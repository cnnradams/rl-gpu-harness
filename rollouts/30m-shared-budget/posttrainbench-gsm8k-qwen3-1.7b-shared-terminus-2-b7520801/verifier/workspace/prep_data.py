import json
from datasets import load_dataset

PROMPT = """Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{q}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:"""

ds = load_dataset("openai/gsm8k", "main", split="train")
rows = []
for r in ds:
    ans = r["answer"]
    parts = ans.split("####")
    target = parts[-1].strip().replace(",", "")
    reasoning = "####".join(parts[:-1]).strip()
    completion = reasoning + "\n\nANSWER: " + target
    rows.append({"prompt": PROMPT.format(q=r["question"]), "completion": completion, "question": r["question"]})

with open("/workspace/train_data.jsonl", "w") as f:
    for row in rows:
        f.write(json.dumps(row) + "\n")
print("wrote", len(rows), "rows")
