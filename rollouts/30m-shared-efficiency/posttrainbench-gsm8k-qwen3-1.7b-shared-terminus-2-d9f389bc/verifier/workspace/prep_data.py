import json, random
from datasets import load_dataset

ds = load_dataset("openai/gsm8k", "main", split="train")
shuf = ds.shuffle(seed=42)
fewshot_samples = [shuf[i] for i in range(10)]

def fmt_fewshot(rec):
    reasoning, ans = rec["answer"].split("####")
    return f"{rec['question']}\n\nReasoning:\n{reasoning.strip()}\n\nANSWER: {ans.strip()}"

system_block = "\n\n".join(fmt_fewshot(r) for r in fewshot_samples)

PROMPT = """Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{q}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:"""

rng = random.Random(0)
out = []
for rec in ds:
    reasoning, ans = rec["answer"].split("####")
    reasoning = reasoning.strip()
    ans = ans.strip()
    if rng.random() < 0.25:
        prompt = f"<|im_start|>system\n{system_block}<|im_end|>\n<|im_start|>user\n{PROMPT.format(q=rec['question'])}<|im_end|>\n<|im_start|>assistant\n"
    else:
        prompt = f"<|im_start|>user\n{PROMPT.format(q=rec['question'])}<|im_end|>\n<|im_start|>assistant\n"
    completion = f"{reasoning}\n\nANSWER: {ans}<|im_end|>"
    out.append({"prompt": prompt, "completion": completion})

with open("/workspace/train_data.jsonl", "w") as f:
    for o in out:
        f.write(json.dumps(o) + "\n")
print("n examples:", len(out))
