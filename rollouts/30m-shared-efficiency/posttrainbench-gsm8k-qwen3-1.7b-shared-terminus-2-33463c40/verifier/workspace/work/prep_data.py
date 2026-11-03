import json, re
from datasets import load_dataset

MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

ds = load_dataset("openai/gsm8k", "main", split="train")
print("train size", len(ds))

def clean_reasoning(ans):
    parts = ans.split("####")
    target = parts[-1].strip()
    reasoning = "####".join(parts[:-1]).strip()
    # remove calculator annotations <<...>>
    reasoning = re.sub(r"<<[^>]*>>", "", reasoning)
    return reasoning, target

out = []
for rec in ds:
    q = rec["question"]
    reasoning, target = clean_reasoning(rec["answer"])
    user = MATH_PROMPT_TEMPLATE.format(prompt=q)
    assistant = f"{reasoning}\n\nANSWER: {target}"
    out.append({"messages": [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]})

with open("/workspace/work/sft_data.jsonl", "w") as f:
    for r in out:
        f.write(json.dumps(r) + "\n")
print("wrote", len(out))
print(out[0]["messages"][0]["content"][:300])
print("---")
print(out[0]["messages"][1]["content"][:300])
