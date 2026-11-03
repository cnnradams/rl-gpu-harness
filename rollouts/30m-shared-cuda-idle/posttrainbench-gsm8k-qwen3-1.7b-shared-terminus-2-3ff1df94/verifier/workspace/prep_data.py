import json, re
from datasets import load_dataset

TEMPLATE = '''Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{question}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.'''

ds = load_dataset("openai/gsm8k", "main", split="train")
out = []
for ex in ds:
    q = ex["question"].strip()
    a = ex["answer"]
    a = re.sub(r"<<[^>]*>>", "", a)
    a = a.replace("####", "ANSWER:")
    a = a.strip()
    user = TEMPLATE.format(question=q)
    text = f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n{a}<|im_end|>\n"
    out.append({"text": text})
with open("/workspace/train_data.jsonl", "w") as f:
    for r in out:
        f.write(json.dumps(r) + "\n")
print("examples:", len(out))
print(out[0]["text"][:500])
