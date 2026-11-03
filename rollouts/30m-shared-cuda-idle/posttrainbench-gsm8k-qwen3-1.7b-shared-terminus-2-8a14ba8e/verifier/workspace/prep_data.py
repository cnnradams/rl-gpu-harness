import json
from datasets import load_dataset

TEMPLATE = 'Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.\n\n{q}\n\nRemember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.\n\nReasoning:'

train = load_dataset("openai/gsm8k", "main", split="train")
few = train.shuffle(seed=42).select(range(10))

def fs(ex):
    reasoning, ans = ex["answer"].rsplit("####", 1)
    return f"{ex['question']}\n\nReasoning:\n{reasoning.strip()}\n\nANSWER: {ans.strip()}"

sysmsg = "\n\n".join(fs(e) for e in few)
print("SYSMSG:", sysmsg[:500])

out = open("/workspace/train.jsonl", "w")
for ex in train:
    reasoning, ans = ex["answer"].rsplit("####", 1)
    user = TEMPLATE.format(q=ex["question"])
    prompt = f"<|im_start|>system\n{sysmsg}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"
    completion = f"{reasoning.strip()}\n\nANSWER: {ans.strip()}<|im_end|>"
    out.write(json.dumps({"prompt": prompt, "completion": completion}) + "\n")
out.close()
print("done", len(train))
