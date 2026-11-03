import re, json
from datasets import load_dataset

PROMPT = """Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{q}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:"""

ds = load_dataset("openai/gsm8k", "main", split="train")
n = 0
with open("/workspace/train_data.jsonl", "w") as f:
    for ex in ds:
        parts = ex["answer"].split("####")
        reasoning = "####".join(parts[:-1]).strip()
        final = parts[-1].strip().replace(",", "")
        reasoning = re.sub(r"<<[^>]*>>", "", reasoning)
        text = ("<|im_start|>user\n" + PROMPT.format(q=ex["question"]) + "<|im_end|>\n"
                "<|im_start|>assistant\n" + reasoning + "\n\nANSWER: " + final + "<|im_end|>")
        f.write(json.dumps({"text": text}) + "\n")
        n += 1
print("wrote", n)
