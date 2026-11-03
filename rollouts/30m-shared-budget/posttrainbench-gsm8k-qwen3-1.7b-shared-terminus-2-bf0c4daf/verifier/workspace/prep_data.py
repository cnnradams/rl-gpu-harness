#!/usr/bin/env python3
"""Build SFT data from the GSM8K *train* split (the test split is never used)."""
import json
from datasets import load_dataset

PROMPT_TEMPLATE = """Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:"""

CHAT_PREFIX = "<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n"

def main():
    ds = load_dataset("openai/gsm8k", "main", split="train")
    print("train examples:", len(ds))
    n_skip = 0
    n = 0
    with open("/workspace/train_data_raw.jsonl", "w") as f:
        for rec in ds:
            q = rec["question"].strip()
            ans = rec["answer"]
            parts = ans.split("####")
            if len(parts) < 2:
                n_skip += 1
                continue
            reasoning = "####".join(parts[:-1]).strip()
            target = parts[-1].strip()
            resp = reasoning + "\n\nANSWER: " + target
            user = PROMPT_TEMPLATE.format(prompt=q)
            text = CHAT_PREFIX.format(p=user) + resp + "<|im_end|>"
            f.write(json.dumps({"text": text}) + "\n")
            n += 1
    print("written:", n, "skipped:", n_skip)

if __name__ == "__main__":
    main()
