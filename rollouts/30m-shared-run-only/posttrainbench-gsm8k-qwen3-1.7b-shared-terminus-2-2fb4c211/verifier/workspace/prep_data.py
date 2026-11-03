import json, random
from datasets import load_dataset
from inspect_ai.dataset import hf_dataset
from inspect_evals.gsm8k.gsm8k import record_to_sample, sample_to_fewshot, MATH_PROMPT_TEMPLATE

fewshots = hf_dataset(path="openai/gsm8k", data_dir="main", split="train",
                      sample_fields=record_to_sample, shuffle=True, seed=42, limit=10)
sys_block = "\n\n".join([sample_to_fewshot(s) for s in fewshots])
fewshot_ids = set(s.id for s in fewshots)

pool = hf_dataset(path="openai/gsm8k", data_dir="main", split="train",
                  sample_fields=record_to_sample, shuffle=True, seed=123, limit=120)
pool_strs = [sample_to_fewshot(s) for s in pool]

ds = load_dataset("openai/gsm8k", "main", split="train")
random.seed(0)
n = 0
with open("/workspace/train_data.jsonl", "w") as f:
    for rec in ds:
        sample = record_to_sample(rec)
        reasoning = sample.metadata["reasoning"]
        target = sample.target
        assistant = f"Reasoning:\n{reasoning}\n\nSo the answer is {target}.\nANSWER: {target}"
        user = MATH_PROMPT_TEMPLATE.format(prompt=sample.input)
        if sample.id in fewshot_ids:
            sys = sys_block
        else:
            r = random.random()
            if r < 0.5:
                sys = sys_block
            elif r < 0.6:
                sys = ""
            else:
                sys = "\n\n".join(random.sample(pool_strs, 10))
        msgs = []
        if sys:
            msgs.append({"role": "system", "content": sys})
        msgs.append({"role": "user", "content": user})
        msgs.append({"role": "assistant", "content": assistant})
        f.write(json.dumps({"messages": msgs}) + "\n")
        n += 1
print("wrote", n)
print("SYS_BLOCK_START")
print(sys_block[:800])
