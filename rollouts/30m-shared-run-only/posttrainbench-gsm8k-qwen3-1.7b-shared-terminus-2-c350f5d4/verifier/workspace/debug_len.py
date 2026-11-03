import re
from datasets import load_dataset
from transformers import AutoTokenizer
from inspect_ai.dataset import hf_dataset
from inspect_evals.gsm8k.gsm8k import record_to_sample, sample_to_fewshot, MATH_PROMPT_TEMPLATE

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B-Base")
fewshots = hf_dataset(path="openai/gsm8k", data_dir="main", split="train", sample_fields=record_to_sample, shuffle=True, seed=42, limit=10)
system_msg = "\n\n".join(sample_to_fewshot(s) for s in fewshots)
sys_len = len(tok(system_msg, add_special_tokens=False).input_ids)
print("system msg tokens:", sys_len)

ds = load_dataset("openai/gsm8k", "main", split="train")
import numpy as np
plens, clens = [], []
for rec in list(ds)[:500]:
    q = rec["question"]
    parts = rec["answer"].split("####")
    target = parts[-1].strip()
    reasoning = re.sub(r"<<[^>]*>>", "", "####".join(parts[:-1])).strip()
    user = MATH_PROMPT_TEMPLATE.format(prompt=q)
    prompt = f"<|im_start|>system\n{system_msg}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"
    completion = f"{reasoning}\n\nANSWER: {target}<|im_end|>"
    plens.append(len(tok(prompt, add_special_tokens=False).input_ids))
    clens.append(len(tok(completion, add_special_tokens=False).input_ids))
plens, clens = np.array(plens), np.array(clens)
tot = plens + clens
print("prompt: mean %.0f p50 %.0f p95 %.0f max %.0f" % (plens.mean(), np.percentile(plens,50), np.percentile(plens,95), plens.max()))
print("completion: mean %.0f p50 %.0f p95 %.0f max %.0f" % (clens.mean(), np.percentile(clens,50), np.percentile(clens,95), clens.max()))
print("total: mean %.0f p50 %.0f p95 %.0f max %.0f" % (tot.mean(), np.percentile(tot,50), np.percentile(tot,95), tot.max()))
print("frac <= 2048:", (tot<=2048).mean(), "frac <= 4096:", (tot<=4096).mean())
