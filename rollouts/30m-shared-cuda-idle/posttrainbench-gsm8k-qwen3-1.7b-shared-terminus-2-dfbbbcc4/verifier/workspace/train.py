import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig

from inspect_ai.dataset import hf_dataset
from inspect_evals.gsm8k.gsm8k import sample_to_fewshot, record_to_sample, MATH_PROMPT_TEMPLATE

MODEL = "Qwen/Qwen3-1.7B-Base"

fewshots = hf_dataset(path="openai/gsm8k", data_dir="main", split="train",
                      sample_fields=record_to_sample, shuffle=True, seed=42, limit=10)
system_msg = "\n\n".join([sample_to_fewshot(s) for s in fewshots])

tok = AutoTokenizer.from_pretrained(MODEL)
chat_template = open("/workspace/templates/qwen3.jinja").read()

ds = load_dataset("openai/gsm8k", "main", split="train")

def build(ex):
    q = ex["question"]
    ans = ex["answer"]
    parts = ans.split("####")
    target = parts[-1].strip()
    reasoning = "####".join(parts[:-1]).strip()
    user = MATH_PROMPT_TEMPLATE.format(prompt=q)
    assistant = f"{reasoning}\n\nANSWER: {target}"
    msgs_prompt = [{"role":"system","content":system_msg},{"role":"user","content":user}]
    prompt = tok.apply_chat_template(msgs_prompt, tokenize=False, add_generation_prompt=True, chat_template=chat_template)
    full_msgs = msgs_prompt + [{"role":"assistant","content":assistant}]
    full = tok.apply_chat_template(full_msgs, tokenize=False, add_generation_prompt=False, chat_template=chat_template)
    assert full.startswith(prompt)
    completion = full[len(prompt):]
    return {"prompt": prompt, "completion": completion}

train_ds = ds.map(build, num_proc=8, remove_columns=ds.column_names)
print("train size:", len(train_ds))
print(train_ds[0]["prompt"][:500])
print("COMPLETION:", train_ds[0]["completion"][:300])

model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")

cfg = SFTConfig(
    output_dir="/workspace/ckpt",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    learning_rate=1e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    logging_steps=10,
    save_strategy="no",
    bf16=True,
    max_length=2048,
    packing=False,
    dataset_num_proc=8,
    report_to=[],
    seed=42,
)

trainer = SFTTrainer(model=model, args=cfg, train_dataset=train_ds, processing_class=tok)
trainer.train()
trainer.save_model("/workspace/final_model")
tok.save_pretrained("/workspace/final_model")
print("DONE")
