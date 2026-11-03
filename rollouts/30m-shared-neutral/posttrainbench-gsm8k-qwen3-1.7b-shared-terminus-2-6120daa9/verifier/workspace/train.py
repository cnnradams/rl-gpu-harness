import os, re, json, random
import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForSeq2Seq

MODEL = "Qwen/Qwen3-1.7B-Base"
OUT = "/workspace/final_model"
MAXLEN = 1792

from inspect_evals.gsm8k.gsm8k import MATH_PROMPT_TEMPLATE, record_to_sample, sample_to_fewshot
from inspect_ai.dataset import hf_dataset

random.seed(42)
tok = AutoTokenizer.from_pretrained(MODEL)
tok.chat_template = open("/workspace/templates/qwen3.jinja").read()

fewshots = hf_dataset(path="openai/gsm8k", data_dir="main", split="train",
                      sample_fields=record_to_sample, shuffle=True, seed=42, limit=10)
SYS = "\n\n".join(sample_to_fewshot(s) for s in fewshots)
print("SYS PROMPT:\n", SYS[:500], flush=True)

ds = load_dataset("openai/gsm8k", "main", split="train")
pat = re.compile(r"<<[^>]*>>")

def build(ex, with_sys):
    reasoning, ans = ex["answer"].split("####")
    reasoning = pat.sub("", reasoning).strip()
    ans = ans.strip().replace(",", "")
    user = MATH_PROMPT_TEMPLATE.replace("{prompt}", ex["question"])
    msgs = ([{"role": "system", "content": SYS}] if with_sys else []) + [{"role": "user", "content": user}]
    prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    completion = f"{reasoning}\n\nANSWER: {ans}<|im_end|>\n"
    pids = tok(prompt, add_special_tokens=False)["input_ids"]
    cids = tok(completion, add_special_tokens=False)["input_ids"]
    if len(pids) + len(cids) > MAXLEN:
        return None
    return {"input_ids": pids + cids, "labels": [-100] * len(pids) + cids, "length": len(pids) + len(cids)}

data = []
for i, ex in enumerate(ds):
    r = build(ex, with_sys=(i % 2 == 0))
    if r:
        data.append(r)
print("num examples:", len(data), flush=True)
dd = Dataset.from_list(data)

model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation="flash_attention_2")
model.config.use_cache = False

args = TrainingArguments(
    output_dir="/workspace/ckpt",
    per_device_train_batch_size=16,
    gradient_accumulation_steps=1,
    num_train_epochs=1,
    learning_rate=1e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    bf16=True,
    logging_steps=20,
    save_strategy="no",
    report_to=[],
    group_by_length=True,
    length_column_name="length",
    dataloader_num_workers=4,
    seed=42,
    optim="adamw_torch_fused",
    max_grad_norm=1.0,
)

trainer = Trainer(model=model, args=args, train_dataset=dd,
                  data_collator=DataCollatorForSeq2Seq(tok, label_pad_token_id=-100))
trainer.train()

model.config.use_cache = True
model.save_pretrained(OUT)
tok.save_pretrained(OUT)
json.dump({"eos_token_id": [151645, 151643], "pad_token_id": 151643, "temperature": 0.0,
           "transformers_version": "4.57.3"}, open(f"{OUT}/generation_config.json", "w"), indent=2)
print("TRAINING DONE", flush=True)
