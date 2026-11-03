import os, re, random, json
import torch
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from inspect_ai.dataset import hf_dataset
from inspect_evals.gsm8k.gsm8k import record_to_sample, sample_to_fewshot, MATH_PROMPT_TEMPLATE

MODEL = "Qwen/Qwen3-1.7B-Base"
OUT = "/workspace/sft_out"
random.seed(123)

tok = AutoTokenizer.from_pretrained(MODEL)

fewshots = hf_dataset(path="openai/gsm8k", data_dir="main", split="train", sample_fields=record_to_sample, shuffle=True, seed=42, limit=10)
eval_sys = "\n\n".join(sample_to_fewshot(s) for s in fewshots)

ds = load_dataset("openai/gsm8k", "main", split="train")
records = list(ds)

def clean(r):
    return re.sub(r"<<[^>]*>>", "", r).strip()

def parse(rec):
    parts = rec["answer"].split("####")
    target = parts[-1].strip()
    reasoning = clean("####".join(parts[:-1]))
    return reasoning, target

def fewshot_block(rec):
    reasoning, target = parse(rec)
    return f"{rec['question']}\n\nReasoning:\n{reasoning}\n\nANSWER: {target}"

examples = []
n_skip = 0
for i, rec in enumerate(records):
    reasoning, target = parse(rec)
    user = MATH_PROMPT_TEMPLATE.format(prompt=rec["question"])
    r = random.random()
    sys_block = None
    if r < 0.5:
        sys_block = None
    elif r < 0.9:
        k = random.choice([1, 2, 3])
        others = random.sample(records, k + 1)
        others = [o for o in others if o is not rec][:k]
        sys_block = "\n\n".join(fewshot_block(o) for o in others)
    else:
        sys_block = eval_sys
    if sys_block:
        prompt = f"<|im_start|>system\n{sys_block}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"
    else:
        prompt = f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"
    completion = f"{reasoning}\n\nANSWER: {target}<|im_end|>"
    p_ids = tok(prompt, add_special_tokens=False).input_ids
    c_ids = tok(completion, add_special_tokens=False).input_ids
    if len(p_ids) + len(c_ids) > 3072:
        n_skip += 1
        continue
    examples.append({"input_ids": p_ids + c_ids, "labels": [-100]*len(p_ids) + c_ids, "length": len(p_ids)+len(c_ids)})

print("num examples:", len(examples), "skipped:", n_skip)
train_ds = Dataset.from_list(examples)

def collate(batch):
    maxlen = max(len(b["input_ids"]) for b in batch)
    input_ids, labels, attn = [], [], []
    for b in batch:
        pad = maxlen - len(b["input_ids"])
        input_ids.append(list(b["input_ids"]) + [151643]*pad)
        labels.append(list(b["labels"]) + [-100]*pad)
        attn.append([1]*len(b["input_ids"]) + [0]*pad)
    return {"input_ids": torch.tensor(input_ids), "labels": torch.tensor(labels), "attention_mask": torch.tensor(attn)}

model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation="flash_attention_2")
model.config.use_cache = False

args = TrainingArguments(
    output_dir=OUT,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    num_train_epochs=2,
    learning_rate=1e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    logging_steps=25,
    save_strategy="steps", save_steps=150, save_only_model=True,
    bf16=True,
    group_by_length=True,
    length_column_name="length",
    remove_unused_columns=False, gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
    report_to=[],
    seed=42,
    dataloader_num_workers=4,
)

trainer = Trainer(model=model, args=args, train_dataset=train_ds, data_collator=collate)
trainer.train()

tok.eos_token = "<|im_end|>"
trainer.save_model("/workspace/sft_final")
tok.save_pretrained("/workspace/sft_final")
gc = json.load(open("/workspace/sft_final/generation_config.json"))
gc["eos_token_id"] = [151645, 151643]
json.dump(gc, open("/workspace/sft_final/generation_config.json", "w"), indent=2)
print("DONE TRAINING")
