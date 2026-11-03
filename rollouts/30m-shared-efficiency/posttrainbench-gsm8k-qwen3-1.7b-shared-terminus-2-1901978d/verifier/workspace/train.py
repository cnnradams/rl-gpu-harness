import json, os, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

MODEL = "Qwen/Qwen3-1.7B-Base"
tok = AutoTokenizer.from_pretrained(MODEL)
IM_START, IM_END = "<|im_start|>", "<|im_end|>"

items = []
for line in open("/workspace/train_data.jsonl"):
    d = json.loads(line)
    prompt = (f"{IM_START}system\n{d['system']}{IM_END}\n" if d.get("system") else "")
    prompt += f"{IM_START}user\n{d['user']}{IM_END}\n{IM_START}assistant\n"
    full = prompt + d["assistant"] + IM_END
    p = tok(prompt, add_special_tokens=False)["input_ids"]
    f = tok(full, add_special_tokens=False)["input_ids"]
    if len(f) > 4096:
        continue
    labels = [-100] * len(p) + f[len(p):]
    items.append({"input_ids": f, "labels": labels, "length": len(f)})
print("NUM_ITEMS", len(items))

def collate(batch):
    maxlen = max(len(b["input_ids"]) for b in batch)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    input_ids, labels, attn = [], [], []
    for b in batch:
        n = maxlen - len(b["input_ids"])
        input_ids.append(b["input_ids"] + [pad] * n)
        labels.append(b["labels"] + [-100] * n)
        attn.append([1] * len(b["input_ids"]) + [0] * n)
    return {"input_ids": torch.tensor(input_ids), "labels": torch.tensor(labels), "attention_mask": torch.tensor(attn)}

model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
model.config.use_cache = False
args = TrainingArguments(output_dir="/workspace/ckpt", per_device_train_batch_size=8, gradient_accumulation_steps=2, gradient_checkpointing=True, num_train_epochs=1,
    learning_rate=2e-5, lr_scheduler_type="cosine", warmup_steps=20, logging_steps=20, save_strategy="no",
    bf16=True, group_by_length=True, length_column_name="length", report_to=[], dataloader_num_workers=2,
    optim="adamw_torch_fused", seed=1)
trainer = Trainer(model=model, args=args, train_dataset=items, data_collator=collate)
trainer.train()
model.config.use_cache = True
model.save_pretrained("/workspace/final_model")
tok.save_pretrained("/workspace/final_model")
gc = {}
if os.path.exists("/workspace/final_model/generation_config.json"):
    gc = json.load(open("/workspace/final_model/generation_config.json"))
gc["eos_token_id"] = [151645, 151643]
json.dump(gc, open("/workspace/final_model/generation_config.json", "w"))
cfg = json.load(open("/workspace/final_model/config.json"))
cfg["eos_token_id"] = 151645
json.dump(cfg, open("/workspace/final_model/config.json", "w"))
print("TRAIN_DONE")
