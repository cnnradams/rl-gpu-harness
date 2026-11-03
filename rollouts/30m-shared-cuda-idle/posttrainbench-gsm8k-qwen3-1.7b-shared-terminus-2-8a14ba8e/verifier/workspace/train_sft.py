import json, torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

MODEL = "Qwen/Qwen3-1.7B-Base"
tok = AutoTokenizer.from_pretrained(MODEL)
rows = [json.loads(l) for l in open("/workspace/train.jsonl")]

def tok_fn(r):
    p = tok(r["prompt"], add_special_tokens=False)["input_ids"]
    c = tok(r["completion"], add_special_tokens=False)["input_ids"]
    if len(p) + len(c) > 4096:
        return {"input_ids": [], "labels": [], "length": 0}
    return {"input_ids": p + c, "labels": [-100]*len(p) + c, "length": len(p)+len(c)}

ds = Dataset.from_list(rows).map(tok_fn, remove_columns=["prompt","completion"], num_proc=8)
ds = ds.filter(lambda r: r["length"] > 0)
print("dataset size", len(ds))
PAD = tok.pad_token_id if tok.pad_token_id is not None else 151643

def collate(batch):
    maxlen = max(len(b["input_ids"]) for b in batch)
    input_ids, labels, attn = [], [], []
    for b in batch:
        n = len(b["input_ids"]); pad = maxlen - n
        input_ids.append(b["input_ids"] + [PAD]*pad)
        labels.append(b["labels"] + [-100]*pad)
        attn.append([1]*n + [0]*pad)
    return {"input_ids": torch.tensor(input_ids), "labels": torch.tensor(labels), "attention_mask": torch.tensor(attn)}

model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
model.config.use_cache = False
model.gradient_checkpointing_enable()
args = TrainingArguments(
    output_dir="/workspace/ckpt",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    learning_rate=1e-5,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    logging_steps=10,
    save_strategy="no",
    bf16=True,
    group_by_length=True,
    length_column_name="length",
    report_to=[],
    dataloader_num_workers=4,
)
trainer = Trainer(model=model, args=args, train_dataset=ds, data_collator=collate)
trainer.train()
model.config.use_cache = True
model.generation_config.eos_token_id = 151645
model.generation_config.pad_token_id = PAD
trainer.save_model("/workspace/final_model")
tok.save_pretrained("/workspace/final_model")
print("SAVED")
