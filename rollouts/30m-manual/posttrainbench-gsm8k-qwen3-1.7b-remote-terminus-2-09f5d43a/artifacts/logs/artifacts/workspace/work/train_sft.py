import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

model_name = "Qwen/Qwen3-1.7B-Base"
tok = AutoTokenizer.from_pretrained(model_name)
ds = load_dataset("json", data_files="train.jsonl", split="train")

def tokenize(ex):
    return tok(ex["text"], add_special_tokens=False)

ds = ds.map(tokenize, remove_columns=["text"], num_proc=8)

block = 3072
def group_texts(examples):
    concat = []
    for ids in examples["input_ids"]:
        concat.extend(ids)
    chunks = [concat[i:i+block] for i in range(0, len(concat) - block + 1, block)]
    return {"input_ids": chunks, "labels": [c[:] for c in chunks]}

packed = ds.map(group_texts, batched=True, batch_size=1000, num_proc=8, remove_columns=ds.column_names)
print("packed blocks:", len(packed))

pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
def collate(features):
    mx = max(len(f["input_ids"]) for f in features)
    input_ids, labels, attn = [], [], []
    for f in features:
        ids = list(f["input_ids"]); lbs = list(f["labels"])
        pad = mx - len(ids)
        input_ids.append(ids + [pad_id]*pad)
        labels.append(lbs + [-100]*pad)
        attn.append([1]*len(ids) + [0]*pad)
    return {"input_ids": torch.tensor(input_ids), "labels": torch.tensor(labels), "attention_mask": torch.tensor(attn)}

model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
model.config.use_cache = False

args = TrainingArguments(
    output_dir="ckpt",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=16,
    learning_rate=2e-5,
    num_train_epochs=2,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    logging_steps=10,
    save_strategy="epoch",
    save_only_model=True,
    bf16=True,
    remove_unused_columns=False,
    report_to=[],
    dataloader_num_workers=2,
    seed=42,
)

trainer = Trainer(model=model, args=args, train_dataset=packed, data_collator=collate)
trainer.train()
model.config.use_cache = True
trainer.save_model("final_model")
tok.save_pretrained("final_model")
print("TRAINING_DONE")
