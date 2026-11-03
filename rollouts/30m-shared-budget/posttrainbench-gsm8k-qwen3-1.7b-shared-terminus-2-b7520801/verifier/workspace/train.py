import json, torch, os
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

MODEL = "Qwen/Qwen3-1.7B-Base"
tok = AutoTokenizer.from_pretrained(MODEL)
im_end = tok.convert_tokens_to_ids("<|im_end|>")
print("im_end id:", im_end)

rows = [json.loads(l) for l in open("/workspace/train_data.jsonl")]
print("rows:", len(rows))

def fmt(p, c):
    text = ("<|im_start|>user\n" + p + "<|im_end|>\n"
            "<|im_start|>assistant\n" + c + "<|im_end|>\n<|endoftext|>")
    return text

def gen():
    for r in rows:
        full = fmt(r["prompt"], r["completion"])
        prompt_part = "<|im_start|>user\n" + r["prompt"] + "<|im_end|>\n<|im_start|>assistant\n"
        full_ids = tok(full, add_special_tokens=False)["input_ids"]
        prompt_ids = tok(prompt_part, add_special_tokens=False)["input_ids"]
        labels = [-100]*len(prompt_ids) + full_ids[len(prompt_ids):]
        yield {"input_ids": full_ids, "labels": labels}

ds = Dataset.from_generator(gen)
print(ds)

model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
model.config.use_cache = False

def collate(batch):
    maxlen = max(len(b["input_ids"]) for b in batch)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    input_ids, labels, attn = [], [], []
    for b in batch:
        n = maxlen - len(b["input_ids"])
        input_ids.append(b["input_ids"] + [pad_id]*n)
        labels.append(b["labels"] + [-100]*n)
        attn.append([1]*len(b["input_ids"]) + [0]*n)
    return {"input_ids": torch.tensor(input_ids), "labels": torch.tensor(labels), "attention_mask": torch.tensor(attn)}

args = TrainingArguments(
    output_dir="/workspace/ckpt",
    per_device_train_batch_size=16,
    gradient_accumulation_steps=2,
    num_train_epochs=1,
    learning_rate=1e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    logging_steps=20,
    save_strategy="no",
    bf16=True,
    report_to=[],
    max_grad_norm=1.0,
    dataloader_num_workers=4,
)

trainer = Trainer(model=model, args=args, train_dataset=ds, data_collator=collate)
trainer.train()

model.config.use_cache = True
model.save_pretrained("/workspace/final_model")
tok.save_pretrained("/workspace/final_model")
import json as _j
gc = {"eos_token_id": [151645, 151643], "pad_token_id": 151643, "temperature": None, "top_p": None, "top_k": None, "do_sample": False, "transformers_version": "4.57.3"}
open("/workspace/final_model/generation_config.json","w").write(_j.dumps(gc, indent=2))
print("SAVED")
