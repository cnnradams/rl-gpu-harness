import json, torch
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

MODEL = "Qwen/Qwen3-1.7B-Base"
tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
chat_template = open("/workspace/templates/qwen3.jinja").read()
rows = [json.loads(l) for l in open("/workspace/train_clean.jsonl")][:3200]

class DS(Dataset):
    def __init__(self, rows, maxlen=4096):
        self.ex = []
        skipped = 0
        for r in rows:
            msgs = [{"role":"system","content":r["system"]},{"role":"user","content":r["user"]}]
            prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, chat_template=chat_template)
            full = prompt + r["assistant"] + "<|im_end|>"
            p = tok(prompt, add_special_tokens=False)["input_ids"]
            f = tok(full, add_special_tokens=False)["input_ids"]
            if len(f) > maxlen:
                skipped += 1
                continue
            labels = [-100]*len(p) + f[len(p):]
            self.ex.append({"input_ids": f, "labels": labels})
        print("examples:", len(self.ex), "skipped:", skipped)
    def __len__(self):
        return len(self.ex)
    def __getitem__(self, i):
        return self.ex[i]

def collate(batch):
    ml = max(len(b["input_ids"]) for b in batch)
    input_ids, labels, attn = [], [], []
    for b in batch:
        n = ml - len(b["input_ids"])
        input_ids.append(b["input_ids"] + [tok.pad_token_id]*n)
        labels.append(b["labels"] + [-100]*n)
        attn.append([1]*len(b["input_ids"]) + [0]*n)
    return {"input_ids": torch.tensor(input_ids), "labels": torch.tensor(labels), "attention_mask": torch.tensor(attn)}

ds = DS(rows)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
model.config.use_cache = False
args = TrainingArguments(
    output_dir="/workspace/ckpt",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    learning_rate=1e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    logging_steps=20,
    save_strategy="no",
    bf16=True,
    report_to=[],
    dataloader_num_workers=4,
    optim="adamw_torch_fused",
    seed=42,
)
trainer = Trainer(model=model, args=args, train_dataset=ds, data_collator=collate)
trainer.train()
model.config.use_cache = True
model.save_pretrained("/workspace/final_model")
tok.save_pretrained("/workspace/final_model")
gc = {"bos_token_id":151643,"eos_token_id":[151645,151643],"pad_token_id":151643,"do_sample":False,"temperature":0.0,"top_p":1.0,"repetition_penalty":1.0}
json.dump(gc, open("/workspace/final_model/generation_config.json","w"), indent=2)
print("SAVED")
