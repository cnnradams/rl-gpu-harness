import json, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from torch.utils.data import Dataset

MODEL = "Qwen/Qwen3-1.7B-Base"
MAXLEN = 2048

tok = AutoTokenizer.from_pretrained(MODEL)

class DS(Dataset):
    def __init__(self, path):
        self.data = []
        rows = [json.loads(l) for l in open(path)]
        for r in rows:
            prompt = "<|im_start|>system\n" + r["system"] + "<|im_end|>\n<|im_start|>user\n" + r["user"] + "<|im_end|>\n<|im_start|>assistant\n"
            pids = tok(prompt, add_special_tokens=False)["input_ids"]
            aids = tok(r["assistant"] + "<|im_end|>", add_special_tokens=False)["input_ids"]
            ids = (pids + aids)[:MAXLEN]
            labels = ([-100]*len(pids) + aids)[:MAXLEN]
            self.data.append((ids, labels))
        print("tokenized", len(self.data), "avg len", sum(len(x[0]) for x in self.data)/len(self.data))
    def __len__(self):
        return len(self.data)
    def __getitem__(self, i):
        ids, labels = self.data[i]
        return {"input_ids": ids, "labels": labels}

def collate(batch):
    pad = 151643
    m = max(len(b["input_ids"]) for b in batch)
    input_ids = torch.tensor([b["input_ids"] + [pad]*(m-len(b["input_ids"])) for b in batch], dtype=torch.long)
    labels = torch.tensor([b["labels"] + [-100]*(m-len(b["labels"])) for b in batch], dtype=torch.long)
    attn = torch.tensor([[1]*len(b["input_ids"]) + [0]*(m-len(b["input_ids"])) for b in batch], dtype=torch.long)
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attn}

def main():
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    model.config.use_cache = False
    ds = DS("/workspace/train_data.jsonl")
    args = TrainingArguments(
        output_dir="/workspace/ckpt",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=1, max_steps=250,
        learning_rate=1e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=20,
        save_strategy="steps",
        save_steps=250,
        save_total_limit=1,
        save_only_model=True,
        bf16=True, gradient_checkpointing=True,
        report_to=[],
        dataloader_num_workers=2,
        remove_unused_columns=False,
        seed=42,
    )
    trainer = Trainer(model=model, args=args, train_dataset=ds, data_collator=collate)
    trainer.train()
    model.config.use_cache = True
    model.config.eos_token_id = 151645
    model.generation_config.eos_token_id = [151645, 151643]
    model.generation_config.pad_token_id = 151643
    model.generation_config.do_sample = False
    tok.eos_token = "<|im_end|>"
    model.save_pretrained("/workspace/final_model")
    tok.save_pretrained("/workspace/final_model")
    p = "/workspace/final_model/generation_config.json"
    gc = json.load(open(p))
    gc["temperature"] = 0.0
    json.dump(gc, open(p, "w"), indent=2)
    print("saved final_model")

if __name__ == "__main__":
    main()
