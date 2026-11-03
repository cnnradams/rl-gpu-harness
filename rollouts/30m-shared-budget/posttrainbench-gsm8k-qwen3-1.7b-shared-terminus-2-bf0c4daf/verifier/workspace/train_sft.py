#!/usr/bin/env python3
import json
import torch
from dataclasses import dataclass
from torch.utils.data import Dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments)

MODEL = "Qwen/Qwen3-1.7B-Base"
DATA = "/workspace/train_data.jsonl"
OUT = "/workspace/final_model"
MAX_LEN = 1024
IM_END = "<|im_end|>"

class SFTDataset(Dataset):
    def __init__(self, path, tok):
        self.examples = []
        with open(path) as f:
            for line in f:
                self.examples.append(json.loads(line)["text"])
        self.tok = tok
    def __len__(self):
        return len(self.examples)
    def __getitem__(self, i):
        tok = self.tok
        text = self.examples[i]
        prompt_part, resp_part = text.split("<|im_start|>assistant\n", 1)
        prompt = prompt_part + "<|im_start|>assistant\n"
        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        r_ids = tok(resp_part, add_special_tokens=False)["input_ids"]
        im_end_id = tok.convert_tokens_to_ids(IM_END)
        # resp_part ends with the literal text '<|im_end|>'; special tokens are
        # skipped by the tokenizer, so always append the id explicitly.
        r_ids = r_ids + [im_end_id]
        ids = p_ids + r_ids
        labels = [-100] * len(p_ids) + r_ids[:]
        ids = ids[:MAX_LEN]
        labels = labels[:MAX_LEN]
        return {"input_ids": ids, "labels": labels}

@dataclass
class Collator:
    pad_id: int
    def __call__(self, feats):
        maxlen = max(len(f["input_ids"]) for f in feats)
        input_ids, labels, attn = [], [], []
        for f in feats:
            ids = f["input_ids"]
            lab = f["labels"]
            pad = maxlen - len(ids)
            input_ids.append(ids + [self.pad_id] * pad)
            labels.append(lab + [-100] * pad)
            attn.append([1] * len(ids) + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids),
            "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(attn),
        }

def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, attn_implementation="flash_attention_2")
    model.config.use_cache = False
    ds = SFTDataset(DATA, tok)
    print("dataset size:", len(ds))
    args = TrainingArguments(
        output_dir="/workspace/sft_ckpt",
        per_device_train_batch_size=16,
        gradient_accumulation_steps=4,
        num_train_epochs=1.5,
        learning_rate=1e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.0,
        bf16=True,
        logging_steps=10,
        save_strategy="no",
        report_to=[],
        gradient_checkpointing=True,
        seed=42,
        dataloader_num_workers=4,
    )
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    trainer = Trainer(model=model, args=args, train_dataset=ds,
                      data_collator=Collator(pad_id))
    trainer.train()
    model.config.use_cache = True
    # make sure vllm stops at <|im_end|> as well as <|endoftext|>
    im_end_id = tok.convert_tokens_to_ids(IM_END)
    eos_ids = list({tok.eos_token_id, im_end_id} - {None})
    model.generation_config.eos_token_id = eos_ids
    model.config.eos_token_id = im_end_id
    trainer.save_model(OUT)
    tok.save_pretrained(OUT)
    print("saved to", OUT, "eos ids:", eos_ids)

if __name__ == "__main__":
    main()
