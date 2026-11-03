import json, os, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForSeq2Seq

MODEL = "Qwen/Qwen3-1.7B-Base"
tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
template = open("/workspace/templates/qwen3.jinja").read()

data = json.load(open("/workspace/sft_data.json"))
rows = []
for ex in data:
    msgs = [{"role": "system", "content": ex["system"]}, {"role": "user", "content": ex["user"]}]
    prompt_ids = tok.apply_chat_template(msgs, chat_template=template, add_generation_prompt=True, tokenize=True)
    comp_ids = tok(ex["completion"] + "<|im_end|>\n", add_special_tokens=False)["input_ids"]
    ids = prompt_ids + comp_ids
    labels = [-100] * len(prompt_ids) + comp_ids
    rows.append({"input_ids": ids, "labels": labels})

print("max len:", max(len(r["input_ids"]) for r in rows), "mean:", sum(len(r["input_ids"]) for r in rows)/len(rows))
print(tok.decode(rows[0]["input_ids"])[:1500])

from datasets import Dataset
ds = Dataset.from_list(rows)

try:
    import flash_attn  # noqa
    attn = "flash_attention_2"
except Exception:
    attn = "flash_attention_2"
print("attn:", attn)

model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, attn_implementation=attn)
model.config.use_cache = False

args = TrainingArguments(
    output_dir="/workspace/ckpt",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=32,
    num_train_epochs=1,
    max_steps=150,
    learning_rate=1e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    logging_steps=10,
    save_strategy="no",
    bf16=True,
    report_to=[],
    seed=42,
)

collator = DataCollatorForSeq2Seq(tokenizer=tok, padding=True, label_pad_token_id=-100)
trainer = Trainer(model=model, args=args, train_dataset=ds, data_collator=collator)
trainer.train()
model.config.use_cache = True
trainer.save_model("/workspace/final_model")
tok.save_pretrained("/workspace/final_model")
print("DONE")
