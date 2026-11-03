import torch, datasets
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForSeq2Seq

MODEL = "Qwen/Qwen3-1.7B-Base"
OUT = "/workspace/ckpt1"
MAXLEN = 3072

tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

ds = datasets.load_dataset("openai/gsm8k", "main", split="train")

def split_ans(a):
    reasoning, target = a.split("####")
    return reasoning.strip(), target.strip()

fs = ds.shuffle(seed=42).select(range(10))
def fewshot_text(r):
    reasoning, target = split_ans(r["answer"])
    return f"{r['question']}\n\nReasoning:\n{reasoning}\n\nANSWER: {target}"
SYSTEM = "\n\n".join(fewshot_text(r) for r in fs)

TEMPLATE = (
'Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.\n\n'
'{q}\n\n'
'Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.\n\n'
'Reasoning:'
)

sys_ids = tok(f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n", add_special_tokens=False)["input_ids"]
mid_ids = tok(f"<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False)["input_ids"]

rows = []
for r in ds:
    reasoning, target = split_ans(r["answer"])
    user = TEMPLATE.replace("{q}", r["question"])
    p = sys_ids + tok(user, add_special_tokens=False)["input_ids"] + mid_ids
    c = tok(f"{reasoning}\n\nANSWER: {target}<|im_end|>", add_special_tokens=False)["input_ids"]
    if len(p) + len(c) > MAXLEN:
        continue
    rows.append({"input_ids": p + c, "labels": [-100]*len(p) + c})
print("train size", len(rows), flush=True)
train_ds = datasets.Dataset.from_list(rows)

model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation="sdpa")
model.config.use_cache = False

args = TrainingArguments(
    output_dir="/workspace/tmp_out",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    num_train_epochs=1, max_steps=150,
    learning_rate=1e-5,
    lr_scheduler_type="cosine",
    warmup_steps=20,
    logging_steps=25,
    bf16=True,
    save_strategy="no",
    report_to=[],
    remove_unused_columns=False,
    dataloader_num_workers=4,
    seed=42,
)

collator = DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100)
trainer = Trainer(model=model, args=args, train_dataset=train_ds, data_collator=collator)
trainer.train()
model.config.use_cache = True
trainer.save_model(OUT)
tok.save_pretrained(OUT)
print("saved to", OUT, flush=True)
