import os, json, torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments

MODEL = "Qwen/Qwen3-1.7B-Base"
OUT = "/root/ckpt/sft"
MAXLEN = 1024

tok = AutoTokenizer.from_pretrained(MODEL)
with open("/workspace/templates/qwen3.jinja") as f:
    chat_template = f.read()

MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

ds = load_dataset("openai/gsm8k", "main", split="train")

def build(ex):
    q = ex["question"]
    reasoning, target = ex["answer"].split("####")
    reasoning = reasoning.strip(); target = target.strip().replace(",", "")
    user = MATH_PROMPT_TEMPLATE.format(prompt=q)
    msgs = [{"role": "user", "content": user}]
    prompt_text = tok.apply_chat_template(msgs, chat_template=chat_template, tokenize=False, add_generation_prompt=True)
    completion = reasoning + "\n\nANSWER: " + target + "<|im_end|>"
    pids = tok(prompt_text, add_special_tokens=False)["input_ids"]
    cids = tok(completion, add_special_tokens=False)["input_ids"]
    ids = pids + cids
    labels = [-100]*len(pids) + cids
    return {"input_ids": ids, "labels": labels, "length": len(ids)}

ds2 = ds.map(build, remove_columns=ds.column_names, num_proc=8)
ds2 = ds2.filter(lambda x: x["length"] <= MAXLEN, num_proc=8)
print("train size", len(ds2))

pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

def collate(batch):
    maxlen = max(len(b["input_ids"]) for b in batch)
    input_ids, labels, attn = [], [], []
    for b in batch:
        n = maxlen - len(b["input_ids"])
        input_ids.append(b["input_ids"] + [pad]*n)
        labels.append(b["labels"] + [-100]*n)
        attn.append([1]*len(b["input_ids"]) + [0]*n)
    return {"input_ids": torch.tensor(input_ids), "labels": torch.tensor(labels), "attention_mask": torch.tensor(attn)}

model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation="sdpa")
model.config.use_cache = False

args = TrainingArguments(
    output_dir="/root/ckpt/tmp",
    per_device_train_batch_size=16,
    gradient_accumulation_steps=1,
    num_train_epochs=3,
    learning_rate=1e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    bf16=True,
    logging_steps=25,
    save_strategy="no",
    report_to=[],
    dataloader_num_workers=4,
    group_by_length=True,
    length_column_name="length",
    seed=42,
)

trainer = Trainer(model=model, args=args, train_dataset=ds2, data_collator=collate)
trainer.train()

os.makedirs(OUT, exist_ok=True)
model.config.use_cache = True
model.save_pretrained(OUT)
tok.save_pretrained(OUT)
with open(os.path.join(OUT, "generation_config.json"), "w") as f:
    json.dump({"bos_token_id": 151643, "eos_token_id": [151645, 151643], "pad_token_id": 151643, "temperature": 0.0, "transformers_version": "4.57.3"}, f, indent=2)
print("DONE SAVING", OUT)
