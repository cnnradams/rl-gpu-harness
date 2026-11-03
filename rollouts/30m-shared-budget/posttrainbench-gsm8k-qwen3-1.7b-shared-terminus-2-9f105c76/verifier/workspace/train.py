import os, torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments

MODEL = "Qwen/Qwen3-1.7B-Base"
MAXLEN = 4096

tok = AutoTokenizer.from_pretrained(MODEL)
chat_template = open("/workspace/templates/qwen3.jinja").read()

MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

ds = load_dataset("openai/gsm8k", "main", split="train")
fs = ds.shuffle(seed=42).select(range(10))

def fewshot_text(ex):
    reasoning, target = ex["answer"].split("####")
    return f"{ex['question']}\n\nReasoning:\n{reasoning.strip()}\n\nANSWER: {target.strip()}"

SYS = "\n\n".join(fewshot_text(e) for e in fs)

def build(ex):
    reasoning, target = ex["answer"].split("####")
    user = MATH_PROMPT_TEMPLATE.format(prompt=ex["question"])
    msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": user}]
    prompt = tok.apply_chat_template(msgs, chat_template=chat_template, tokenize=False, add_generation_prompt=True)
    completion = reasoning.strip() + "\n\nANSWER: " + target.strip() + "<|im_end|>\n"
    p = tok(prompt, add_special_tokens=False)["input_ids"]
    c = tok(completion, add_special_tokens=False)["input_ids"]
    return {"input_ids": p + c, "labels": [-100]*len(p) + c}

data = ds.map(build, remove_columns=ds.column_names, num_proc=8)
data = data.filter(lambda x: len(x["input_ids"]) <= MAXLEN, num_proc=8)
data = data.shuffle(seed=0).select(range(4000))
print("train size:", len(data), flush=True)

def collate(feats):
    maxlen = max(len(f["input_ids"]) for f in feats)
    pad = tok.pad_token_id or tok.eos_token_id
    ids, labs, att = [], [], []
    for f in feats:
        n = maxlen - len(f["input_ids"])
        ids.append(f["input_ids"] + [pad]*n)
        labs.append(f["labels"] + [-100]*n)
        att.append([1]*len(f["input_ids"]) + [0]*n)
    return {"input_ids": torch.tensor(ids), "labels": torch.tensor(labs), "attention_mask": torch.tensor(att)}

model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation="sdpa")
model.config.use_cache = False

args = TrainingArguments(
    output_dir="/workspace/ckpt",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    learning_rate=1e-5,
    lr_scheduler_type="cosine",
    warmup_steps=10,
    logging_steps=10,
    save_strategy="no",
    bf16=True,
    report_to=[],
    dataloader_num_workers=4,
    remove_unused_columns=False,
)

trainer = Trainer(model=model, args=args, train_dataset=data, data_collator=collate)
trainer.train()
model.config.use_cache = True
model.save_pretrained("/workspace/final_model")
tok.save_pretrained("/workspace/final_model")
print("DONE")
