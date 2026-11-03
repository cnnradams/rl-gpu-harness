import os, re, random, json
os.environ.setdefault("HF_HOME", "/workspace/hf_cache")
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments

MODEL = "Qwen/Qwen3-1.7B-Base"
MAXLEN = 3072
random.seed(0)

tok = AutoTokenizer.from_pretrained(MODEL)
ds = load_dataset("openai/gsm8k", "main", split="train")

def split_answer(a):
    parts = a.split("####")
    tgt = parts[-1].strip()
    reas = "####".join(parts[:-1]).strip()
    return reas, tgt

def clean(s):
    return re.sub(r"<<[^>]*>>", "", s)

TEMPLATE = (
    'Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.\n\n'
    '{q}\n\n'
    'Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.\n\n'
    'Reasoning:'
)

fs = ds.shuffle(seed=42).select(range(10))
sysmsg = "\n\n".join(
    r["question"] + "\n\nReasoning:\n" + split_answer(r["answer"])[0] + "\n\nANSWER: " + split_answer(r["answer"])[1]
    for r in fs
)

chat_template = open("/workspace/templates/qwen3.jinja").read()

def render_prompt(q, use_sys):
    msgs = []
    if use_sys:
        msgs.append({"role": "system", "content": sysmsg})
    msgs.append({"role": "user", "content": TEMPLATE.format(q=q)})
    return tok.apply_chat_template(msgs, chat_template=chat_template, tokenize=False, add_generation_prompt=True)

examples = []
idx = list(range(len(ds)))
random.shuffle(idx)
dropped = 0
for n, i in enumerate(idx):
    r = ds[i]
    reas, tgt = split_answer(r["answer"])
    reas = clean(reas)
    use_sys = (n % 4 == 0)
    prompt = render_prompt(r["question"], use_sys)
    completion = reas + "\n\nANSWER: " + tgt + "<|im_end|>"
    pids = tok(prompt, add_special_tokens=False)["input_ids"]
    cids = tok(completion, add_special_tokens=False)["input_ids"]
    ids = pids + cids
    if len(ids) > MAXLEN:
        dropped += 1
        continue
    labels = [-100]*len(pids) + cids
    examples.append({"input_ids": ids, "labels": labels, "attention_mask": [1]*len(ids)})

print("num examples", len(examples), "dropped", dropped, flush=True)

class DS(torch.utils.data.Dataset):
    def __len__(self): return len(examples)
    def __getitem__(self, i): return examples[i]

pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

def collate(batch):
    m = max(len(b["input_ids"]) for b in batch)
    input_ids, labels, attn = [], [], []
    for b in batch:
        p = m - len(b["input_ids"])
        input_ids.append(b["input_ids"] + [pad_id]*p)
        labels.append(b["labels"] + [-100]*p)
        attn.append(b["attention_mask"] + [0]*p)
    return {"input_ids": torch.tensor(input_ids), "labels": torch.tensor(labels), "attention_mask": torch.tensor(attn)}

model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
model.config.use_cache = False
model.gradient_checkpointing_enable()

args = TrainingArguments(
    output_dir="/workspace/ckpt",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    num_train_epochs=1,
    learning_rate=1e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    logging_steps=20,
    save_strategy="no",
    bf16=True,
    report_to=[],
    remove_unused_columns=False,
    dataloader_num_workers=2,
)

trainer = Trainer(model=model, args=args, train_dataset=DS(), data_collator=collate)
trainer.train()

model.config.use_cache = True
model.save_pretrained("/workspace/final_model")
tok.save_pretrained("/workspace/final_model")

gc_path = "/workspace/final_model/generation_config.json"
gc = {}
if os.path.exists(gc_path):
    gc = json.load(open(gc_path))
gc["eos_token_id"] = [151645, 151643]
json.dump(gc, open(gc_path, "w"), indent=2)
cfg = json.load(open("/workspace/final_model/config.json"))
cfg["eos_token_id"] = 151645
json.dump(cfg, open("/workspace/final_model/config.json", "w"), indent=2)
print("DONE", flush=True)
