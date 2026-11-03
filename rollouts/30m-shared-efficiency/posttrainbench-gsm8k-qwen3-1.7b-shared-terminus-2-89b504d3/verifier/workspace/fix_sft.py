import os, re, random, json
os.environ.setdefault("HF_HOME", "/workspace/hf_cache")
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments

MODEL = "/workspace/final_model"
MAXLEN = 3072
random.seed(0)
tok = AutoTokenizer.from_pretrained(MODEL)
IM_END = 151645

ds = load_dataset("openai/gsm8k", "main", split="train")

def split_answer(a):
    parts = a.split("####")
    return "####".join(parts[:-1]).strip(), parts[-1].strip()

def clean(s):
    return re.sub(r"<<[^>]*>>", "", s)

TEMPLATE = ('Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.\n\n{q}\n\nRemember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.\n\nReasoning:')

fs = ds.shuffle(seed=42).select(range(10))
sysmsg = "\n\n".join(r["question"] + "\n\nReasoning:\n" + split_answer(r["answer"])[0] + "\n\nANSWER: " + split_answer(r["answer"])[1] for r in fs)
chat_template = open("/workspace/templates/qwen3.jinja").read()

def render_prompt(q, use_sys):
    msgs = ([{"role": "system", "content": sysmsg}] if use_sys else []) + [{"role": "user", "content": TEMPLATE.format(q=q)}]
    return tok.apply_chat_template(msgs, chat_template=chat_template, tokenize=False, add_generation_prompt=True)

examples = []
idx = list(range(len(ds))); random.shuffle(idx)
for n, i in enumerate(idx[:4000]):
    r = ds[i]
    reas, tgt = split_answer(r["answer"])
    prompt = render_prompt(r["question"], n % 4 == 0)
    completion = clean(reas) + "\n\nANSWER: " + tgt
    pids = tok(prompt, add_special_tokens=False)["input_ids"]
    cids = tok(completion, add_special_tokens=False)["input_ids"] + [IM_END]
    ids = pids + cids
    if len(ids) > MAXLEN: continue
    examples.append({"input_ids": ids, "labels": [-100]*len(pids)+cids, "attention_mask": [1]*len(ids)})
print("examples", len(examples), flush=True)

class DS(torch.utils.data.Dataset):
    def __len__(self): return len(examples)
    def __getitem__(self, i): return examples[i]

pad_id = tok.pad_token_id or tok.eos_token_id
def collate(batch):
    m = max(len(b["input_ids"]) for b in batch)
    ii, ll, aa = [], [], []
    for b in batch:
        p = m-len(b["input_ids"])
        ii.append(b["input_ids"]+[pad_id]*p); ll.append(b["labels"]+[-100]*p); aa.append(b["attention_mask"]+[0]*p)
    return {"input_ids": torch.tensor(ii), "labels": torch.tensor(ll), "attention_mask": torch.tensor(aa)}

model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
model.config.use_cache = False
model.gradient_checkpointing_enable()
args = TrainingArguments(output_dir="/workspace/ckpt2", per_device_train_batch_size=8, gradient_accumulation_steps=2, max_steps=120, learning_rate=5e-6, lr_scheduler_type="constant", warmup_steps=5, logging_steps=20, save_strategy="no", bf16=True, report_to=[], remove_unused_columns=False)
Trainer(model=model, args=args, train_dataset=DS(), data_collator=collate).train()
model.config.use_cache = True
model.save_pretrained("/workspace/final_model2")
tok.save_pretrained("/workspace/final_model2")
gc = json.load(open("/workspace/final_model2/generation_config.json")) if os.path.exists("/workspace/final_model2/generation_config.json") else {}
gc["eos_token_id"] = [151645, 151643]
json.dump(gc, open("/workspace/final_model2/generation_config.json","w"), indent=2)
cfg = json.load(open("/workspace/final_model2/config.json")); cfg["eos_token_id"] = 151645
json.dump(cfg, open("/workspace/final_model2/config.json","w"), indent=2)
print("DONE2", flush=True)
