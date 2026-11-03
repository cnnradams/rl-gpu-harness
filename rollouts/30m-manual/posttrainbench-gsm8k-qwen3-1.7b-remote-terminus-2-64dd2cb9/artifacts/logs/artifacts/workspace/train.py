import re, os, json, torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments

MODEL = "Qwen/Qwen3-1.7B-Base"
OUT = "/workspace/final_model"

tok = AutoTokenizer.from_pretrained(MODEL)
chat_template = open("/workspace/templates/qwen3.jinja").read()

from inspect_ai.dataset import hf_dataset
from inspect_evals.gsm8k.gsm8k import sample_to_fewshot, record_to_sample, MATH_PROMPT_TEMPLATE
fewshots = hf_dataset(path="openai/gsm8k", data_dir="main", split="train",
                      sample_fields=record_to_sample, shuffle=True, seed=42, limit=10)
SYS = "\n\n".join(sample_to_fewshot(s) for s in fewshots)
print("SYSMSG_HEAD:", SYS[:300])

def build(ex):
    sol, final = ex["answer"].split("####")
    sol = re.sub(r"<<[^>]*>>", "", sol).strip()
    final = final.strip().replace(",", "")
    user = MATH_PROMPT_TEMPLATE.format(prompt=ex["question"])
    msgs = [{"role":"system","content":SYS},{"role":"user","content":user}]
    p = tok.apply_chat_template(msgs, chat_template=chat_template, tokenize=False, add_generation_prompt=True)
    c = sol + "\nANSWER: " + final + "<|im_end|>"
    return p, c

ds = load_dataset("openai/gsm8k", "main", split="train")
data = []
skipped = 0
for ex in ds:
    p, c = build(ex)
    pid = tok(p, add_special_tokens=False).input_ids
    cid = tok(c, add_special_tokens=False).input_ids
    ids = pid + cid
    if len(ids) > 4096:
        skipped += 1
        continue
    labels = [-100]*len(pid) + cid
    data.append({"input_ids": ids, "labels": labels})
print("examples:", len(data), "skipped:", skipped)

class Collator:
    def __call__(self, feats):
        maxlen = max(len(f["input_ids"]) for f in feats)
        pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        input_ids, labels, attn = [], [], []
        for f in feats:
            n = maxlen - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [pad]*n)
            labels.append(f["labels"] + [-100]*n)
            attn.append([1]*len(f["input_ids"]) + [0]*n)
        return {"input_ids": torch.tensor(input_ids), "labels": torch.tensor(labels),
                "attention_mask": torch.tensor(attn)}

model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
                                             attn_implementation="flash_attention_2")
model.config.use_cache = False

args = TrainingArguments(
    output_dir="/workspace/ckpt",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=1e-5,
    num_train_epochs=1,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    logging_steps=25,
    save_strategy="no",
    bf16=True,
    report_to=[],
    dataloader_num_workers=2,
    seed=42,
)
trainer = Trainer(model=model, args=args, train_dataset=data, data_collator=Collator())
trainer.train()
model.config.use_cache = True
trainer.save_model(OUT)
tok.save_pretrained(OUT)
with open(os.path.join(OUT, "generation_config.json"), "w") as f:
    json.dump({"eos_token_id": [151645, 151643], "pad_token_id": 151643,
               "do_sample": False, "temperature": None, "top_p": None, "top_k": None}, f, indent=2)
print("DONE_TRAIN")
