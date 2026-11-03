import os, json, random
import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments

MODEL = "Qwen/Qwen3-1.7B-Base"
OUT = "/workspace/final_model"

tok = AutoTokenizer.from_pretrained(MODEL)
ds = load_dataset("openai/gsm8k", "main", split="train")

few = ds.shuffle(seed=42).select(range(10))
def fs(ex):
    r, t = ex["answer"].split("####")
    return f"{ex['question']}\n\nReasoning:\n{r.strip()}\n\nANSWER: {t.strip()}"
sysmsg = "\n\n".join(fs(e) for e in few)

TEMPLATE = (
"Solve the following math problem step by step. The last line of your response should be of the form \"ANSWER: $ANSWER\" (without quotes) where $ANSWER is the answer to the problem.\n\n"
"{q}\n\n"
"Remember to put your answer on its own line at the end in the form \"ANSWER: $ANSWER\" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.\n\n"
"Reasoning:")

def build(ex):
    user = TEMPLATE.format(q=ex["question"])
    prompt = f"<|im_start|>system\n{sysmsg}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"
    r, t = ex["answer"].split("####")
    completion = f"{r.strip()}\n\nANSWER: {t.strip()}<|im_end|>"
    return prompt, completion

MAXLEN = 3072
examples = []
for ex in ds:
    p, c = build(ex)
    pi = tok(p, add_special_tokens=False).input_ids
    ci = tok(c, add_special_tokens=False).input_ids
    ids = pi + ci
    if len(ids) > MAXLEN:
        continue
    labels = [-100]*len(pi) + list(ci)
    examples.append({"input_ids": ids, "labels": labels})
print("num examples:", len(examples), flush=True)
random.seed(0)
random.shuffle(examples); examples = examples[:3000]

class Collator:
    def __call__(self, batch):
        maxlen = max(len(b["input_ids"]) for b in batch)
        pad = 151643
        input_ids, labels, attn = [], [], []
        for b in batch:
            n = maxlen - len(b["input_ids"])
            input_ids.append(list(b["input_ids"]) + [pad]*n)
            labels.append(list(b["labels"]) + [-100]*n)
            attn.append([1]*len(b["input_ids"]) + [0]*n)
        return {"input_ids": torch.tensor(input_ids), "labels": torch.tensor(labels), "attention_mask": torch.tensor(attn)}

train_ds = Dataset.from_list(examples)

model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
model.config.use_cache = False

args = TrainingArguments(
    output_dir="/workspace/ckpt",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    learning_rate=1e-5,
    lr_scheduler_type="cosine",
    warmup_steps=15,
    logging_steps=10,
    save_strategy="no",
    bf16=True,
    report_to=[], remove_unused_columns=False, gradient_checkpointing=True,
    dataloader_num_workers=4,
)

trainer = Trainer(model=model, args=args, train_dataset=train_ds, data_collator=Collator())
trainer.train()

model.config.use_cache = True
model.save_pretrained(OUT)
tok.save_pretrained(OUT)
gc_path = os.path.join(OUT, "generation_config.json")
gc = {}
if os.path.exists(gc_path):
    gc = json.load(open(gc_path))
gc["eos_token_id"] = [151645, 151643]
gc["pad_token_id"] = 151643
json.dump(gc, open(gc_path, "w"), indent=2)
print("DONE_TRAINING_SAVED", flush=True)
