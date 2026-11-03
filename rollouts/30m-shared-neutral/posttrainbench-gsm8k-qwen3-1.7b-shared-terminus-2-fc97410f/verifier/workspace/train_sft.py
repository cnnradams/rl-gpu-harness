import torch, random, shutil, os
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

MODEL = "Qwen/Qwen3-1.7B-Base"
OUT = "/workspace/sft_out"

tok = AutoTokenizer.from_pretrained(MODEL)
tok.padding_side = "right"
if tok.pad_token is None:
    tok.pad_token = "<|endoftext|>"

IM_START = tok.convert_tokens_to_ids("<|im_start|>")
IM_END = tok.convert_tokens_to_ids("<|im_end|>")
NL = tok.encode("\n")

MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

def sample_to_fewshot(q, reasoning, target):
    return f"{q}\n\nReasoning:\n{reasoning}\n\nANSWER: {target}"

ds = load_dataset("openai/gsm8k", "main")["train"]
recs = list(ds)

rng = random.Random(42)
fs_idx = rng.sample(range(len(recs)), 10)
system_msg = "\n\n".join(
    sample_to_fewshot(recs[i]["question"], recs[i]["answer"].split("####")[0].strip(), recs[i]["answer"].split("####")[-1].strip())
    for i in fs_idx
)

def build(ex, with_system):
    parts = []
    if with_system:
        parts.append("<|im_start|>system\n" + system_msg + "<|im_end|>\n")
    parts.append("<|im_start|>user\n" + MATH_PROMPT_TEMPLATE.format(prompt=ex["question"]) + "<|im_end|>\n")
    prompt_str = "".join(parts) + "<|im_start|>assistant\n"
    ans = ex["answer"]
    reasoning = ans.split("####")[0].strip()
    final = ans.split("####")[-1].strip()
    target_str = reasoning + "\n\nANSWER: " + final
    return prompt_str, target_str

MAXLEN = 4096

def gen():
    for i, ex in enumerate(recs):
        with_system = (i % 5 != 0)
        p, t = build(ex, with_system)
        p_ids = tok(p, add_special_tokens=False)["input_ids"]
        t_ids = tok(t, add_special_tokens=False)["input_ids"] + [IM_END] + NL
        ids = p_ids + t_ids
        labels = [-100] * len(p_ids) + t_ids
        if len(ids) > MAXLEN:
            continue
        yield {"input_ids": ids, "labels": labels}

from datasets import Dataset
train_ds = Dataset.from_generator(gen)
print("train size:", len(train_ds))

def collate(batch):
    maxlen = max(len(b["input_ids"]) for b in batch)
    pad_id = tok.pad_token_id
    input_ids, labels, attn = [], [], []
    for b in batch:
        n = maxlen - len(b["input_ids"])
        input_ids.append(b["input_ids"] + [pad_id] * n)
        labels.append(b["labels"] + [-100] * n)
        attn.append([1] * len(b["input_ids"]) + [0] * n)
    return {
        "input_ids": torch.tensor(input_ids),
        "labels": torch.tensor(labels),
        "attention_mask": torch.tensor(attn),
    }

model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
model.config.use_cache = False

args = TrainingArguments(
    output_dir=OUT,
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=16,
    learning_rate=1e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    weight_decay=0.0,
    logging_steps=5,
    save_strategy="no",
    bf16=True,
    report_to=[],
    seed=42,
)

trainer = Trainer(model=model, args=args, train_dataset=train_ds, data_collator=collate)
trainer.train()

model.config.use_cache = True
model.save_pretrained(OUT)
tok.save_pretrained(OUT)

# ensure vllm stops at <|im_end|>
src = os.path.join(OUT, "generation_config.json")
gen_cfg = {"eos_token_id": IM_END, "pad_token_id": tok.pad_token_id, "do_sample": False}
import json
with open(src, "w") as f:
    json.dump(gen_cfg, f, indent=2)

# patch tokenizer_config eos
with open(os.path.join(OUT, "tokenizer_config.json")) as f:
    tc = json.load(f)
tc["eos_token"] = "<|im_end|>"
with open(os.path.join(OUT, "tokenizer_config.json"), "w") as f:
    json.dump(tc, f, indent=2)

with open(os.path.join(OUT, "config.json")) as f:
    cfg = json.load(f)
cfg["eos_token_id"] = IM_END
with open(os.path.join(OUT, "config.json"), "w") as f:
    json.dump(cfg, f, indent=2)
print("DONE")
