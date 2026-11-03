import re, torch, random
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig

MODEL = "Qwen/Qwen3-1.7B-Base"
tok = AutoTokenizer.from_pretrained(MODEL)
tok.pad_token = "<|endoftext|>"
chat_template = open("/workspace/templates/qwen3.jinja").read()

MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

ds = load_dataset("openai/gsm8k", "main", split="train").shuffle(seed=42)

def split_ans(a):
    parts = a.split("####")
    target = parts[-1].strip()
    reasoning = "####".join(parts[:-1]).strip()
    return reasoning, target

def fewshot_block(rec):
    reasoning, target = split_ans(rec["answer"])
    return f"{rec['question']}\n\nReasoning:\n{reasoning}\n\nANSWER: {target}"

sys_msg = "\n\n".join(fewshot_block(r) for r in ds.select(range(10)))

def clean(r):
    return re.sub(r"<<[^>]*>>", "", r)

rows = []
N = 6500
for i in range(N):
    rec = ds[i]
    reasoning, target = split_ans(rec["answer"])
    reasoning = clean(reasoning)
    user = MATH_PROMPT_TEMPLATE.format(prompt=rec["question"])
    if i % 2 == 0:
        msgs = [{"role": "system", "content": sys_msg}, {"role": "user", "content": user}]
    else:
        msgs = [{"role": "user", "content": user}]
    prompt = tok.apply_chat_template(msgs, chat_template=chat_template, tokenize=False, add_generation_prompt=True)
    completion = reasoning + f"\n\nANSWER: {target}<|im_end|>"
    rows.append({"prompt": prompt, "completion": completion})

train_ds = Dataset.from_list(rows)

model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
model.config.use_cache = False

cfg = SFTConfig(
    output_dir="/workspace/sft_out",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    learning_rate=1e-5,
    num_train_epochs=1,
    lr_scheduler_type="cosine",
    warmup_steps=15,
    logging_steps=20,
    save_strategy="no",
    bf16=True,
    max_length=2048,
    group_by_length=True,
    report_to=[],
    dataset_num_proc=8,
    completion_only_loss=True,
    seed=42,
)
trainer = SFTTrainer(model=model, args=cfg, train_dataset=train_ds, processing_class=tok)
trainer.train()

IM_END = 151645
model.config.eos_token_id = IM_END
tok.eos_token = "<|im_end|>"
model.generation_config.eos_token_id = [IM_END, 151643]
model.generation_config.pad_token_id = 151643
model.save_pretrained("/workspace/final_model")
tok.save_pretrained("/workspace/final_model")
print("DONE_TRAINING")
