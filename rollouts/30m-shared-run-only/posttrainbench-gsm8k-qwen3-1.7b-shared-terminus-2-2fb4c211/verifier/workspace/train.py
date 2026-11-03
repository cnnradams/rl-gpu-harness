import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import json
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig

tok = AutoTokenizer.from_pretrained("/workspace/base_model")
with open("/workspace/templates/qwen3.jinja") as f:
    chat_template = f.read()

rows = []
with open("/workspace/train_data.jsonl") as f:
    for line in f:
        d = json.loads(line)
        msgs = d["messages"]
        prompt_msgs = [m for m in msgs if m["role"] != "assistant"]
        assistant = [m for m in msgs if m["role"] == "assistant"][0]["content"]
        prompt = tok.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True, chat_template=chat_template)
        completion = assistant + "<|im_end|>"
        rows.append({"prompt": prompt, "completion": completion})
print("examples:", len(rows))
ds = Dataset.from_list(rows)

model = AutoModelForCausalLM.from_pretrained("/workspace/base_model", dtype=torch.bfloat16, attn_implementation="flash_attention_2")

cfg = SFTConfig(
    output_dir="/workspace/ckpt_v1",
    num_train_epochs=2,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    learning_rate=1e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    logging_steps=10,
    save_strategy="steps",
    save_steps=150,
    save_total_limit=4,
    bf16=True,
    max_length=4096,
    packing=False,
    group_by_length=True,
    completion_only_loss=True,
    eos_token="<|im_end|>",
    report_to=[],
    seed=0,
)

trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds, processing_class=tok)
trainer.train()
trainer.save_model("/workspace/model_v1")
tok.save_pretrained("/workspace/model_v1")
print("ALL_DONE")
