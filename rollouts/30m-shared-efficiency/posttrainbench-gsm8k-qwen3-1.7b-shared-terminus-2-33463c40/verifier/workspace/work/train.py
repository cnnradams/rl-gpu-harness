import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig

MODEL = "Qwen/Qwen3-1.7B-Base"

tok = AutoTokenizer.from_pretrained(MODEL)
with open("/workspace/templates/qwen3.jinja") as f:
    tok.chat_template = f.read()

ds = load_dataset("json", data_files="/workspace/work/sft_data.jsonl", split="train")

model = AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.bfloat16, attn_implementation="flash_attention_2"
)
model.config.use_cache = False

cfg = SFTConfig(
    output_dir="/workspace/work/ckpt",
    num_train_epochs=2,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=2,
    learning_rate=1e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    logging_steps=10,
    save_strategy="no",
    bf16=True,
    max_length=1024,
    packing=False,
    completion_only_loss=True,
    report_to=[],
    seed=42,
)

trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds, processing_class=tok)
trainer.train()
trainer.save_model("/workspace/work/model_out")
tok.save_pretrained("/workspace/work/model_out")
print("DONE")
