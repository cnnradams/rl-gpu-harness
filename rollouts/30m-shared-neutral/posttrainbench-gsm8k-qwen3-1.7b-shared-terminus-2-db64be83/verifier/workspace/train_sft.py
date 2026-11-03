import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig

MODEL = "Qwen/Qwen3-1.7B-Base"
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation="flash_attention_2")

ds = load_dataset("json", data_files="/workspace/train.jsonl", split="train")

cfg = SFTConfig(
    output_dir="/workspace/sft_out",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    learning_rate=2e-5,
    num_train_epochs=1, max_steps=150,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    logging_steps=20,
    save_strategy="no",
    bf16=True,
    max_length=2048,
    packing=False,
    gradient_checkpointing=True,
    report_to=[],
    seed=42,
)
trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds, processing_class=tok)
trainer.train()
trainer.save_model("/workspace/final_model")
tok.save_pretrained("/workspace/final_model")
print("DONE_TRAIN")
