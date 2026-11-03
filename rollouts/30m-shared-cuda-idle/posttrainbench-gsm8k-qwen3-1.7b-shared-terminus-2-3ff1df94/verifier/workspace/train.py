import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

model_name = "Qwen/Qwen3-1.7B-Base"
ds = load_dataset("json", data_files="/workspace/train_data.jsonl", split="train")
tok = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
cfg = SFTConfig(
    output_dir="/workspace/ckpt",
    per_device_train_batch_size=16,
    gradient_accumulation_steps=2,
    num_train_epochs=1,
    learning_rate=1e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    logging_steps=20,
    save_strategy="no",
    bf16=True,
    max_length=1024,
    dataset_text_field="text",
    packing=False,
    report_to=[],
)
trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds, processing_class=tok)
trainer.train()
trainer.save_model("/workspace/final_model")
tok.save_pretrained("/workspace/final_model")
print("DONE")
