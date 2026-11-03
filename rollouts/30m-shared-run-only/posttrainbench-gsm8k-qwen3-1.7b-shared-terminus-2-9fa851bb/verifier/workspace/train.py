import json, os, torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig

BASE = "Qwen/Qwen3-1.7B-Base"
OUT = "/workspace/final_model_v2"

tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
model.config.use_cache = False

ds = load_dataset("json", data_files="/workspace/train_data.jsonl", split="train")

cfg = SFTConfig(
    output_dir="/workspace/ckpt",
    per_device_train_batch_size=16,
    gradient_accumulation_steps=2,
    learning_rate=1e-5,
    num_train_epochs=3,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    logging_steps=20,
    save_strategy="no",
    bf16=True,
    max_length=1024,
    packing=False,
    report_to=[],
    dataset_num_proc=8,
    seed=42,
)

trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds, processing_class=tok)
trainer.train()

trainer.save_model(OUT)
tok.save_pretrained(OUT)
with open(os.path.join(OUT, "generation_config.json"), "w") as f:
    json.dump({"temperature": None, "top_p": None, "top_k": None, "do_sample": False,
               "max_new_tokens": 512, "eos_token_id": [151645, 151643], "pad_token_id": 151643,
               "transformers_version": "4.57.3"}, f, indent=2)
print("DONE_SAVING")
