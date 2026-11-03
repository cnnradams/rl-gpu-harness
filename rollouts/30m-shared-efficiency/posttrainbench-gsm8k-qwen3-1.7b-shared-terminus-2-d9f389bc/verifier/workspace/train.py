import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"]="expandable_segments:True"
import torch, json
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig

model_name = "Qwen/Qwen3-1.7B-Base"
tok = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
model.config.use_cache = False

ds = load_dataset("json", data_files="/workspace/train_data.jsonl", split="train")

cfg = SFTConfig(
    output_dir="/workspace/ckpt",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    learning_rate=1e-5,
    num_train_epochs=1,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    logging_steps=10,
    save_strategy="no",
    bf16=True,
    max_length=2816,
    report_to=[],
    packing=False,
    seed=42,
)
trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds, processing_class=tok)
trainer.train()
trainer.save_model("/workspace/final_model")
tok.save_pretrained("/workspace/final_model")

# fix eos so vllm stops at <|im_end|>
with open("/workspace/final_model/config.json") as f:
    c = json.load(f)
c["eos_token_id"] = 151645
with open("/workspace/final_model/config.json", "w") as f:
    json.dump(c, f, indent=2)
import os
p = "/workspace/final_model/generation_config.json"
gen = {}
if os.path.exists(p):
    with open(p) as f:
        gen = json.load(f)
gen["eos_token_id"] = [151645, 151643]
gen["pad_token_id"] = 151643
with open(p, "w") as f:
    json.dump(gen, f, indent=2)
print("DONE")
