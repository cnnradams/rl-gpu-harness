import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig

MODEL = "Qwen/Qwen3-1.7B-Base"
tok = AutoTokenizer.from_pretrained(MODEL)
with open("/workspace/templates/qwen3.jinja") as f:
    tok.chat_template = f.read()

ds = load_dataset("json", data_files="/workspace/work/sft_data.jsonl", split="train").select(range(1500))

def to_text(ex):
    return {"text": tok.apply_chat_template(ex["messages"], tokenize=False)}
ds = ds.map(to_text)

model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation="eager")
model.config.use_cache = False

cfg = SFTConfig(
    output_dir="/workspace/work/ckpt2",
    num_train_epochs=1,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=2,
    learning_rate=5e-6,
    lr_scheduler_type="constant",
    logging_steps=10,
    save_strategy="no",
    bf16=True,
    max_length=1024,
    report_to=[],
    seed=42,
    dataset_text_field="text",
)

trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds, processing_class=tok)
trainer.train()
trainer.save_model("/workspace/work/model2")
tok.save_pretrained("/workspace/work/model2")
print("DONE2")
