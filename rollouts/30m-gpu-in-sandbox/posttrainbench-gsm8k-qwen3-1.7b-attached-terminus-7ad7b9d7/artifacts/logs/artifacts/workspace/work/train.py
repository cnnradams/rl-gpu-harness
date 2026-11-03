import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig

MODEL = 'Qwen/Qwen3-1.7B-Base'
tok = AutoTokenizer.from_pretrained(MODEL)
ds = load_dataset('json', data_files='/workspace/work/train.jsonl', split='train')

model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32, attn_implementation='sdpa')
model.config.use_cache = False

cfg = SFTConfig(
    output_dir='/workspace/work/sft_out',
    num_train_epochs=2,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    learning_rate=1e-5,
    lr_scheduler_type='cosine',
    warmup_ratio=0.03,
    logging_steps=20,
    save_strategy='epoch',
    bf16=True,
    max_length=1024,
    report_to=[],
    seed=42,
)
trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds, processing_class=tok)
trainer.train()
trainer.save_model('/workspace/work/final_raw')
tok.save_pretrained('/workspace/work/final_raw')
print('TRAINING DONE')
