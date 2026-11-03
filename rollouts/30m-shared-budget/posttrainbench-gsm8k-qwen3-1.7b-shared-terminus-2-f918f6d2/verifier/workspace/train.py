import os
os.environ['HF_HOME']='/workspace/hf_cache'
import torch
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name='Qwen/Qwen3-1.7B-Base'
tok=AutoTokenizer.from_pretrained(model_name)
model=AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16, attn_implementation='flash_attention_2')
ds=load_dataset('json', data_files='/workspace/train_data.jsonl', split='train')
cfg=SFTConfig(
  output_dir='/workspace/ckpt',
  per_device_train_batch_size=4,
  gradient_accumulation_steps=8,
  learning_rate=1e-5,
  max_steps=165,
  lr_scheduler_type='cosine',
  warmup_ratio=0.03,
  logging_steps=10,
  save_strategy='no',
  bf16=True,
  max_length=4096,
  completion_only_loss=True,
  gradient_checkpointing=False,
  report_to=[],
)
trainer=SFTTrainer(model=model, args=cfg, train_dataset=ds, processing_class=tok)
trainer.train()
trainer.save_model('/workspace/final_model')
tok.save_pretrained('/workspace/final_model')
print('TRAIN_DONE')
