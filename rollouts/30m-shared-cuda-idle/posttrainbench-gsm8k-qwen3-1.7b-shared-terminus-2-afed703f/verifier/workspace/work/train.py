import os, json, argparse, torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig

p = argparse.ArgumentParser()
p.add_argument('--output', default='/workspace/work/ckpt1')
p.add_argument('--epochs', type=float, default=2.0)
p.add_argument('--lr', type=float, default=1e-5)
args = p.parse_args()

model_id = '/workspace/final_model'
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, attn_implementation='flash_attention_2')
model.config.use_cache = False

rows = [json.loads(l) for l in open('/workspace/work/train_text.jsonl')]
ds = Dataset.from_list(rows)

cfg = SFTConfig(
    output_dir=args.output + '_out',
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    learning_rate=args.lr,
    num_train_epochs=args.epochs,
    lr_scheduler_type='cosine',
    warmup_ratio=0.03,
    logging_steps=20,
    save_strategy='no',
    save_steps=400,
    save_total_limit=1,
    bf16=True,
    max_length=1024,
    dataset_text_field='text',
    packing=False,
    group_by_length=True,
    report_to=[],
    seed=42,
)
trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds, processing_class=tok)
trainer.train()
model.config.use_cache = True
model.save_pretrained(args.output)
tok.save_pretrained(args.output)
gc = json.load(open(os.path.join(args.output, 'generation_config.json')))
gc['eos_token_id'] = [151645, 151643]
json.dump(gc, open(os.path.join(args.output, 'generation_config.json'), 'w'), indent=2)
print('DONE TRAINING')
