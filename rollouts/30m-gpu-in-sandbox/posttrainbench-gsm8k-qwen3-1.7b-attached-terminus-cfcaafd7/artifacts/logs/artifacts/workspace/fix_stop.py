import json, os, torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

BASE = '/workspace/final_model'
OUT = '/workspace/final_model2'
MAXLEN = 1024
tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-1.7B-Base')
EOT = 151643

TMPL = """Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:"""

ds = load_dataset('openai/gsm8k', 'main', split='train')
examples = []
for r in ds:
    parts = r['answer'].split('####')
    target = parts[-1].strip()
    reasoning = '####'.join(parts[:-1]).strip()
    prompt = f"<|im_start|>user\n{TMPL.format(prompt=r['question'].strip())}<|im_end|>\n<|im_start|>assistant\n"
    completion = f"{reasoning}\n\nANSWER: {target}"
    pids = tok(prompt, add_special_tokens=False)['input_ids']
    cids = tok(completion, add_special_tokens=False)['input_ids'] + [EOT]
    if len(pids)+len(cids) > MAXLEN:
        continue
    examples.append({'input_ids': pids+cids, 'labels': [-100]*len(pids)+cids})
print('n', len(examples))

def collate(features):
    m = max(len(f['input_ids']) for f in features)
    ii, ll, aa = [], [], []
    for f in features:
        pad = m-len(f['input_ids'])
        ii.append(f['input_ids']+[EOT]*pad); ll.append(f['labels']+[-100]*pad); aa.append([1]*len(f['input_ids'])+[0]*pad)
    return {'input_ids': torch.tensor(ii), 'labels': torch.tensor(ll), 'attention_mask': torch.tensor(aa)}

model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16)
model.config.use_cache = False
model.gradient_checkpointing_enable()
model.cuda()
args = TrainingArguments(output_dir='/workspace/ckpt2', per_device_train_batch_size=8, gradient_accumulation_steps=8, max_steps=120, learning_rate=1e-5, lr_scheduler_type='cosine', warmup_steps=5, logging_steps=20, save_strategy='no', bf16=True, report_to=[], group_by_length=True, optim='adamw_torch_fused', seed=0)
Trainer(model=model, args=args, train_dataset=examples, data_collator=collate).train()
model.config.use_cache = True
model.config.eos_token_id = EOT
model.save_pretrained(OUT, safe_serialization=True)
tok.eos_token = '<|endoftext|>'
tok.save_pretrained(OUT)
json.dump({'eos_token_id': EOT, 'pad_token_id': EOT, 'do_sample': False}, open(os.path.join(OUT,'generation_config.json'),'w'), indent=2)
print('saved', OUT)
