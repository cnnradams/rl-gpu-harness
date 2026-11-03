import random, re, json, os
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, default_data_collator

BASE = 'Qwen/Qwen3-1.7B-Base'
OUT = '/workspace/final_model'
MAXLEN = 1600
random.seed(0)

tok = AutoTokenizer.from_pretrained(BASE)
IM_END = tok.convert_tokens_to_ids('<|im_end|>')
print('im_end id', IM_END, 'eos', tok.eos_token_id)

TMPL = """Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:"""

ds = load_dataset('openai/gsm8k', 'main', split='train')
records = []
for r in ds:
    parts = r['answer'].split('####')
    target = parts[-1].strip()
    reasoning = '####'.join(parts[:-1]).strip()
    records.append({'q': r['question'].strip(), 'reasoning': reasoning, 'target': target})

def fewshot_block(exclude_idx):
    idxs = random.sample(range(len(records)), 10)
    blocks = []
    for i in idxs:
        if i == exclude_idx:
            continue
        rr = records[i]
        blocks.append(f"{rr['q']}\n\nReasoning:\n{rr['reasoning']}\n\nANSWER: {rr['target']}")
    return '\n\n'.join(blocks)

examples = []
for i, r in enumerate(records):
    user = TMPL.format(prompt=r['q'])
    if random.random() < 0.5:
        sys = fewshot_block(i)
        prompt = f"<|im_start|>system\n{sys}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"
    else:
        prompt = f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"
    completion = f"{r['reasoning']}\n\nANSWER: {r['target']}"
    pids = tok(prompt, add_special_tokens=False)['input_ids']
    cids = tok(completion, add_special_tokens=False)['input_ids'] + [IM_END]
    ids = pids + cids
    if len(ids) > MAXLEN:
        continue
    labels = [-100]*len(pids) + cids
    examples.append({'input_ids': ids, 'labels': labels})

print('num examples', len(examples))
print(tok.decode(examples[0]['input_ids'][-60:]))

model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16, attn_implementation='sdpa')
model.config.use_cache = False
model.gradient_checkpointing_enable()
model.cuda()

args = TrainingArguments(
    output_dir='/workspace/ckpt',
    per_device_train_batch_size=4,
    gradient_accumulation_steps=16,
    num_train_epochs=1,
    learning_rate=1e-5,
    lr_scheduler_type='cosine',
    warmup_steps=10,
    logging_steps=10,
    save_strategy='no',
    bf16=True,
    report_to=[],
    group_by_length=True,
    optim='adamw_torch_fused',
    seed=0,
)

def collate(features):
    maxlen = max(len(f['input_ids']) for f in features)
    input_ids, labels, attn = [], [], []
    for f in features:
        ids = f['input_ids']; lab = f['labels']
        pad = maxlen - len(ids)
        input_ids.append(ids + [151643]*pad)
        labels.append(lab + [-100]*pad)
        attn.append([1]*len(ids) + [0]*pad)
    return {'input_ids': torch.tensor(input_ids), 'labels': torch.tensor(labels), 'attention_mask': torch.tensor(attn)}

trainer = Trainer(model=model, args=args, train_dataset=examples, data_collator=collate)
trainer.train()

model.config.use_cache = True
model.config.eos_token_id = IM_END
tok.eos_token = '<|im_end|>'
model.save_pretrained(OUT, safe_serialization=True)
tok.save_pretrained(OUT)
with open(os.path.join(OUT, 'generation_config.json'), 'w') as f:
    json.dump({'temperature': 0.0, 'eos_token_id': IM_END, 'pad_token_id': 151643, 'bos_token_id': None}, f, indent=2)
print('saved to', OUT)
