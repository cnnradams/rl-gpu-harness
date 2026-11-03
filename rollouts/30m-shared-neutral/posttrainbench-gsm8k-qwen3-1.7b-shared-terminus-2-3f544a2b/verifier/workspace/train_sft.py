import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from torch.utils.data import Dataset

MODEL = 'Qwen/Qwen3-1.7B-Base'
OUT = '/workspace/final_model'
TEMPLATE_PATH = '/workspace/templates/qwen3.jinja'

MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

tok = AutoTokenizer.from_pretrained(MODEL)
with open(TEMPLATE_PATH) as f:
    chat_template = f.read()

delimiter = '####'

def to_sample(rec):
    q = rec['question']
    parts = rec['answer'].split(delimiter)
    target = parts.pop().strip()
    reasoning = delimiter.join(parts).strip()
    return q, reasoning, target

def fewshot_str(q, reasoning, target):
    return f"{q}\n\nReasoning:\n{reasoning}\n\nANSWER: {target}"

train = load_dataset('openai/gsm8k', 'main', split='train')
fewshots = train.shuffle(seed=42).select(range(10))
system_msg = "\n\n".join(fewshot_str(*to_sample(r)) for r in fewshots)

def build_prompt(q, with_system):
    user = MATH_PROMPT_TEMPLATE.format(prompt=q)
    msgs = ([{'role': 'system', 'content': system_msg}] if with_system else []) + [{'role': 'user', 'content': user}]
    return tok.apply_chat_template(msgs, chat_template=chat_template, tokenize=False, add_generation_prompt=True)

examples = []
maxlen = 0
for i, rec in enumerate(train):
    q, reasoning, target = to_sample(rec)
    with_system = (i % 4 == 0)
    prompt = build_prompt(q, with_system)
    completion = reasoning + f"\n\nANSWER: {target}<|im_end|>"
    p_ids = tok(prompt, add_special_tokens=False)['input_ids']
    c_ids = tok(completion, add_special_tokens=False)['input_ids']
    input_ids = p_ids + c_ids
    maxlen = max(maxlen, len(input_ids))
    labels = [-100] * len(p_ids) + c_ids
    examples.append({'input_ids': input_ids, 'labels': labels})
print('num examples:', len(examples), 'max len:', maxlen, flush=True)

class Packed(Dataset):
    def __init__(self, examples, seq_len):
        self.seq_len = seq_len
        self.blocks = []
        cur_ids, cur_lab = [], []
        dropped = 0
        for ex in examples:
            ids, labs = ex['input_ids'], ex['labels']
            if len(ids) > seq_len:
                dropped += 1
                continue
            if len(cur_ids) + len(ids) > seq_len:
                self._flush(cur_ids, cur_lab)
                cur_ids, cur_lab = [], []
            cur_ids += ids
            cur_lab += labs
        self._flush(cur_ids, cur_lab)
        print('dropped:', dropped, flush=True)
    def _flush(self, ids, labs):
        if not ids:
            return
        pad = self.seq_len - len(ids)
        ids = ids + [tok.pad_token_id] * pad
        labs = labs + [-100] * pad
        self.blocks.append({'input_ids': ids, 'labels': labs})
    def __len__(self):
        return len(self.blocks)
    def __getitem__(self, i):
        b = self.blocks[i]
        return {'input_ids': torch.tensor(b['input_ids']), 'labels': torch.tensor(b['labels'])}

ds = Packed(examples, 3072)
print('packed blocks:', len(ds), flush=True)

model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation='sdpa')
model.config.use_cache = False

args = TrainingArguments(
    output_dir='/workspace/ckpt',
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    num_train_epochs=2,
    learning_rate=2e-5,
    lr_scheduler_type='cosine',
    warmup_ratio=0.03,
    weight_decay=0.0,
    logging_steps=10,
    save_strategy="epoch",
    save_only_model=True,
    bf16=True,
    report_to=[],
    seed=42,
)

trainer = Trainer(model=model, args=args, train_dataset=ds)
trainer.train()

model.config.use_cache = True
model.save_pretrained(OUT)
tok.save_pretrained(OUT)
print('SAVED to', OUT, flush=True)
