import os, json, random
os.environ['HF_HOME']='/workspace/hf_cache'
from datasets import load_dataset
from transformers import AutoTokenizer

ds = load_dataset('openai/gsm8k','main',split='train')
tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-1.7B-Base')
with open('/workspace/templates/qwen3.jinja') as f:
    chat_template = f.read()

def fmt(r):
    return r.replace('\n',' ').strip()

def ans_of(a):
    return a.split('####')[1].strip()

def reas_of(a):
    return fmt(a.split('####')[0])

fewshot = list(ds.shuffle(seed=42).select(range(10)))
system_msg = "\n\n".join([f"{ex['question']}\n\nReasoning:\n{reas_of(ex['answer'])}\n\nANSWER: {ans_of(ex['answer'])}" for ex in fewshot])

PROMPT = '''Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:'''

rows=[]
for ex in ds:
    msgs=[{'role':'system','content':system_msg},{'role':'user','content':PROMPT.format(prompt=ex['question'])}]
    pt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, chat_template=chat_template)
    ct = reas_of(ex['answer']) + "\n\nANSWER: " + ans_of(ex['answer']) + "<|im_end|>\n"
    rows.append({'prompt':pt,'completion':ct})
random.seed(0); random.shuffle(rows)
with open('/workspace/train_data.jsonl','w') as f:
    for r in rows: f.write(json.dumps(r)+'\n')
with open('/workspace/train_questions.txt','w') as f:
    for ex in ds: f.write(ex['question'].replace('\n',' ')+'\n')
print('N',len(rows))
