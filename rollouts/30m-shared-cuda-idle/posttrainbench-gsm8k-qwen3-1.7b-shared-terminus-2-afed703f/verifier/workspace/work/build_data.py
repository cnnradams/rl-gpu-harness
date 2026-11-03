import json, re
from datasets import load_dataset
from inspect_ai.dataset import hf_dataset
from inspect_evals.gsm8k.gsm8k import record_to_sample, sample_to_fewshot

MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

fewshots = hf_dataset(path="openai/gsm8k", data_dir="main", split="train", sample_fields=record_to_sample, shuffle=True, seed=42, limit=10)
sys_msg = "\n\n".join(sample_to_fewshot(s) for s in fewshots)
open('/workspace/work/system_message.txt','w').write(sys_msg)

ds = load_dataset('openai/gsm8k','main')['train']
def clean(ans):
    parts = ans.split('####')
    target = parts[-1].strip().replace(',','')
    reasoning = '####'.join(parts[:-1])
    reasoning = re.sub(r'<<[^>]*>>','',reasoning).strip()
    return reasoning, target

out, docs = [], []
for ex in ds:
    r,t = clean(ex['answer'])
    user = MATH_PROMPT_TEMPLATE.format(prompt=ex['question'])
    asst = r + "\n\nANSWER: " + t
    text = f"<|im_start|>system\n{sys_msg}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n{asst}<|im_end|>"
    out.append({'text': text})
    docs.append({'text': ex['question'] + '\n' + r + '\n' + t})

with open('/workspace/work/train_text.jsonl','w') as f:
    for o in out: f.write(json.dumps(o)+'\n')
with open('/workspace/work/docs.jsonl','w') as f:
    for d in docs: f.write(json.dumps(d)+'\n')
print(len(out), 'examples built')
