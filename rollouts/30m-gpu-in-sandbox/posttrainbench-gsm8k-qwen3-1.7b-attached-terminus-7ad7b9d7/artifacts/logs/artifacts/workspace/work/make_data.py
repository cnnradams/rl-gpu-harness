import json
from datasets import load_dataset
from transformers import AutoTokenizer
from jinja2 import Template

MODEL = 'Qwen/Qwen3-1.7B-Base'
tok = AutoTokenizer.from_pretrained(MODEL)
chat_template = open('/workspace/templates/qwen3.jinja').read()

MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

ds = load_dataset('openai/gsm8k', 'main')['train']
fewshots = ds.shuffle(seed=42).select(range(10))

def to_fewshot(rec):
    ans = rec['answer'].split('####')
    target = ans[-1].strip()
    reasoning = '####'.join(ans[:-1]).strip()
    return f"{rec['question']}\n\nReasoning:\n{reasoning}\n\nANSWER: {target}"

system_msg = "\n\n".join([to_fewshot(r) for r in fewshots])

with open('/workspace/work/train.jsonl', 'w') as f:
    for rec in ds:
        ans = rec['answer'].split('####')
        target = ans[-1].strip()
        reasoning = '####'.join(ans[:-1]).strip()
        user = MATH_PROMPT_TEMPLATE.format(prompt=rec['question'])
        msgs = [{'role': 'system', 'content': system_msg}, {'role': 'user', 'content': user}]
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, chat_template=chat_template)
        completion = reasoning + f"\n\nANSWER: {target}" + "<|im_end|>"
        f.write(json.dumps({'prompt': prompt, 'completion': completion}) + '\n')
print('done', len(ds))
