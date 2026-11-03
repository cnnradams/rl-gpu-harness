import json, random
from datasets import load_dataset
from inspect_evals.gsm8k.gsm8k import MATH_PROMPT_TEMPLATE, sample_to_fewshot, record_to_sample

ds = load_dataset('openai/gsm8k', 'main')
train = ds['train']
# replicate the exact fewshot system message used by the eval (seed 42, 10 shots)
few = train.shuffle(seed=42).select(range(10))
sys_msgs = []
for r in few:
    s = record_to_sample(r)
    sys_msgs.append(sample_to_fewshot(s))
SYS = "\n\n".join(sys_msgs)

excl = set(few['question'])
out = []
for r in train:
    if r['question'] in excl:
        continue
    ans = r['answer']
    reasoning, final = ans.split('####')
    reasoning = reasoning.strip()
    final = final.strip()
    user = MATH_PROMPT_TEMPLATE.format(prompt=r['question'])
    assistant = reasoning + "\n\nANSWER: " + final
    text = ("<|im_start|>system\n" + SYS + "<|im_end|>\n"
            + "<|im_start|>user\n" + user + "<|im_end|>\n"
            + "<|im_start|>assistant\n" + assistant + "<|im_end|>")
    out.append({"text": text})

random.seed(0)
with open('train.jsonl', 'w') as f:
    for o in out:
        f.write(json.dumps(o) + "\n")
print("wrote", len(out), "examples")
