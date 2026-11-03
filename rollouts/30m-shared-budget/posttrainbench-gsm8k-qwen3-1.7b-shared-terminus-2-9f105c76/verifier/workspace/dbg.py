from datasets import load_dataset
from transformers import AutoTokenizer
MODEL = "Qwen/Qwen3-1.7B-Base"
tok = AutoTokenizer.from_pretrained(MODEL)
chat_template = open("/workspace/templates/qwen3.jinja").read()
ds = load_dataset("openai/gsm8k", "main", split="train")
fs = ds.shuffle(seed=42).select(range(10))
def fewshot_text(ex):
    reasoning, target = ex["answer"].split("####")
    return f"{ex['question']}\n\nReasoning:\n{reasoning.strip()}\n\nANSWER: {target.strip()}"
SYS = "\n\n".join(fewshot_text(e) for e in fs)
msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": "QUESTION HERE"}]
prompt = tok.apply_chat_template(msgs, chat_template=chat_template, tokenize=False, add_generation_prompt=True)
print(repr(prompt[:200]))
p = tok(prompt, add_special_tokens=False)["input_ids"]
print("prompt len:", len(p))
