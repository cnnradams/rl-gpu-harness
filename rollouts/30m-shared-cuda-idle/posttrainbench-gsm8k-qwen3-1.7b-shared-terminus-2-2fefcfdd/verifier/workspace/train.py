import os
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from trl import SFTTrainer, SFTConfig

MODEL = "Qwen/Qwen3-1.7B-Base"
OUT = "/workspace/final_model"

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
model.config.use_cache = False

ds = load_dataset("openai/gsm8k", "main", split="train")

USER_TMPL = (
    "Solve the following math problem step by step. The last line of your response should be of the form \"ANSWER: $ANSWER\" (without quotes) where $ANSWER is the answer to the problem.\n\n"
    "{q}\n\n"
    "Remember to put your answer on its own line at the end in the form \"ANSWER: $ANSWER\" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.\n\nReasoning:"
)

def to_text(ex):
    reasoning, final = ex["answer"].rsplit("####", 1)
    reasoning = reasoning.strip()
    final = final.strip().replace(",", "")
    user = USER_TMPL.format(q=ex["question"])
    assistant = reasoning + "\n\nANSWER: " + final
    return {"text": f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n{assistant}<|im_end|>"}

ds = ds.map(to_text, remove_columns=ds.column_names, num_proc=8)
print(ds[0]["text"][:500])

cfg = SFTConfig(
    output_dir="/workspace/ckpt",
    per_device_train_batch_size=32,
    gradient_accumulation_steps=1,
    num_train_epochs=2,
    learning_rate=1e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    logging_steps=40,
    save_strategy="no",
    bf16=True,
    max_length=768,
    packing=False,
    dataset_num_proc=8,
    report_to=[],
    seed=42,
)

trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds, processing_class=tok)
trainer.train()

im_end = tok.convert_tokens_to_ids("<|im_end|>")
endoftext = tok.convert_tokens_to_ids("<|endoftext|>")
model.config.eos_token_id = im_end
model.generation_config = GenerationConfig(eos_token_id=[im_end, endoftext], pad_token_id=endoftext)
trainer.save_model(OUT)
tok.save_pretrained(OUT)
print("DONE SAVING", flush=True)
