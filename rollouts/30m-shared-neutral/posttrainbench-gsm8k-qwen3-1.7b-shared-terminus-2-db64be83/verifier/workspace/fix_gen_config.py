from transformers import GenerationConfig
g = GenerationConfig.from_pretrained("/workspace/final_model")
g.eos_token_id = [151645, 151643]
g.pad_token_id = 151643
g.temperature = None
g.top_p = None
g.top_k = None
g.do_sample = False
g.save_pretrained("/workspace/final_model")
print("DONE_FIX")
