import json
p = '/workspace/final_model/tokenizer_config.json'
cfg = json.load(open(p))
atd = cfg.get('added_tokens_decoder', {})
for tid, t in atd.items():
    if t.get('content') == '<|im_end|>':
        t['special'] = False
json.dump(cfg, open(p, 'w'), indent=2)
print('patched tokenizer_config: im_end special=False')
