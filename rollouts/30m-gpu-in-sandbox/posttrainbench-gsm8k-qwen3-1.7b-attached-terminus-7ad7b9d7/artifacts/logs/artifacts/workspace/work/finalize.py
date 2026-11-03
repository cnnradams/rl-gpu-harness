import json, shutil, os
src = '/workspace/work/final_raw'
dst = '/workspace/final_model'
if os.path.exists(dst):
    shutil.rmtree(dst)
shutil.copytree(src, dst)
for fn in ['config.json', 'generation_config.json']:
    p = os.path.join(dst, fn)
    if os.path.exists(p):
        d = json.load(open(p))
        d['eos_token_id'] = [151643, 151645]
        json.dump(d, open(p, 'w'), indent=2)
print('finalized')
