#!/bin/bash
set -e
cd /workspace
python train_sft.py
rm -rf /workspace/final_model
mkdir -p /workspace/final_model
cp /workspace/sft_out/config.json /workspace/sft_out/generation_config.json /workspace/sft_out/model.safetensors /workspace/final_model/
cp /workspace/sft_out/tokenizer.json /workspace/sft_out/tokenizer_config.json /workspace/sft_out/vocab.json /workspace/sft_out/merges.txt /workspace/sft_out/special_tokens_map.json /workspace/final_model/ 2>/dev/null || true
ls -la /workspace/final_model
echo ALL_DONE
