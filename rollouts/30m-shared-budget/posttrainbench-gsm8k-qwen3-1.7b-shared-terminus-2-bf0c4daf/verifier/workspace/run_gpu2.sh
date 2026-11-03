#!/bin/bash
exec > >(tee /workspace/run2.log) 2>&1
set -e
cd /workspace
echo "=== train2 start $(date) ==="
python train_sft.py
python patch_tok.py
echo "=== evaluate2 start $(date) ==="
python evaluate.py --model-path final_model --limit 150 --max-connections 24 --gpu-memory-utilization 0.85 --json-output-file result2_150.json > eval2_150.log 2>&1 || true
tail -5 eval2_150.log || true
cat result2_150.json 2>/dev/null || true
echo "=== DONE2 $(date) ==="
