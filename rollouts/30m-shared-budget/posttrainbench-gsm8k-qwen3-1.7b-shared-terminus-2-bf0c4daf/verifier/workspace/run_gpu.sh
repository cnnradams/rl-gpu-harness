#!/bin/bash
exec > >(tee /workspace/run.log) 2>&1
set -e
cd /workspace
echo "=== train start $(date) ==="
python train_sft.py
echo "=== evaluate start $(date) ==="
python evaluate.py --model-path final_model --limit 150 --max-connections 16 --gpu-memory-utilization 0.85 --json-output-file result_150.json > eval_150.log 2>&1 || true
tail -20 eval_150.log || true
cat result_150.json 2>/dev/null || true
echo "=== DONE $(date) ==="
