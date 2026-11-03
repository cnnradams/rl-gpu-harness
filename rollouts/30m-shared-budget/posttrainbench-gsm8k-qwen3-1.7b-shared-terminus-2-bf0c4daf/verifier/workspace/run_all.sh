#!/bin/bash
set -e
cd /workspace
echo "=== prep data ==="
python prep_data.py
echo "=== contamination check ==="
python /home/agent/contamination_check.py --reference /home/agent/test_data.json --input train_data_raw.jsonl --text-field text > flagged.jsonl || true
wc -l flagged.jsonl || true
python filter_contaminated.py train_data_raw.jsonl flagged.jsonl train_data.jsonl
wc -l train_data.jsonl
echo "=== train ==="
python train_sft.py
echo "=== evaluate ==="
python evaluate.py --model-path final_model --limit 150 --max-connections 16 --gpu-memory-utilization 0.85 --json-output-file result_150.json > eval_150.log 2>&1 || true
tail -20 eval_150.log || true
cat result_150.json 2>/dev/null || true
echo "=== DONE ==="
