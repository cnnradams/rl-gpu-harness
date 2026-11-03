#!/bin/bash
set -x
cd /workspace
python3 train_sft.py > train.log 2>&1
echo TRAIN_EXIT=$?
python3 fix_gen_config.py >> train.log 2>&1
echo FIX_EXIT=$?
echo SKIP_EVAL
echo EVAL_EXIT=$?
tail -5 /workspace/metrics.json 2>/dev/null
echo ALL_DONE
