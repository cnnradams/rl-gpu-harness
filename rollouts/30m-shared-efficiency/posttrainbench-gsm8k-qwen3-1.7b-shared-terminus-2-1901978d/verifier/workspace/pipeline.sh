#!/bin/bash
set -x
cd /workspace
python3 prep_data.py > prep.log 2>&1
python3 /home/agent/contamination_check.py --reference /home/agent/test_data.json --input train_texts.txt --input-format text > contam.out 2>&1
echo "contam_exit=$?" >> prep.log
python3 train.py > train.log 2>&1
echo DONE > pipeline.done
