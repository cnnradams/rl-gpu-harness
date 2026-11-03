#!/bin/bash
set -e
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /workspace
python -c "import accelerate" 2>/dev/null || pip install -q accelerate
python prep_data.py
python train_sft.py
