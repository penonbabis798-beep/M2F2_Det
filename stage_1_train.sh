#!/bin/bash
source ~/.bashrc
conda activate m2f2

# 设置显卡
export CUDA_VISIBLE_DEVICES=0,1
# 关键：优化显存分配，防止碎片化
export PYTORCH_ALLOC_CONF=expandable_segments:True

# 路径设置
current_path=$(pwd)
export PYTHONPATH="$current_path:$PYTHONPATH"
export HF_ENDPOINT=https://hf-mirror.com

python -u stage_1_detection.py 2>&1 | tee outputs/train_stage_1_$(date +%Y%m%d_%H%M%S).log