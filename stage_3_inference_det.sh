#!/bin/bash
source ~/.bashrc

current_path=$(pwd)
export PYTHONPATH="$current_path:$PYTHONPATH"

# --- 关键修改：强制离线模式，直接使用本地缓存的 CLIP 模型 ---
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# 指定运行的显卡
CUDA_NUM=0
CUDA_VISIBLE_DEVICES=$CUDA_NUM python -m llava.serve.cli_DDVQA_det \
    --model-path ./checkpoints/llava-M2F2-Det-Final