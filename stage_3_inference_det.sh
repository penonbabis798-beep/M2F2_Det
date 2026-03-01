#!/bin/bash
source ~/.bashrc

current_path=$(pwd)
export PYTHONPATH="$current_path:$PYTHONPATH"

CUDA_NUM=0
CUDA_VISIBLE_DEVICES=$CUDA_NUM python -m llava.serve.cli_DDVQA_det \
    --model-path /data/tangchengwen/Deepfake视频检测/M2F2_Det/checkpoints/deepfake-M2F2-Det-Final