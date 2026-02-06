#!/bin/bash
source ~/.bashrc
# 如果你的 conda 环境名是 m2f2，请保持不变
eval "$(conda shell.bash hook)"
conda activate m2f2

# --- 核心配置 ---
# 1. 启用 0,1,2,3 四张显卡
export CUDA_VISIBLE_DEVICES=0,1,2,3

# 2. 强制 HuggingFace 离线模式 (避免联网报错)
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# 3. 显存分配优化
export PYTORCH_ALLOC_CONF=expandable_segments:True

# 4. 路径设置
current_path=$(pwd)
export PYTHONPATH="$current_path:$PYTHONPATH"

# --- 启动训练 ---
echo "开始 4 卡训练 (Batch Size = 160)..."
# 使用 tee 同时输出到屏幕和日志文件
python -u stage_1_detection.py 2>&1 | tee outputs/train_stage_1_4gpu_$(date +%Y%m%d_%H%M%S).log