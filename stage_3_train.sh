#!/bin/bash

# --- 1. 环境配置 ---
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# --- 2. 核心路径 ---
# 基础 LLM (Vicuna)
MODEL_VERSION="./utils/weights/vicuna-7b-v1.5"

# Stage 2 训练出来的 Projector 权重 (关键！)
# 注意：DeepSpeed 保存的可能是 mm_projector.bin 或 pytorch_model.bin
# 这里先指向 Stage 2 的输出目录，代码通常会自动寻找
# 修改这一行
PRETRAIN_ADAPTER="./checkpoints/stage_2/mm_projector_fixed.bin"

# Stage 1 Deepfake Encoder 权重
STAGE1_WEIGHTS="./checkpoints/stage_1/stage1_best.pth"

# 数据路径 (Stage 3 通常使用全量数据或特定指令数据)
# 如果你有专门的 Stage 3 json，请替换这里；否则沿用 Stage 2 的
DATA_PATH="./utils/DDVQA_split/c40/train_DDVQA_format.json"
IMAGE_FOLDER="./utils/DDVQA_images/c40/images_all"

# --- 3. 启动指令 ---
# 注意变化：
# 1. --freeze_backbone False  <-- 解冻 LLM，开始训练大脑
# 2. --tune_mm_mlp_adapter True <-- 继续微调连接层
# 3. --learning_rate 通常比 Stage 2 低 (例如 2e-5)
deepspeed llava/train/train_deepfake.py \
    --deepspeed scripts/zero2.json \
    --model_name_or_path $MODEL_VERSION \
    --version v1 \
    --data_path $DATA_PATH \
    --image_folder $IMAGE_FOLDER \
    --vision_tower ./utils/weights/vision_tower.pth \
    --pretrain_mm_mlp_adapter $PRETRAIN_ADAPTER \
    --deepfake_ckpt_path $STAGE1_WEIGHTS \
    --tune_deepfake_mlp_adapter True \
    --tune_mm_mlp_adapter True \
    --freeze_mm_mlp_adapter False \
    --freeze_backbone False \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_vision_select_feature cls_patch \
    --mm_use_im_start_end False \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/stage_3 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    --report_to tensorboard