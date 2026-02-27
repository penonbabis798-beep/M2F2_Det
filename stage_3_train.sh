#!/bin/bash

# --- 1. 环境配置 ---
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# --- 2. 核心路径 ---
# [关键] 加载刚刚合并好的新底座！
MODEL_VERSION="./checkpoints/llava-v1.5-7b-deepfake-stage-2-merged"

# [关键] 加载官方 CLIP Projector (通用视觉能力)
PRETRAIN_ADAPTER="./utils/weights/llava-pretrain/mm_projector.bin"

# [关键 - 补回] Stage 1 训练好的 Encoder 权重
STAGE1_WEIGHTS="./checkpoints/stage_1/stage1_best.pth"

DATA_PATH="./utils/DDVQA_split/c40/train_DDVQA_format.json"
IMAGE_FOLDER="./utils/DDVQA_images/c40/images_all"

# --- 3. 启动指令 ---
# 策略：ZeRO-2 + LoRA
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
    --lora_enable True \
    --lora_r 128 \
    --lora_alpha 256 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_vision_select_feature cls_patch \
    --mm_use_im_start_end False \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/stage_3 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
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