import json
import os

# 你的 Vicuna 权重路径
config_path = "./utils/weights/vicuna-7b-v1.5/config.json"

if not os.path.exists(config_path):
    print(f"错误: 找不到文件 {config_path}")
    exit(1)

print(f"正在读取: {config_path}")
with open(config_path, 'r') as f:
    config = json.load(f)

# --- 添加/修正 LLaVA 参数 ---
updates = {
    "mm_vision_tower": "openai/clip-vit-large-patch14-336",
    "mm_hidden_size": 1024,
    "mm_projector_type": "mlp2x_gelu",
    "mm_vision_select_layer": -2,
    "mm_vision_select_feature": "cls_patch",  # <--- 关键修改：必须是 cls_patch
    "mm_use_im_start_end": False,
    "mm_use_im_patch_token": False,
    "mm_patch_merge_type": "flat"
}

print("正在更新参数:")
for k, v in updates.items():
    # 强制覆盖，确保修正生效
    print(f"Set {k}: {v}")
    config[k] = v

# 保存回文件
with open(config_path, 'w') as f:
    json.dump(config, f, indent=2)

print("\n成功！config.json 已修正为 cls_patch 模式。")