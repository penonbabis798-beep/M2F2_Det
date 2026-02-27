import torch
import os

# 路径配置
input_path = "./checkpoints/stage_2/mm_projector.bin"
output_path = "./checkpoints/stage_2/mm_projector_final.bin"

print(f"正在加载: {input_path}")
state_dict = torch.load(input_path, map_location="cpu")

new_state_dict = {}
print("开始转换键名...")

for k, v in state_dict.items():
    # 核心逻辑：把 deepfake_projector 改为 mm_projector
    if "deepfake_projector" in k:
        new_k = k.replace("deepfake_projector", "mm_projector")
        new_state_dict[new_k] = v
        print(f"  [修改] {k} -> {new_k}")
    # 防止本身已经是 mm_projector 的情况被漏掉
    elif "mm_projector" in k:
        new_state_dict[k] = v
        print(f"  [保留] {k}")
    else:
        # 如果还有其他情况，为了保险起见，我们可以试着强制加上前缀
        # 但通常 stage 2 只保存了 projector，所以这里主要处理上述情况
        print(f"  [跳过] {k} (不包含目标关键词)")

print(f"正在保存到: {output_path}")
torch.save(new_state_dict, output_path)
print("✅ 转换完成！请在 Stage 3 脚本中使用 mm_projector_final.bin")