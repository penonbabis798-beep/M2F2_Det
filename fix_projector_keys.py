import torch
import os

# 路径配置
input_path = "./checkpoints/stage_2/mm_projector.bin"
output_path = "./checkpoints/stage_2/mm_projector_fixed.bin"

print(f"正在加载: {input_path}")
state_dict = torch.load(input_path, map_location="cpu")

new_state_dict = {}
for k, v in state_dict.items():
    # 去掉常见的前缀
    new_k = k.replace("model.mm_projector.", "").replace("mm_projector.", "")
    new_state_dict[new_k] = v
    print(f"转换键名: {k} -> {new_k}")

print(f"正在保存到: {output_path}")
torch.save(new_state_dict, output_path)
print("完成！请在 Stage 3 脚本中使用新文件。")