import torch
# 检查原始文件
print("--- 原始文件键名 ---")
sd = torch.load("checkpoints/stage_2/mm_projector.bin", map_location="cpu")
for k in list(sd.keys())[:5]: print(k)

# 检查修复后的文件 (如果存在)
print("\n--- 修复后文件键名 ---")
try:
    sd_fix = torch.load("checkpoints/stage_2/mm_projector_fixed.bin", map_location="cpu")
    for k in list(sd_fix.keys())[:5]: print(k)
except:
    pass