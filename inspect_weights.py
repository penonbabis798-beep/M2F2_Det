import torch
# path = "checkpoints/stage_2/mm_projector.bin"  # 或者是 pytorch_model.bin，取决于你有什么
path = "utils/weights/llava-pretrain/mm_projector.bin"  # 或者是 pytorch_model.bin，取决于你有什么
sd = torch.load(path, map_location="cpu")
print("Total keys:", len(sd))
for k, v in sd.items():
    print(f"{k}: {v.shape}")