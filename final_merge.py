import torch
import os
from transformers import AutoTokenizer
from peft import PeftModel
from llava.model.language_model.llava_llama import LlavaLlamaForCausalLMDeepfake

# --- 路径配置 ---
# 1. 刚才用的底座
model_base_path = "./checkpoints/llava-v1.5-7b-deepfake-stage-2-merged"
# 2. Stage 3 训练出的 LoRA 目录
lora_path = "./checkpoints/stage_3"
# 3. 最终模型保存位置
save_path = "./checkpoints/M2F2-Det-Final"

print(f"🔄 1. 加载底座模型: {model_base_path}")
tokenizer = AutoTokenizer.from_pretrained(model_base_path, use_fast=False)
model = LlavaLlamaForCausalLMDeepfake.from_pretrained(
    model_base_path,
    low_cpu_mem_usage=False, # 关掉以避免兼容性报错
    torch_dtype=torch.float16
)

print(f"🔄 2. 加载 LoRA 权重: {lora_path}")
model = PeftModel.from_pretrained(model, lora_path)

print("⚡ 3. 正在合并 LoRA 到底座 (Merge and Unload)...")
model = model.merge_and_unload()

# --- 关键：加载 Stage 3 微调过的非 LoRA 参数 (Projectors) ---
# 在 Stage 3 中，我们设定了 tune_deepfake_mlp_adapter=True 和 tune_mm_mlp_adapter=True
# 这些层的权重不包含在 LoRA 里，而是保存在 non_lora_trainables.bin 或类似的 bin 文件中
# 我们需要找到并覆盖它们，否则 Projector 还是 Stage 2 的旧版本
print("🔍 4. 检查并加载更新的 Projector 权重...")

# 常见的非 LoRA 权重文件名
possible_files = [
    "non_lora_trainables.bin",
    "mm_projector.bin",
    "pytorch_model.bin" # 有时候 DeepSpeed 只存这个
]

projector_weights = None
for fname in possible_files:
    fpath = os.path.join(lora_path, fname)
    if os.path.exists(fpath):
        print(f"   ✅ 发现权重文件: {fname}")
        projector_weights = torch.load(fpath, map_location="cpu")
        break

if projector_weights is not None:
    # 智能键名匹配（去除可能的 deepfake_encoder 前缀，因为这里加载的是 LLM 部分）
    new_state_dict = {}
    for k, v in projector_weights.items():
        # 移除 'module.' 前缀（如果是 DDP 产生的）
        key = k.replace("module.", "")
        # 如果是 base_model.model. 这种 PEFT 前缀，也要去掉
        key = key.replace("base_model.model.", "")
        
        # 确保可以直接加载进 model
        new_state_dict[key] = v
        
    print(f"   📦 正在覆盖更新 {len(new_state_dict)} 个 Projector 张量...")
    m, u = model.load_state_dict(new_state_dict, strict=False)
    print(f"   更新完成。Unexpected keys (应为0): {len(u)}")
else:
    print("⚠️ 警告：在 stage_3 目录中未找到独立的 Projector 权重文件。")
    print("   如果你的 LoRA checkpoint 里已经包含了所有权重，则忽略此警告。")

# --- 保存 ---
print(f"💾 5. 保存最终模型到: {save_path}")
if not os.path.exists(save_path):
    os.makedirs(save_path)

model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

print("🎉 恭喜！M2F2-Det 最终模型制作完成！")